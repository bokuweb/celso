//! delarocha (Zig コア + IPADIC) による分かち書きと、言語モデル用のキー化。
//!
//! 辞書は MeCab 形式の生ファイル (UTF-8 化した mecab-ipadic) から一度だけバイナリへ変換し
//! (`celso build-dict`)、起動時はそれを mmap する。ワーカーはスレッドごとに 1 つ持つ。

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result};
use delarocha::ffi::{ZigTokenizer, ZigWorker};

/// 数字列は 1 トークンにまとめて、このキーで言語モデルに渡す (「第34号」と「第5号」を同一視する)。
pub const NUM_KEY: &str = "<num>";

/// 既定の辞書パス。環境変数 CELSO_DIC で上書きできる。
pub fn default_dict_path() -> PathBuf {
    std::env::var_os("CELSO_DIC")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/ipadic.dic"))
}

/// MeCab 形式の生辞書 (lex.csv / matrix.def / char.def / unk.def を含むディレクトリ) をバイナリへ変換する。
pub fn build_dict(raw_dir: &Path, out: &Path) -> Result<()> {
    ZigTokenizer::write_binary_from_raw_paths(
        raw_dir.join("lex.csv"),
        raw_dir.join("matrix.def"),
        raw_dir.join("char.def"),
        raw_dir.join("unk.def"),
        out,
    )?;
    Ok(())
}

#[derive(Debug, Clone)]
pub struct Token {
    pub surface: String,
    /// 入力テキスト内の文字オフセット (半開区間)。
    pub start: usize,
    pub end: usize,
    // 品詞・活用の情報は種類が少ないので共有の文字列 (intern) にし、トークンごとに確保しない
    pub pos: &'static str,
    pub pos1: &'static str,
    pub conj_type: &'static str,
    pub conj_form: &'static str,
    /// 原形。活用語 (動詞・形容詞・助動詞) だけ持つ (活用候補の表引きにしか使わないため)。
    pub base: &'static str,
    /// 読み (カタカナ)。同音異字の候補引きにしか使わないので、漢字を含む内容語だけ持つ。
    pub reading: &'static str,
}

impl Token {
    pub fn is_num(&self) -> bool {
        self.pos1 == "数"
            && self
                .surface
                .chars()
                .all(|c| c.is_ascii_digit() || c == ',' || c == '.')
    }

    /// 語彙外の語の代わりに使う品詞クラス。活用語は活用形まで含める
    /// (「<動詞-自立-連用形>」)。助詞・活用の誤りを見るのに、内容語の表層形までは要らない。
    pub fn class_key(&self) -> String {
        if self.conj_form.is_empty() {
            format!("<{}-{}>", self.pos, self.pos1)
        } else {
            format!("<{}-{}-{}>", self.pos, self.pos1, self.conj_form)
        }
    }

    pub fn key(&self) -> &str {
        if self.is_num() {
            NUM_KEY
        } else {
            &self.surface
        }
    }
}

/// 品詞・活用形・原形の文字列を共有する。種類は辞書の範囲に限られる (数千程度) ので、
/// 一度確保したものを使い回してトークンごとの確保をなくす。
fn intern(s: &str) -> &'static str {
    use std::sync::RwLock;
    static TABLE: OnceLock<RwLock<rustc_hash::FxHashSet<&'static str>>> = OnceLock::new();
    if s.is_empty() || s == "*" {
        return "";
    }
    let t = TABLE.get_or_init(Default::default);
    if let Some(v) = t.read().unwrap().get(s) {
        return v;
    }
    let mut w = t.write().unwrap();
    if let Some(v) = w.get(s) {
        return v;
    }
    let v: &'static str = Box::leak(s.to_string().into_boxed_str());
    w.insert(v);
    v
}

fn has_kanji(s: &str) -> bool {
    s.chars()
        .any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c) || c == '々')
}

// 辞書はプロセスで 1 つだけ読み、ワーカーがそれを 'static で借りる
static DICT: OnceLock<ZigTokenizer> = OnceLock::new();

thread_local! {
    static WORKER: RefCell<Option<ZigWorker<'static>>> = const { RefCell::new(None) };
    /// 語 ID → 解析済みの品詞情報。同じ語は同じ feature を持つので、文字列の分割と intern を
    /// 語ごとに 1 回で済ませる (トークンの組み立てが分かち書き本体より重かったため)。
    static FEATS: RefCell<rustc_hash::FxHashMap<u32, Feat>> = RefCell::new(Default::default());
}

/// キャッシュの上限 (語の種類数)。長時間動くサーバーで際限なく増えないよう、超えたら捨てる。
const FEAT_CACHE_LIMIT: usize = 100_000;

#[derive(Clone, Copy)]
struct Feat {
    pos: &'static str,
    pos1: &'static str,
    conj_type: &'static str,
    conj_form: &'static str,
    base: &'static str,
    /// 漢字を含む内容語の読み (それ以外は空)。未知語は surface 依存なので使わない
    reading: &'static str,
}

fn parse_feat(feature: &str, surface: &str) -> Feat {
    let mut f = feature.split(',');
    let mut next = || f.next().filter(|s| *s != "*").unwrap_or("");
    let (pos, pos1, _p2, _p3, conj_type, conj_form, base, reading) = (
        next(),
        next(),
        next(),
        next(),
        next(),
        next(),
        next(),
        next(),
    );
    let pos = intern(pos);
    let content = matches!(pos, "名詞" | "動詞" | "形容詞" | "副詞");
    Feat {
        pos,
        pos1: intern(pos1),
        conj_type: intern(conj_type),
        conj_form: intern(conj_form),
        base: if conj_type.is_empty() {
            ""
        } else {
            intern(base)
        },
        reading: if content && has_kanji(surface) {
            intern(reading)
        } else {
            ""
        },
    }
}

#[derive(Clone, Copy)]
pub struct Tokenizer {
    dict: &'static ZigTokenizer,
}

impl Tokenizer {
    /// 既定パスの辞書で作る。
    pub fn new() -> Result<Self> {
        Self::from_path(&default_dict_path())
    }

    pub fn from_path(path: &Path) -> Result<Self> {
        if DICT.get().is_none() {
            let t = ZigTokenizer::from_binary_path(path)
                .with_context(|| format!("辞書 {path:?} を開けない (celso build-dict で作る)"))?;
            let _ = DICT.set(t);
        }
        Ok(Self {
            dict: DICT.get().unwrap(),
        })
    }

    /// `text` は [`crate::norm::norm`] 済みであること。空白トークンは落とす。
    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        WORKER.with(|w| {
            let mut w = w.borrow_mut();
            if w.is_none() {
                *w = self.dict.create_worker().ok();
            }
            let Some(worker) = w.as_mut() else {
                return Vec::new();
            };
            let Ok(views) = worker.tokenize_borrowed_views(text) else {
                return Vec::new();
            };
            let mut out: Vec<Token> = Vec::with_capacity(views.len());
            for v in views.iter() {
                let surface = v.surface();
                if surface.trim().is_empty() {
                    continue;
                }
                // 未知語の feature は文字種ごとに共通なので語 ID ではキャッシュできるが、
                // 読みは持たない (has_kanji の判定が surface 依存になるため、未知語は毎回解析する)
                let wid = v.word_id();
                let feat = if v.is_unknown() {
                    parse_feat(v.feature(), surface)
                } else {
                    FEATS.with(|c| {
                        let mut c = c.borrow_mut();
                        if let Some(f) = c.get(&wid) {
                            return *f;
                        }
                        if c.len() >= FEAT_CACHE_LIMIT {
                            c.clear();
                        }
                        let f = parse_feat(v.feature(), surface);
                        c.insert(wid, f);
                        f
                    })
                };
                let tok = Token {
                    surface: surface.to_string(),
                    start: v.start_char,
                    end: v.end_char,
                    pos: feat.pos,
                    pos1: feat.pos1,
                    conj_type: feat.conj_type,
                    conj_form: feat.conj_form,
                    base: feat.base,
                    reading: feat.reading,
                };
                // 「1」「,」「500」のように割れた数字列を 1 トークンへまとめる
                if let Some(prev) = out.last_mut()
                    && prev.is_num()
                    && prev.end == tok.start
                    && tok.is_num()
                {
                    prev.surface.push_str(&tok.surface);
                    prev.end = tok.end;
                    continue;
                }
                out.push(tok);
            }
            out
        })
    }
}
