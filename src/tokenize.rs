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
    pub pos: String,
    pub pos1: String,
    pub conj_type: String,
    pub conj_form: String,
    pub base: String,
    /// 読み (カタカナ)。同音異字の候補引きに使う。未知語は空。
    pub reading: String,
}

impl Token {
    pub fn is_num(&self) -> bool {
        self.pos1 == "数"
            && self
                .surface
                .chars()
                .all(|c| c.is_ascii_digit() || c == ',' || c == '.')
    }

    pub fn key(&self) -> &str {
        if self.is_num() {
            NUM_KEY
        } else {
            &self.surface
        }
    }
}

// 辞書はプロセスで 1 つだけ読み、ワーカーがそれを 'static で借りる
static DICT: OnceLock<ZigTokenizer> = OnceLock::new();

thread_local! {
    static WORKER: RefCell<Option<ZigWorker<'static>>> = const { RefCell::new(None) };
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
                let mut f = v.feature().split(',');
                let mut next = || f.next().filter(|s| *s != "*").unwrap_or("").to_string();
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
                let tok = Token {
                    surface: surface.to_string(),
                    start: v.start_char,
                    end: v.end_char,
                    pos,
                    pos1,
                    conj_type,
                    conj_form,
                    base,
                    reading,
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
