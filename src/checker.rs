//! 修正候補を作って言語モデルで比べるチェッカー。
//!
//! 単語としては正しいが並びがおかしい誤り (余計な助詞・助詞の取り違え・活用の誤り) を狙う。
//! 各位置で「よくある誤りを元に戻す編集」を候補として作り、編集の前後で
//! 影響を受ける範囲 (編集箇所 + 後続 order-1 語) の対数確率を比べる。
//! 改善幅 Δ (log10) が種類ごとの閾値を超えたものを指摘する。

use rustc_hash::FxHashMap;

use crate::lm::{BOS, EOS, LanguageModel, UNK};
use crate::mlm::{Mlm, Query};
use crate::norm::norm;
use crate::tokenize::{Token, Tokenizer};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EditKind {
    /// 余計な助詞などを削る (宿泊から施設 → 宿泊施設)
    Delete,
    /// 助詞を別の助詞へ (西口側までは → 西口側には)
    Substitute,
    /// 活用形を直す (多くあろう → 多くある)
    Inflection,
    /// 脱落した助詞を補う
    Insert,
    /// 同音異字 (対象 → 対照)
    Homophone,
    /// 文字単位の衍字・脱字・転字
    Char,
    /// 実際の誤字から集めた書き換えパターン (をを → を、れいる → れている)。[`crate::patterns`]
    Pattern,
}

impl EditKind {
    pub fn label(self) -> &'static str {
        match self {
            EditKind::Delete => "delete",
            EditKind::Substitute => "substitute",
            EditKind::Inflection => "inflection",
            EditKind::Insert => "insert",
            EditKind::Homophone => "homophone",
            EditKind::Char => "char",
            EditKind::Pattern => "pattern",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Finding {
    /// 元テキストの文字オフセット (半開区間)。挿入は start == end。
    pub start: usize,
    pub end: usize,
    pub original: String,
    pub replacement: String,
    pub kind: EditKind,
    /// 最終スコア (log10 換算の改善幅。MLM 併用時は n-gram と MLM の加重和)。
    pub delta: f32,
    /// 同じ箇所の別の修正案 (スコア順、最良案を含まない)。
    pub alternatives: Vec<Suggestion>,
}

/// 別の修正案。範囲は最良案と違うことがある (「まで」を消す案と「まで」→「に」の案など)。
#[derive(Debug, Clone)]
pub struct Suggestion {
    pub start: usize,
    pub end: usize,
    pub original: String,
    pub replacement: String,
    pub kind: EditKind,
    pub score: f32,
}

/// 文書の種類。閾値の組を切り替える。
///
/// 法令文向けの閾値は、文書内繰り返しの抑制と例規集コーパスで誤検出が少ないぶん低めにしてある。
/// 一般文ではコーパスの網羅が薄く、そのままだと誤検出が増えるので、高めの組を使う。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Domain {
    Legal,
    General,
    /// 契約書 (モデル契約の解説文を含む)。法令文より言い回しが多様で、一般文より法令用語が多い
    Contract,
}

/// 法令文らしさで文書の種類を推定する。条・項・号の参照や法令用語が 1,000 字あたり 2 回以上なら法令文。
pub fn detect_domain(text: &str) -> Domain {
    let chars = text.chars().count().max(1);
    let mut hits = 0usize;
    for pat in [
        "又は",
        "若しくは",
        "当該",
        "の規定",
        "に掲げる",
        "に定める",
        "ものとする",
    ] {
        hits += text.matches(pat).count();
    }
    // 「第N条」「第N項」「第N号」(N は数字列)
    for (i, _) in text.match_indices('第') {
        let rest = &text[i + '第'.len_utf8()..];
        let digits = rest
            .chars()
            .take_while(|c| {
                c.is_ascii_digit()
                    || ('０'..='９').contains(c)
                    || "一二三四五六七八九十百".contains(*c)
            })
            .count();
        if digits > 0 {
            let after: String = rest.chars().skip(digits).take(1).collect();
            if matches!(after.as_str(), "条" | "項" | "号") {
                hits += 1;
            }
        }
    }
    // 契約書らしさ: 当事者の呼び方と「本契約」。条番号や法令用語もあるので、法令文より先に判定する
    let mut contract = 0usize;
    for pat in [
        "甲",
        "乙",
        "本契約",
        "委託者",
        "受託者",
        "発注者",
        "受注者",
        "当事者",
        "貸主",
        "借主",
        "ユーザ",
        "ベンダ",
    ] {
        contract += text.matches(pat).count();
    }
    if contract * 1000 >= chars * 3 {
        Domain::Contract
    } else if hits * 1000 >= chars * 2 {
        Domain::Legal
    } else {
        Domain::General
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    /// 法令文向けの閾値
    pub thresholds: FxHashMap<EditKind, f32>,
    /// 一般文向けの閾値
    pub general_thresholds: FxHashMap<EditKind, f32>,
    /// 契約書向けの閾値
    pub contract_thresholds: FxHashMap<EditKind, f32>,
    /// 文書の種類を固定する (None なら文書ごとに自動判定)
    pub domain: Option<Domain>,
    pub enable_insert: bool,
    /// 元の文の編集箇所まわりに、この次数の n-gram として「未出現」の並びがあるときだけ指摘する。
    /// 正しい文どうしの言い換え (「期間が」⇔「期間の」) を拾わないための足切り。0 で無効。
    pub novelty_order: usize,
    /// 指摘箇所の前後を含む文字列が文書内にこの回数以上あれば指摘しない (0 で無効)。
    pub doc_repeat_limit: usize,
    /// 同音異字の候補に足す共起の差の重み (log10 換算の Δ に、PMI 差 × 重み / ln10 を足す)。
    pub cooc_weight: f32,
    /// 文法モデル (内容語を品詞クラスにまとめた高次 n-gram) の Δ に掛ける重み。助詞の削除・置換・補いにだけ足す。
    pub aux_weight: f32,
    /// MLM の重み (最終スコア = n-gram Δ + mlm_weight × MLM Δ)。
    pub mlm_weight: f32,
    /// MLM 併用時、1 段目は閾値からこの幅だけ下の候補まで残して 2 段目に回す。
    pub stage1_slack: f32,
    /// 1 箇所あたり 2 段目に回す候補数。
    pub top_k: usize,
    /// MLM で採点する範囲 (編集箇所の前後この文字数に掛かるサブワード)。
    pub mlm_margin: usize,
    /// MLM の長さ補正 (サブワード 1 つあたりの nats)。
    pub mlm_length_penalty: f32,
    /// n-gram の最良スコアが「閾値 + これ」以上で、かつ 2 位との差が mlm_gap 以上なら MLM を使わずに確定する
    /// (速度のため。MLM は判断が割れる箇所だけに使う)。
    pub mlm_band: f32,
    pub mlm_gap: f32,
    /// true なら PLL (高精度・低速)、false なら穴埋め採点 (既定・高速)。
    pub mlm_pll: bool,
    /// true なら MLM は「確定した箇所の修正案選び」だけに使う (既定)。false なら採否にも使う。
    pub mlm_choice_only: bool,
}

impl Default for Config {
    fn default() -> Self {
        let mut t = FxHashMap::default();
        // 配布モデル (語彙 1 万・3-gram・強い足切り) 向けに、市税条例の人工誤り (調整用シード) で
        // 原文での誤検出が 1 万字あたり 1 件弱になるよう決めた値
        t.insert(EditKind::Delete, 4.0);
        t.insert(EditKind::Substitute, 4.5);
        t.insert(EditKind::Inflection, 1.5);
        t.insert(EditKind::Insert, 3.5);
        t.insert(EditKind::Homophone, 4.0);
        // 文字単位の編集は遅く誤検出も多いので既定では無効 (README 参照)
        t.insert(EditKind::Char, f32::INFINITY);
        t.insert(EditKind::Pattern, 0.0);
        Self {
            // 一般文: JWTD の gold (開発用) で種類ごとに決めた値。活用と取り違えは
            // 一般文で正しい言い換えを拾いやすいので、法令文より大きく上げる
            general_thresholds: [
                (EditKind::Delete, 4.5),
                (EditKind::Substitute, 5.5),
                (EditKind::Inflection, 3.0),
                (EditKind::Insert, 4.0),
                (EditKind::Homophone, 4.5),
                (EditKind::Char, f32::INFINITY),
                (EditKind::Pattern, 0.0),
            ]
            .into_iter()
            .collect(),
            // 契約書: JEITA のモデル契約 (解説付き、開発用) で決めた値
            contract_thresholds: [
                (EditKind::Delete, 5.0),
                (EditKind::Substitute, 6.0),
                (EditKind::Inflection, 3.5),
                (EditKind::Insert, 4.5),
                (EditKind::Homophone, 5.0),
                (EditKind::Char, f32::INFINITY),
                (EditKind::Pattern, 0.0),
            ]
            .into_iter()
            .collect(),
            thresholds: t,
            domain: None,
            enable_insert: true,
            novelty_order: 3,
            doc_repeat_limit: 2,
            mlm_weight: 1.0,
            cooc_weight: 1.0,
            aux_weight: 0.0,
            stage1_slack: 1.5,
            top_k: 4,
            mlm_margin: 1,
            mlm_length_penalty: 2.0,
            mlm_band: 2.0,
            mlm_gap: 1.0,
            mlm_pll: false,
            mlm_choice_only: true,
        }
    }
}

/// 置換候補にする助詞 (単独トークンになるもの)。
const PARTICLES: &[&str] = &[
    "が", "の", "を", "に", "へ", "と", "で", "から", "まで", "より", "は", "も", "や", "て", "ば",
    "し", "ので", "のに",
];
/// 法令文で使う複合助詞 (IPADIC では 1 語)。取り違えの置換先と、脱落の補完に使う。
const COMPOUND_PARTICLES: &[&str] = &[
    "について",
    "による",
    "により",
    "によって",
    "において",
    "に関する",
    "に対する",
    "に対して",
    "に係る",
];
/// 文字単位の脱字として補うかな。
const INSERT_KANA: &[&str] = &[
    "い", "て", "の", "に", "を", "が", "し", "た", "る", "な", "っ", "ん", "か", "と", "で", "は",
    "れ", "ら", "う", "く",
];

fn is_kanji(c: char) -> bool {
    ('\u{4E00}'..='\u{9FFF}').contains(&c) || c == '々'
}

fn is_kana(c: char) -> bool {
    ('ぁ'..='ゖ').contains(&c) || ('ァ'..='ヺ').contains(&c) || c == 'ー'
}

/// 挿入候補にする助詞。
const INSERT_PARTICLES: &[&str] = &[
    "の", "を", "に", "が", "は", "で", "と", "から", "まで", "も",
];

pub struct Checker {
    pub tok: Tokenizer,
    pub lm: Box<dyn LanguageModel>,
    pub cfg: Config,
    /// (原形, 活用型) → その語の活用した表層形の一覧
    inflections: FxHashMap<(String, String), Vec<String>>,
    /// 読み → (表層形, 出現数) 出現数の多い順
    readings: FxHashMap<String, Vec<(String, u32)>>,
    /// 2 段目のマスク言語モデル (無ければ n-gram だけで判定)
    mlm: Option<Mlm>,
    /// 同音異字の判定に使う文内共起モデル (無ければ n-gram だけで判定)
    cooc: Option<crate::cooc::Cooc>,
    /// 文法モデル (無ければ使わない)。機能語だけ表層形で持ち、内容語は品詞クラスにした高次 n-gram で、
    /// 単語 3-gram (前後 2 語) より長い範囲の助詞の並びを見る
    aux: Option<Box<dyn LanguageModel>>,
    /// 採否の判定器 (無ければ種類ごとの閾値で決める)
    rerank: Option<crate::rerank::Reranker>,
    /// 実際の誤字から集めた書き換えパターン (無ければ使わない)
    patterns: Option<crate::patterns::Patterns>,
    /// カタカナ語の出現数の表 (カタカナ語の打ち間違いの検出に使う。無ければ使わない)
    katakana: Option<crate::katakana::Katakana>,
    /// 文字単位の言語モデル (語の中の 1 字の誤りの検出に使う。無ければ使わない)
    charcheck: Option<crate::charcheck::CharChecker>,
    /// 文単位の結果キャッシュ (正規化済みの文のハッシュ → その文の指摘)。
    /// 指摘は文の中身だけで決まる (文書内繰り返しの抑制は文書全体で毎回かけ直す) ので、
    /// 編集されていない文は再計算しなくてよい。
    cache: Option<std::sync::RwLock<FxHashMap<u64, Vec<Finding>>>>,
    /// `CELSO_TRACE` が設定されていれば、閾値に届かなかった候補も含めて採点の内訳を stderr へ出す
    /// (ケースの調査用。組み込み先では設定しない)
    trace: bool,
}

/// キャッシュの上限 (文の数)。超えたら丸ごと捨てる (単純さ優先。1 文書は数千文程度)。
const CACHE_LIMIT: usize = 200_000;

fn sentence_key(s: &str, d: Domain) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    s.hash(&mut h);
    d.hash(&mut h);
    h.finish()
}

/// 判定器の特徴量の材料 (候補ごとに scan で求めた値)。
#[derive(Clone, Copy)]
struct Signals {
    /// n-gram の Δ
    ngram: f32,
    /// 文法モデルの Δ (助詞の削除・置換・補いだけ)
    aux: Option<f32>,
    /// 同音異字の共起の差 (同音異字だけ)
    cooc: Option<f32>,
    /// 編集箇所まわりの未出現の並びの数
    novel: usize,
    /// 文全体で未出現の並びになっている位置の割合 (崩れた文・コーパスに無い分野の文ほど大きい)
    sent_novel: f32,
}

/// 修正候補。置き換え先は常に 1 語以下なので、借用した文字列 1 つで持つ (候補ごとの確保をなくす)。
struct Cand<'a> {
    a: usize,
    b: usize,
    repl: Option<&'a str>,
    kind: EditKind,
}

impl Checker {
    pub fn new(
        tok: Tokenizer,
        lm: Box<dyn LanguageModel>,
        cfg: Config,
        inflections: FxHashMap<(String, String), Vec<String>>,
        readings: FxHashMap<String, Vec<(String, u32)>>,
    ) -> Self {
        Self {
            tok,
            lm,
            cfg,
            inflections,
            readings,
            mlm: None,
            cooc: None,
            aux: None,
            rerank: None,
            patterns: None,
            katakana: None,
            charcheck: None,
            cache: None,
            trace: std::env::var_os("CELSO_TRACE").is_some(),
        }
    }

    /// 同音異字の判定に文内共起モデルを使う。
    /// 文法モデルを使う (重みは [`Config::aux_weight`])。
    #[must_use]
    pub fn with_aux(mut self, aux: Box<dyn LanguageModel>) -> Self {
        self.aux = Some(aux);
        self
    }

    /// 採否を判定器で決める (閾値は判定器の文書種類ごとの値になる)。
    #[must_use]
    pub fn with_rerank(mut self, rerank: crate::rerank::Reranker) -> Self {
        self.rerank = Some(rerank);
        self
    }

    /// 文字単位の言語モデルで語の中の 1 字の誤りを拾う。
    #[must_use]
    pub fn with_charcheck(mut self, c: crate::charcheck::CharChecker) -> Self {
        self.charcheck = Some(c);
        self
    }

    /// 文字モデルを後から付ける (playground で、本体の準備ができてから読み込むため)。
    /// 結果が変わるので文単位のキャッシュは捨てる。
    ///
    /// # Panics
    /// キャッシュのロックが poison している場合 (別スレッドが検査中に panic した場合)。
    pub fn set_charcheck(&mut self, c: crate::charcheck::CharChecker) {
        self.charcheck = Some(c);
        if let Some(cache) = &self.cache {
            cache.write().unwrap().clear();
        }
    }

    /// カタカナ語の打ち間違いの検出を使う。
    #[must_use]
    pub fn with_katakana(mut self, katakana: crate::katakana::Katakana) -> Self {
        self.katakana = Some(katakana);
        self
    }

    /// 実際の誤字から集めた書き換えパターンを使う。
    #[must_use]
    pub fn with_patterns(mut self, patterns: crate::patterns::Patterns) -> Self {
        self.patterns = Some(patterns);
        self
    }

    pub fn with_cooc(mut self, cooc: crate::cooc::Cooc) -> Self {
        self.cooc = Some(cooc);
        self
    }

    /// 文単位の結果キャッシュを有効にする (2 回目以降の検査で、変わっていない文を再計算しない)。
    pub fn with_cache(mut self) -> Self {
        self.cache = Some(std::sync::RwLock::new(FxHashMap::default()));
        self
    }

    /// キャッシュ済みの文の数。
    pub fn cached_sentences(&self) -> usize {
        self.cache.as_ref().map_or(0, |c| c.read().unwrap().len())
    }

    pub fn with_mlm(mut self, mlm: Mlm) -> Self {
        self.mlm = Some(mlm);
        self
    }

    /// テキストを検査する。文 (句点・改行・空白) ごとに独立に処理する。
    pub fn check(&self, text: &str) -> Vec<Finding> {
        self.check_many(&[text]).pop().unwrap_or_default()
    }

    /// 文書全体を検査し、最後に文書内の繰り返しで誤検出を抑える。
    pub fn check_document(&self, text: &str) -> Vec<Finding> {
        let found = self.check(text);
        let normalized = norm(text);
        let chars: Vec<char> = normalized.chars().collect();
        let found: Vec<Finding> = found
            .into_iter()
            .filter(|f| self.doc_repeats(&normalized, &chars, f, 0) < self.cfg.doc_repeat_limit)
            .collect();
        drop_old_style_small_kana(found)
    }

    /// 複数のテキストをまとめて検査する。MLM の採点要求を全テキスト分まとめて大きなバッチで流すので、
    /// 1 つずつ `check` するより速い (評価用)。
    pub fn check_many(&self, texts: &[&str]) -> Vec<Vec<Finding>> {
        use rayon::prelude::*;
        let dbg = std::env::var_os("CELSO_DEBUG").is_some();
        // 時計はデバッグ表示のときだけ読む (wasm32-unknown-unknown では Instant::now が panic するため)
        let t0 = dbg.then(std::time::Instant::now);
        // (テキスト番号, 文の開始オフセット, 正規化済みの文, 文書の種類)
        let mut sents: Vec<(usize, usize, String, Domain)> = Vec::new();
        for (ti, text) in texts.iter().enumerate() {
            let normalized = norm(text);
            let dom = self
                .cfg
                .domain
                .unwrap_or_else(|| detect_domain(&normalized));
            let chars: Vec<char> = normalized.chars().collect();
            let mut s = 0;
            for i in 0..=chars.len() {
                // 空白は、その前がラベルらしい (平仮名を含まない 15 文字以内) ときだけ区切りにする。
                // 条例の「第41条　固定資産税は」のようにラベルと本文を空白で分ける書き方では、
                // ひと続きの文として採点すると「第41条」直後の語が不自然に見えてしまう。
                // 一方「期間が 1 年」のように文中の空白で切ると、「期間が」で文が終わったことになり、
                // 文末の助詞を削る誤検出になる (契約書で多かった)。
                // 「第29条の2」のように、ラベルに入る平仮名は「の」だけ
                let label_like = |seg: &[char]| {
                    seg.len() <= 15 && !seg.iter().any(|c| *c != 'の' && ('ぁ'..='ゖ').contains(c))
                };
                let end_here = i == chars.len()
                    || chars[i] == '\n'
                    || chars[i] == '。'
                    || (chars[i].is_whitespace() && label_like(&chars[s..i]));
                if !end_here {
                    continue;
                }
                let e = if i < chars.len() && chars[i] == '。' {
                    i + 1
                } else {
                    i
                };
                if e > s {
                    sents.push((ti, s, chars[s..e].iter().collect(), dom));
                }
                s = i + 1;
            }
        }
        // キャッシュに無い文だけを計算する
        let keys: Vec<u64> = sents
            .iter()
            .map(|(_, _, t, d)| sentence_key(t, *d))
            .collect();
        let mut done: Vec<Option<Vec<Finding>>> = match &self.cache {
            Some(c) => {
                let c = c.read().unwrap();
                keys.iter().map(|k| c.get(k).cloned()).collect()
            }
            None => vec![None; sents.len()],
        };
        let todo: Vec<usize> = (0..sents.len()).filter(|&i| done[i].is_none()).collect();
        if dbg {
            eprintln!(
                "  split+lookup {:.2?} ({} sents, {} todo)",
                t0.map(|t| t.elapsed()).unwrap_or_default(),
                sents.len(),
                todo.len()
            );
        }
        // 1 段目 (n-gram) は文ごとに並列
        let mut works: Vec<Vec<Vec<Finding>>> = todo
            .par_iter()
            .map(|&i| self.stage1(&sents[i].2, sents[i].3))
            .collect();
        // 2 段目 (MLM) は全文まとめて
        if let Some(mlm) = &self.mlm {
            let texts: Vec<&str> = todo.iter().map(|&i| sents[i].2.as_str()).collect();
            let doms: Vec<Domain> = todo.iter().map(|&i| sents[i].3).collect();
            self.mlm_rescore_all(mlm, &texts, &doms, &mut works);
        }
        for (&i, sites) in todo.iter().zip(works) {
            let mut fs = self.finalize(sites, sents[i].3);
            restore_dropped_i(&sents[i].2, &mut fs);
            done[i] = Some(fs);
        }
        if let Some(c) = &self.cache {
            let mut c = c.write().unwrap();
            if c.len() + todo.len() > CACHE_LIMIT {
                c.clear();
            }
            for &i in &todo {
                c.insert(keys[i], done[i].clone().unwrap_or_default());
            }
        }
        if dbg {
            eprintln!(
                "  compute+store {:.2?}",
                t0.map(|t| t.elapsed()).unwrap_or_default()
            );
        }
        let mut out: Vec<Vec<Finding>> = vec![Vec::new(); texts.len()];
        for ((ti, off, _, _), fs) in sents.iter().zip(done) {
            for mut f in fs.unwrap_or_default() {
                f.start += off;
                f.end += off;
                for a in &mut f.alternatives {
                    a.start += off;
                    a.end += off;
                }
                out[*ti].push(f);
            }
        }
        // original は元テキスト (正規化前) から切り出し直す
        for (ti, fs) in out.iter_mut().enumerate() {
            let orig: Vec<char> = texts[ti].chars().collect();
            for f in fs.iter_mut() {
                f.original = orig[f.start..f.end].iter().collect();
                for a in &mut f.alternatives {
                    a.original = orig[a.start..a.end].iter().collect();
                }
            }
        }
        out
    }

    /// 正規化済みの 1 文を検査する。
    pub fn check_sentence(&self, sent: &str) -> Vec<Finding> {
        self.check_many(&[sent]).pop().unwrap_or_default()
    }

    /// 1 段目: n-gram で候補を採点し、閾値 (MLM 併用時は少し緩めた値) を超えたものを残す。
    /// 重なる・隣接する候補を「箇所」にまとめ、箇所ごとに上位 top_k 件を返す (delta は n-gram の Δ)。
    #[allow(clippy::too_many_lines)] // 候補の出どころ (単語・パターン・カタカナ語・文字モデル・規則) を 1 か所で見渡せるようにしている
    fn stage1(&self, sent: &str, d: Domain) -> Vec<Vec<Finding>> {
        let toks = self.tok.tokenize(sent);
        if toks.is_empty() {
            return Vec::new();
        }
        let ids = self.ids_of(&toks);
        let mut cands: Vec<Finding> = self
            .scan(&toks, &ids, d, false, None)
            .into_iter()
            .map(|(f, _)| f)
            .collect();
        if self.threshold(EditKind::Char, d).is_finite() {
            cands.extend(self.char_edits(sent, &toks, &ids, d));
        }
        if let Some(p) = &self.patterns {
            let found = p.find(sent);
            if !found.is_empty() {
                // パターンは前後の文脈を見ないので (「なってしまします → しまいます」は正しいが
                // 「いたしまします」には当てはまらない)、直した文の n-gram の尤度の改善幅とパターンの支持数で採否を決める
                let base = self.sentence_logp(&ids);
                for f in found {
                    let fixed = apply_finding(sent, &f);
                    let sd = self.sentence_logp(&self.ids_of(&self.tok.tokenize(&fixed))) - base;
                    if self.trace {
                        eprintln!(
                            "  [pattern] 「{}」→「{}」 文の Δ={sd:.2} 支持={:.2}\t{fixed}",
                            f.original, f.replacement, f.delta
                        );
                    }
                    // スコアは採否にだけ使い、箇所の中での順位は支持数 (log10) のままにする
                    // (スコアをそのまま使うと、同じ箇所の正しい活用の修正案より上に来てしまう)
                    if pattern_accepted(d, sd, f.delta) {
                        cands.push(f);
                    }
                }
            }
        }
        if let Some(k) = &self.katakana {
            cands.extend(k.find(sent));
        }
        // 文字モデルは一般文だけで使う。法令文・契約書は「主監」「副参事」のような
        // 一般のコーパスで珍しい語が多く、語の中の 1 字の誤りより誤検出のほうがずっと多くなる
        // (一宮市の例規で 1 万字あたり 1.8 → 4.5 件)
        if let Some(c) = self.charcheck.as_ref().filter(|_| d == Domain::General) {
            let found = c.candidates(sent);
            if !found.is_empty() {
                // 文字モデルだけでは珍しいが正しい並び (固有名詞・専門語) を拾いすぎるので、
                // 単語の言語モデルでも直した文が自然になるかを確かめる
                let base = self.sentence_logp(&ids);
                let unk = ids.iter().filter(|&&x| x == UNK).count();
                let aux_base = self.aux_sentence_logp(&toks);
                let cooc_ctx = self.cooc.as_ref().map(|cooc| {
                    cooc.context(self.lm.as_ref(), toks.iter().map(|t| t.surface.as_str()))
                });
                let mut sorted_ids = ids[1..ids.len() - 1].to_vec();
                sorted_ids.sort_unstable();
                // 単語モデル・パターンなどですでに指摘する箇所 (と 1 文字以内で隣り合う箇所) は調べない。
                // 指摘の数は変わらず、文字モデルの役目は単語モデルが見落とす箇所を拾うことなので
                // (判定器もそういう候補だけで学習している)
                let taken: Vec<(usize, usize)> = cands
                    .iter()
                    .filter(|g| g.delta >= self.threshold(g.kind, d))
                    .map(|g| (g.start, g.end.max(g.start + 1)))
                    .collect();
                for cc in found {
                    let f = &cc.finding;
                    let end = f.end.max(f.start + 1);
                    if taken.iter().any(|&(a, b)| f.start <= b + 1 && a <= end + 1)
                        || char_deletion_left_to_word_model(&toks, f)
                    {
                        continue;
                    }
                    let fixed = apply_finding(sent, f);
                    let fixed_toks = self.tok.tokenize(&fixed);
                    let fixed_ids = self.ids_of(&fixed_toks);
                    let sd = self.sentence_logp(&fixed_ids) - base;
                    let ad = if self.aux.is_some() {
                        self.aux_sentence_logp(&fixed_toks) - aux_base
                    } else {
                        0.0
                    };
                    // 同音異字と同じく、文全体の語との相性 (直して入る語 − 直して消える語)
                    let co = match (&self.cooc, &cooc_ctx) {
                        (Some(cooc), Some(ctx)) => {
                            let mut fx = fixed_ids[1..fixed_ids.len() - 1].to_vec();
                            fx.sort_unstable();
                            let gained: f32 = sorted_minus(&fx, &sorted_ids)
                                .iter()
                                .map(|&h| cooc.score(h, ctx, None))
                                .sum();
                            let lost: f32 = sorted_minus(&sorted_ids, &fx)
                                .iter()
                                .map(|&h| cooc.score(h, ctx, None))
                                .sum();
                            (gained - lost) / std::f32::consts::LN_10
                        }
                        _ => 0.0,
                    };
                    let cd = cc.delta;
                    if self.trace {
                        let fixed_unk = fixed_ids.iter().filter(|&&x| x == UNK).count();
                        eprintln!(
                            "  [char] 「{}」→「{}」 文字={cd:.2} 単語={sd:.2} 位置={} logp={:.2} 差={:.2} 怪しい={} 未知語={} 語数={} 文法={ad:.2} 共起={co:.2}\t{fixed}",
                            f.original,
                            f.replacement,
                            f.start,
                            cc.logp,
                            cc.margin,
                            cc.suspicious_count,
                            unk as i64 - fixed_unk as i64,
                            fixed_ids.len() as i64 - ids.len() as i64,
                        );
                    }
                    let accepted = match &c.rank {
                        Some(rk) => {
                            let fixed_chars: Vec<char> = fixed.chars().collect();
                            let w = crate::charcheck::WordSignals {
                                delta: sd,
                                unknown_removed: unk as i32
                                    - fixed_ids.iter().filter(|&&x| x == UNK).count() as i32,
                                token_count_change: fixed_ids.len() as i32 - ids.len() as i32,
                                aux_delta: ad,
                                cooc_delta: co,
                            };
                            let score = rk.score(&cc, &fixed_chars, &w);
                            let tau = rk.tau_for(&f.original, &f.replacement, d);
                            if self.trace {
                                eprintln!("    判定={score:.2} (閾値 {tau:.2})");
                            }
                            score > tau
                        }
                        None => char_accepted(&f.original, &f.replacement, cd, sd),
                    };
                    if accepted {
                        cands.push(cc.finding);
                    }
                }
            }
        }
        cands.extend(repeated_function_words(&toks));
        cands.extend(wrong_case_before_ni_verbs(&toks));
        cands.extend(missing_ni_and_iu(&toks));
        // 箇所にまとめる: 1 文字以内で隣り合う候補は同じ誤りの別解とみなす
        // (「飲食は店」の「は」を消す案と「店」を消す案など)
        cands.sort_by_key(|f| (f.start, f.end));
        let mut sites: Vec<Vec<Finding>> = Vec::new();
        let mut site_end = 0usize;
        for f in cands {
            let (a, b) = (f.start, f.end.max(f.start + 1));
            match sites.last_mut() {
                Some(site) if a <= site_end + 1 => {
                    site_end = site_end.max(b);
                    site.push(f);
                }
                _ => {
                    site_end = b;
                    sites.push(vec![f]);
                }
            }
        }
        for site in &mut sites {
            dedup_site(sent, site);
            site.truncate(self.cfg.top_k);
        }
        sites
    }

    /// 判定器の学習データ用: 文の候補 (n-gram Δ が `floor` 以上) と特徴量を返す。
    /// `Finding::delta` は判定器を使わない場合と同じ n-gram の Δ (共起・文法モデルの重みを含む)。
    pub fn candidate_features(
        &self,
        sent: &str,
        d: Domain,
        floor: f32,
    ) -> Vec<(Finding, crate::rerank::Features)> {
        let toks = self.tok.tokenize(sent);
        if toks.is_empty() {
            return Vec::new();
        }
        let ids = self.ids_of(&toks);
        self.scan(&toks, &ids, d, true, Some(floor))
            .into_iter()
            .filter_map(|(f, x)| x.map(|x| (f, x)))
            .collect()
    }

    /// 修正候補を採点して、閾値 (判定器があればその床) を超えたものを返す。
    /// 判定器があるときは `delta` を判定器の対数オッズに置き換える。
    /// `dump` が真なら判定器を通さず、`floor` 以上の候補をすべて特徴量つきで返す。
    #[allow(clippy::too_many_lines)] // 候補ごとの採点の流れ (足切り → n-gram → 文法モデル → 共起 → 判定器) を 1 か所で追えるようにしている
    fn scan(
        &self,
        toks: &[Token],
        ids: &[u32],
        d: Domain,
        dump: bool,
        floor: Option<f32>,
    ) -> Vec<(Finding, Option<crate::rerank::Features>)> {
        // MLM が無くても、閾値の 1.0 下までは別案として見せるために残す
        let slack = if self.mlm.is_some() {
            self.cfg.stage1_slack.max(1.0)
        } else {
            1.0
        };
        let rerank = if dump { None } else { self.rerank.as_ref() };
        let mut out = Vec::new();
        let mut buf: Vec<u32> = Vec::with_capacity(32);
        let mut repl_buf = [0u32; 1];
        // 未出現ゲートの判定は語の位置ごとに 1 回だけ行い、候補間で使い回す
        let novel = self.novel_positions(ids);
        let sent_novel = novel.iter().filter(|v| **v).count() as f32 / novel.len().max(1) as f32;
        let mut cooc_ctx: Option<Vec<u32>> = None;
        let mut aux_ids: Option<(Vec<u32>, Vec<f32>)> = None;
        // 元の文の各位置の対数確率 (候補ごとに同じ値を計算し直さない)
        let lp = position_logps(self.lm.as_ref(), ids);
        for c in self.candidates(toks) {
            let repl_ids: &[u32] = match c.repl {
                None => &[],
                Some(w) => {
                    repl_buf[0] = self.lm.word_id(w);
                    &repl_buf
                }
            };
            if repl_ids.contains(&UNK) {
                continue;
            }
            // 判定器を使わない候補は、種類ごとの閾値で決める。`#exempt` には種類名のほか、
            // 「活用語 + 推量の助動詞」をまとめて直す活用の候補 (「多くあろう → ある」) を表す
            // `inflection-aux` を書ける (判定器は JWTD の「〜であろう」の言い換えに引きずられて強く嫌うため)
            let aux_drop = c.kind == EditKind::Inflection && c.b > c.a + 1;
            let exempt = |r: &crate::rerank::Reranker| {
                r.is_exempt(c.kind.label()) || (aux_drop && r.is_exempt(INFLECTION_AUX))
            };
            let rerank_here = rerank.filter(|r| !exempt(r));
            let th = match (floor, rerank_here) {
                (Some(f), _) => f,
                (None, Some(r)) => r.floor,
                (None, None) => self.base_threshold(c.kind, d) - slack,
            };
            if !th.is_finite() || !Self::is_novel(ids, &novel, c.a + 1, c.b + 1) {
                if self.trace && th.is_finite() {
                    self.trace_cand(toks, &c, "既出の並び", None, None);
                }
                continue;
            }
            let ngram = delta_pre(
                self.lm.as_ref(),
                ids,
                &lp,
                c.a + 1,
                c.b + 1,
                repl_ids,
                &mut buf,
            );
            let mut delta = ngram;
            // 文法モデル (前後 4 語の助詞の並び) の Δ。判定器の特徴量にも使う
            let mut aux_delta: Option<f32> = None;
            if matches!(
                c.kind,
                EditKind::Delete | EditKind::Substitute | EditKind::Insert
            ) && let Some(aux) = &self.aux
                && (self.cfg.aux_weight != 0.0 || dump || rerank.is_some())
            {
                let aux_repl = c.repl.map(|w| aux.word_id(w));
                if aux_repl != Some(UNK) {
                    let (s, alp) = aux_ids.get_or_insert_with(|| {
                        let mut v = Vec::with_capacity(toks.len() + 2);
                        v.push(BOS);
                        v.extend(toks.iter().map(|t| aux.token_id(t)));
                        v.push(EOS);
                        let lp = position_logps(aux.as_ref(), &v);
                        (v, lp)
                    });
                    let r: &[u32] = match &aux_repl {
                        Some(id) => std::slice::from_ref(id),
                        None => &[],
                    };
                    let ad = delta_pre(aux.as_ref(), s, alp, c.a + 1, c.b + 1, r, &mut buf);
                    aux_delta = Some(ad);
                    if c.kind == EditKind::Substitute {
                        delta += self.cfg.aux_weight * ad;
                    }
                }
            }
            // 同音異字は文全体の語との相性も足す (n-gram の前後 2 語だけでは決まらないため)
            let mut cooc_term: Option<f32> = None;
            if c.kind == EditKind::Homophone
                && let Some(cooc) = &self.cooc
            {
                let ctx = cooc_ctx.get_or_insert_with(|| {
                    cooc.context(self.lm.as_ref(), toks.iter().map(|t| t.surface.as_str()))
                });
                let orig = self.lm.word_id(toks[c.a].key());
                // 置き換える元の語そのものは手がかりに数えない
                let skip = cooc.ctx_id(self.lm.as_ref(), &toks[c.a].surface);
                let diff = cooc.score(repl_ids[0], ctx, skip) - cooc.score(orig, ctx, skip);
                let term = diff / std::f32::consts::LN_10;
                cooc_term = Some(term);
                delta += self.cfg.cooc_weight * term;
            }
            if delta < th {
                if self.trace {
                    self.trace_cand(toks, &c, "Δ不足", Some(delta), None);
                }
                continue;
            }
            let nov = || {
                novel[c.a + 1..(c.b + 3).min(novel.len())]
                    .iter()
                    .filter(|v| **v)
                    .count()
            };
            let signals = || Signals {
                ngram,
                aux: aux_delta,
                cooc: cooc_term,
                novel: nov(),
                sent_novel,
            };
            let feats = if dump {
                Some(self.features(toks, &c, d, &signals()))
            } else {
                None
            };
            // 判定器があるのに使わなかった候補は、種類ごとの閾値との差を判定器の閾値の尺度へ移す
            // (最後の採否と箇所内の順位付けは判定器の閾値で行うため)
            if let Some(r) = rerank.filter(|r| exempt(r)) {
                delta = delta - self.base_threshold(c.kind, d) + r.tau_for(c.kind.label(), d);
            }
            if let Some(r) = rerank_here {
                delta = self.rerank_score(r, toks, &c, d, &signals());
                // 名詞の間の「の」などを消す候補は、他の削除と別の閾値 (`#tau_kind delete-gen` など) で決める。
                // 最後の採否は種類 (delete) の閾値で行うので、閾値の差だけ Δ をずらす
                if c.kind == EditKind::Delete
                    && c.b == c.a + 1
                    && let Some(class) = delete_class(toks, c.a)
                    && r.tau_kind.contains_key(class)
                {
                    delta += r.tau_for(c.kind.label(), d) - r.tau_for(class, d);
                }
                if self.trace {
                    self.trace_cand(
                        toks,
                        &c,
                        "判定器",
                        Some(ngram),
                        Some((delta, r.tau_for(c.kind.label(), d))),
                    );
                }
            }
            let start = toks
                .get(c.a)
                .map(|t| t.start)
                .unwrap_or_else(|| toks.last().unwrap().end);
            let end = if c.b > c.a { toks[c.b - 1].end } else { start };
            out.push((
                Finding {
                    start,
                    end,
                    original: toks[c.a..c.b].iter().map(|t| t.surface.as_str()).collect(),
                    replacement: c.repl.unwrap_or("").to_string(),
                    kind: c.kind,
                    delta,
                    alternatives: Vec::new(),
                },
                feats,
            ));
        }
        out
    }

    fn trace_cand(
        &self,
        toks: &[Token],
        c: &Cand<'_>,
        why: &str,
        ngram: Option<f32>,
        rerank: Option<(f32, f32)>,
    ) {
        let orig: String = toks[c.a..c.b].iter().map(|t| t.surface.as_str()).collect();
        let before: String = toks[c.a.saturating_sub(2)..c.a]
            .iter()
            .map(|t| t.surface.as_str())
            .collect();
        let after: String = toks[c.b..(c.b + 2).min(toks.len())]
            .iter()
            .map(|t| t.surface.as_str())
            .collect();
        let ngram = ngram.map_or_else(String::new, |x| format!(" Δ={x:.2}"));
        let rerank =
            rerank.map_or_else(String::new, |(z, t)| format!(" 判定={z:.2} (閾値 {t:.2})"));
        eprintln!(
            "  [{why}] {} {before}[{orig}→{}]{after}{ngram}{rerank}",
            c.kind.label(),
            c.repl.unwrap_or("")
        );
    }

    /// 判定器の特徴量 (学習データの書き出し用に名前つきで集める)。
    fn features(
        &self,
        toks: &[Token],
        c: &Cand<'_>,
        d: Domain,
        sig: &Signals,
    ) -> crate::rerank::Features {
        let mut f: crate::rerank::Features = Vec::with_capacity(20);
        self.emit_features(toks, c, d, sig, &mut |k, v| {
            f.push((k.to_string(), v));
        });
        f
    }

    /// 判定器のスコア (特徴量の名前を確保せず、使い回しのバッファでハッシュを引く)。
    fn rerank_score(
        &self,
        r: &crate::rerank::Reranker,
        toks: &[Token],
        c: &Cand<'_>,
        d: Domain,
        sig: &Signals,
    ) -> f32 {
        let mut z = 0.0;
        self.emit_features(toks, c, d, sig, &mut |k, v| z += r.weight(k) * v);
        z
    }

    /// 判定器の特徴量を 1 つずつ `emit(名前, 値)` に渡す。名前は「種類:内容」の形にして、
    /// 種類ごとに別の重みを持たせる。
    #[allow(clippy::too_many_lines)] // 特徴量の一覧を 1 か所で見渡せるようにしている
    fn emit_features(
        &self,
        toks: &[Token],
        c: &Cand<'_>,
        d: Domain,
        sig: &Signals,
        emit: &mut dyn FnMut(&str, f32),
    ) {
        use std::fmt::Write as _;
        let Signals {
            ngram,
            aux,
            cooc,
            novel,
            sent_novel,
        } = *sig;
        let k = c.kind.label();
        let bin = |x: f32| (x.floor() as i32).clamp(-2, 15);
        // 機能語 (助詞・助動詞・記号) は表層形、それ以外は品詞で表す
        let key = |t: Option<&Token>, buf: &mut String| match t {
            None => buf.push_str("<s>"),
            Some(t) if matches!(t.pos, "助詞" | "助動詞" | "記号") => {
                buf.push_str(&t.surface);
            }
            Some(t) => {
                let _ = write!(buf, "{}-{}", t.pos, t.pos1);
            }
        };
        let mut buf = String::with_capacity(64);
        let put = |buf: &mut String, v: f32, emit: &mut dyn FnMut(&str, f32)| {
            emit(buf, v);
            buf.clear();
        };
        emit("b", 1.0);
        let _ = write!(buf, "k={k}");
        put(&mut buf, 1.0, emit);
        let _ = write!(buf, "{k}:d");
        put(&mut buf, ngram, emit);
        let _ = write!(buf, "{k}:db{}", bin(ngram));
        put(&mut buf, 1.0, emit);
        let _ = write!(buf, "{k}:dom={d:?}");
        put(&mut buf, 1.0, emit);
        let _ = write!(buf, "{k}:nov{novel}");
        put(&mut buf, 1.0, emit);
        let _ = write!(buf, "{k}:sn{}", ((sent_novel * 10.0) as i32).clamp(0, 10));
        put(&mut buf, 1.0, emit);
        if let Some(a) = aux {
            let _ = write!(buf, "{k}:aux");
            put(&mut buf, a, emit);
            let _ = write!(buf, "{k}:ab{}", bin(a));
            put(&mut buf, 1.0, emit);
        }
        if let Some(x) = cooc {
            let _ = write!(buf, "{k}:cooc");
            put(&mut buf, x, emit);
        }
        let repl = c.repl.unwrap_or("");
        match c.kind {
            EditKind::Delete | EditKind::Substitute | EditKind::Insert => {
                let orig_len: usize = toks[c.a..c.b]
                    .iter()
                    .map(|t| t.surface.chars().count())
                    .sum();
                if orig_len <= 4 {
                    let _ = write!(buf, "{k}:o=");
                    for t in &toks[c.a..c.b] {
                        buf.push_str(&t.surface);
                    }
                    put(&mut buf, 1.0, emit);
                }
                let _ = write!(buf, "{k}:r={repl}");
                put(&mut buf, 1.0, emit);
                if c.kind == EditKind::Delete
                    && c.b == c.a + 1
                    && is_genitive_between_nouns(toks, c.a)
                {
                    let _ = write!(buf, "{k}:gen_nn");
                    put(&mut buf, 1.0, emit);
                }
                if c.kind == EditKind::Substitute {
                    let _ = write!(buf, "{k}:or=");
                    for t in &toks[c.a..c.b] {
                        buf.push_str(&t.surface);
                    }
                    let _ = write!(buf, ">{repl}");
                    put(&mut buf, 1.0, emit);
                }
            }
            EditKind::Inflection => {
                if let Some(t) = toks.get(c.a) {
                    let _ = write!(buf, "{k}:of={}", t.conj_form);
                    put(&mut buf, 1.0, emit);
                    let _ = write!(buf, "{k}:ob={}", t.base);
                    put(&mut buf, 1.0, emit);
                }
            }
            EditKind::Homophone => {
                if let Some(t) = toks.get(c.a) {
                    // 同じ読みの語どうしの出現数の比 (よく使われる語への置き換えほど誤変換らしい)
                    let count = |w: &str| {
                        self.readings
                            .get(t.reading)
                            .and_then(|v| v.iter().find(|(x, _)| x == w))
                            .map_or(0, |(_, n)| *n)
                    };
                    let freq =
                        ((count(repl) as f32 + 1.0) / (count(&t.surface) as f32 + 1.0)).log10();
                    let _ = write!(buf, "{k}:freq");
                    put(&mut buf, freq, emit);
                    let _ = write!(buf, "{k}:fb{}", bin(freq * 2.0));
                    put(&mut buf, 1.0, emit);
                    // 漢字を共有する組 (対象/対照) は打ち間違いが多く、共有しない組は意味の離れた別語のことが多い
                    if t.surface
                        .chars()
                        .any(|ch| is_kanji(ch) && repl.contains(ch))
                    {
                        let _ = write!(buf, "{k}:share");
                        put(&mut buf, 1.0, emit);
                    }
                    let _ = write!(buf, "{k}:len{}", t.surface.chars().count().min(4));
                    put(&mut buf, 1.0, emit);
                }
            }
            EditKind::Char | EditKind::Pattern => {}
        }
        // 元の語・置き換え先が言語モデルの語彙に無い (品詞クラスで採点している) か
        if c.b == c.a + 1 && self.lm.word_id(toks[c.a].key()) == UNK {
            let _ = write!(buf, "{k}:oov_o");
            put(&mut buf, 1.0, emit);
        }
        if !repl.is_empty() && self.lm.word_id(repl) == UNK {
            let _ = write!(buf, "{k}:oov_r");
            put(&mut buf, 1.0, emit);
        }
        let prev = if c.a > 0 { toks.get(c.a - 1) } else { None };
        let next = toks.get(c.b);
        let _ = write!(buf, "{k}:p=");
        key(prev, &mut buf);
        put(&mut buf, 1.0, emit);
        let _ = write!(buf, "{k}:n=");
        key(next, &mut buf);
        put(&mut buf, 1.0, emit);
        let _ = write!(buf, "{k}:pn=");
        key(prev, &mut buf);
        buf.push('|');
        key(next, &mut buf);
        put(&mut buf, 1.0, emit);
        if next.is_none_or(|t| t.surface == "。") {
            let _ = write!(buf, "{k}:end");
            put(&mut buf, 1.0, emit);
        }
    }

    /// 箇所ごとに最良案のスコアが閾値を超えたら指摘し、残りは別案として添える。
    fn finalize(&self, sites: Vec<Vec<Finding>>, d: Domain) -> Vec<Finding> {
        let mut out = Vec::new();
        for mut site in sites {
            site.sort_by(|x, y| y.delta.total_cmp(&x.delta));
            // 採否は「自分の種類の閾値を超えた案」の中で最良のもので決める
            // (閾値の低い種類の案が、閾値に届かない別種の案に埋もれないように)
            let Some(bi) = site
                .iter()
                .position(|f| f.delta >= self.threshold(f.kind, d))
            else {
                continue;
            };
            let mut f = site[bi].clone();
            f.alternatives = site
                .iter()
                .enumerate()
                .filter(|(i, a)| *i != bi && a.delta >= self.threshold(a.kind, d) - 1.0)
                .map(|(_, a)| a)
                .take(2)
                .map(|a| Suggestion {
                    start: a.start,
                    end: a.end,
                    original: a.original.clone(),
                    replacement: a.replacement.clone(),
                    kind: a.kind,
                    score: a.delta,
                })
                .collect();
            out.push(f);
        }
        out
    }

    /// 2 段目: MLM で採点し直す。delta を「n-gram Δ + 重み × MLM Δ」に書き換える。
    /// 1 文に 2 箇所以上が採用されそうなら、ほかの箇所を直した文脈でもう一度採点する
    /// (「西口側までは宿泊から施設…」の「まで」を、「から」を直した文で判断するため)。
    fn mlm_rescore_all(
        &self,
        mlm: &Mlm,
        sents: &[&str],
        doms: &[Domain],
        works: &mut [Vec<Vec<Finding>>],
    ) {
        let ng: Vec<Vec<Vec<f32>>> = works
            .iter()
            .map(|sites| {
                sites
                    .iter()
                    .map(|s| s.iter().map(|f| f.delta).collect())
                    .collect()
            })
            .collect();
        // MLM に回す箇所: 採否が閾値すれすれか、1 位と 2 位の案が拮抗しているもの
        let need: Vec<Vec<bool>> = works
            .iter()
            .zip(&ng)
            .zip(doms)
            .map(|((sites, ng), &d)| {
                sites
                    .iter()
                    .zip(ng)
                    .map(|(site, sc)| {
                        let best = sc[0];
                        let th = self.threshold(site[0].kind, d);
                        let close_second = sc.len() > 1 && best - sc[1] < self.cfg.mlm_gap;
                        if self.cfg.mlm_choice_only {
                            // 速度優先: 採否は n-gram で決め、確定した箇所の「どの案にするか」だけ MLM に聞く
                            best >= th && close_second
                        } else {
                            let near_threshold = best < th + self.cfg.mlm_band;
                            near_threshold || close_second
                        }
                    })
                    .collect()
            })
            .collect();
        let all: Vec<usize> = (0..works.len())
            .filter(|&i| need[i].iter().any(|b| *b))
            .collect();
        let scores = self.mlm_scores(mlm, sents, works, &ng, &all, None, &need);
        // 2 回目が必要な文
        let mut fixed_all: Vec<Option<Vec<Option<usize>>>> = vec![None; works.len()];
        for (&wi, sc) in all.iter().zip(&scores) {
            let accepted: Vec<Option<usize>> = works[wi]
                .iter()
                .zip(sc)
                .map(|(site, sc)| {
                    let (bi, bs) = sc.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1))?;
                    (*bs >= self.threshold(site[bi].kind, doms[wi])).then_some(bi)
                })
                .collect();
            if accepted.iter().filter(|a| a.is_some()).count() >= 2 {
                fixed_all[wi] = Some(accepted);
            }
        }
        // MLM を使わない箇所も、2 回目で「直した文脈」を作るために最良案を採用扱いにする
        for (wi, fx) in fixed_all.iter_mut().enumerate() {
            if let Some(fx) = fx {
                for (si, site) in works[wi].iter().enumerate() {
                    if !need[wi][si]
                        && fx[si].is_none()
                        && ng[wi][si][0] >= self.threshold(site[0].kind, doms[wi])
                    {
                        fx[si] = Some(0);
                    }
                }
            }
        }
        let again: Vec<usize> = (0..works.len())
            .filter(|&i| fixed_all[i].is_some())
            .collect();
        let scores2 = self.mlm_scores(mlm, sents, works, &ng, &again, Some(&fixed_all), &need);
        let mut final_scores: FxHashMap<usize, Vec<Vec<f32>>> =
            all.into_iter().zip(scores).collect();
        for (wi, sc) in again.into_iter().zip(scores2) {
            final_scores.insert(wi, sc);
        }
        for (wi, sc) in final_scores {
            for (site, ss) in works[wi].iter_mut().zip(sc) {
                for (f, s) in site.iter_mut().zip(ss) {
                    f.delta = s;
                }
            }
        }
    }

    /// `targets` の各文について、箇所ごと・候補ごとの最終スコアを返す。
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn mlm_scores(
        &self,
        mlm: &Mlm,
        sents: &[&str],
        works: &[Vec<Vec<Finding>>],
        ng: &[Vec<Vec<f32>>],
        targets: &[usize],
        fixed: Option<&[Option<Vec<Option<usize>>>]>,
        need: &[Vec<bool>],
    ) -> Vec<Vec<Vec<f32>>> {
        let mut texts: Vec<(String, usize, usize)> = Vec::new();
        // (targets 内の番号, 箇所, 候補 or 元)
        let mut index: Vec<(usize, usize, Option<usize>)> = Vec::new();
        for (ti, &wi) in targets.iter().enumerate() {
            let chars: Vec<char> = sents[wi].chars().collect();
            let sites = &works[wi];
            let fx = fixed.and_then(|f| f[wi].as_ref());
            for (si, site) in sites.iter().enumerate() {
                if !need[wi][si] {
                    continue;
                }
                // 土台の文 (fixed があれば、ほかの箇所を直したもの) と、この箇所の開始位置のずれ
                let (base, shift) = match fx {
                    None => (chars.clone(), 0isize),
                    Some(fx) => {
                        let mut out: Vec<char> = Vec::with_capacity(chars.len() + 4);
                        let mut pos = 0;
                        let mut shift = 0isize;
                        for (sj, other) in sites.iter().enumerate() {
                            let Some(bi) = fx[sj] else { continue };
                            if sj == si {
                                continue;
                            }
                            let f = &other[bi];
                            out.extend_from_slice(&chars[pos..f.start]);
                            out.extend(f.replacement.chars());
                            if sj < si {
                                shift += f.replacement.chars().count() as isize
                                    - (f.end - f.start) as isize;
                            }
                            pos = f.end;
                        }
                        out.extend_from_slice(&chars[pos..]);
                        (out, shift)
                    }
                };
                let mv = |x: usize| (x as isize + shift) as usize;
                let s0 = site.iter().map(|f| f.start).min().unwrap();
                let s1 = site.iter().map(|f| f.end).max().unwrap();
                texts.push((base.iter().collect(), mv(s0), mv(s1)));
                index.push((ti, si, None));
                for (ci, f) in site.iter().enumerate() {
                    let (a, b) = (mv(f.start), mv(f.end));
                    let mut t: String = base[..a].iter().collect();
                    t.push_str(&f.replacement);
                    t.extend(base[b..].iter());
                    // 比べる範囲は「箇所全体」を編集後の座標で表したもの
                    let grow = f.replacement.chars().count() as isize - (f.end - f.start) as isize;
                    let e1 = (mv(s1) as isize + grow).max(mv(s0) as isize) as usize;
                    texts.push((t, mv(s0), e1));
                    index.push((ti, si, Some(ci)));
                }
            }
        }
        let qs: Vec<Query> = texts
            .iter()
            .map(|(t, a, b)| Query {
                text: t,
                start: *a,
                end: *b,
            })
            .collect();
        let pll = if self.cfg.mlm_pll {
            mlm.window_pll(&qs, self.cfg.mlm_margin)
        } else {
            // 穴埋め採点は直す範囲だけをマスクする (前後の語は文脈として見せる)
            mlm.fill_scores(&qs, self.cfg.mlm_margin)
        }
        .unwrap_or_else(|_| vec![(0.0, 0); qs.len()]);
        let mut out: Vec<Vec<Vec<f32>>> = targets.iter().map(|&wi| ng[wi].clone()).collect();
        let mut orig = (0f32, 0usize);
        for ((ti, si, ci), (p, n)) in index.iter().zip(&pll) {
            match ci {
                None => orig = (*p, *n),
                Some(ci) => {
                    // 削除は採点対象のサブワードが減るぶん PLL が有利になるので、1 サブワードあたり
                    // length_penalty (nats) を差し引いて釣り合わせる
                    let d =
                        (p - orig.0) + self.cfg.mlm_length_penalty * (*n as f32 - orig.1 as f32);
                    let wi = targets[*ti];
                    out[*ti][*si][*ci] =
                        ng[wi][*si][*ci] + self.cfg.mlm_weight * d / std::f32::consts::LN_10;
                }
            }
        }
        out
    }

    /// 指摘箇所の前後 2 文字を含む元の文字列が、文書内に何回出てくるか (`extra` は外部から足す回数)。
    /// 誤字はたいてい 1 回限りなので、同じ並びが繰り返し出てくるなら意図した表記 (専門用語・定型句) とみなす。
    pub fn doc_repeats(&self, doc: &str, chars: &[char], f: &Finding, extra: usize) -> usize {
        let a = f.start.saturating_sub(2);
        let b = (f.end + 2).min(chars.len());
        let ctx: String = chars[a..b].iter().collect();
        if ctx.chars().count() < 3 {
            return 1 + extra;
        }
        // 上限 (doc_repeat_limit) 回見つかれば十分なので打ち切る。memmem は SIMD で走査するので、指摘ごとに文書全体を
        // 見ても 20 万字で数十 µs で済む (str::matches だと数 ms かかり、キャッシュ命中時の大半を占めた)
        memchr::memmem::find_iter(doc.as_bytes(), ctx.as_bytes())
            .take(self.cfg.doc_repeat_limit)
            .count()
            + extra
    }

    fn threshold(&self, k: EditKind, d: Domain) -> f32 {
        // 判定器があるときは、種類によらず判定器の閾値 (対数オッズ) で決める (文字単位の候補は除く)
        // (判定器を使わなかった候補も、scan で Δ をこの尺度へ移してある)
        if let Some(r) = &self.rerank
            && !matches!(k, EditKind::Char | EditKind::Pattern)
        {
            return r.tau_for(k.label(), d);
        }
        self.base_threshold(k, d)
    }

    /// 種類ごとの閾値 (判定器を使わない場合の値)。
    fn base_threshold(&self, k: EditKind, d: Domain) -> f32 {
        let t = match d {
            Domain::Legal => &self.cfg.thresholds,
            Domain::General => &self.cfg.general_thresholds,
            Domain::Contract => &self.cfg.contract_thresholds,
        };
        t.get(&k).copied().unwrap_or(f32::INFINITY)
    }

    fn ids_of(&self, toks: &[Token]) -> Vec<u32> {
        let mut ids: Vec<u32> = Vec::with_capacity(toks.len() + 2);
        ids.push(BOS);
        ids.extend(toks.iter().map(|t| self.lm.token_id(t)));
        ids.push(EOS);
        ids
    }

    fn sentence_logp(&self, ids: &[u32]) -> f32 {
        seq_logp(self.lm.as_ref(), ids)
    }

    /// 文法モデルでの文の対数確率 (文法モデルが無ければ 0)。
    fn aux_sentence_logp(&self, toks: &[Token]) -> f32 {
        self.aux.as_ref().map_or(0.0, |aux| {
            let mut v = Vec::with_capacity(toks.len() + 2);
            v.push(BOS);
            v.extend(toks.iter().map(|t| aux.token_id(t)));
            v.push(EOS);
            seq_logp(aux.as_ref(), &v)
        })
    }

    /// 文字単位の編集 (1 文字削除・隣接入れ替え・かな 1 文字補完) を、
    /// 「コーパスに出てこない並び」がある語の周辺だけで試す。編集後は分かち書きし直して文全体で比べる。
    fn char_edits(&self, sent: &str, toks: &[Token], ids: &[u32], d: Domain) -> Vec<Finding> {
        let th = self.threshold(EditKind::Char, d);
        let n = self.cfg.novelty_order.max(2);
        let chars: Vec<char> = sent.chars().collect();
        // 未出現の並びに関わる語 (S の添字 j → toks[j-1]) の前後 1 語を対象範囲にする
        let mut in_region = vec![false; chars.len() + 1];
        // 固有名詞・カタカナ語・語彙外の語は、1 文字消すと既知語に割れて確率が上がりやすい (誤検出の主因)
        let risky = |t: &Token| {
            t.pos1 == "固有名詞"
                || t.surface
                    .chars()
                    .all(|c| is_kana(c) && !('ぁ'..='ゖ').contains(&c))
                || self.lm.token_id(t) == UNK
        };
        for j in 1..ids.len() - 1 {
            if ids[j] == UNK || ids[j - 1] == UNK {
                continue;
            }
            let ti = j - 1;
            if risky(&toks[ti])
                || (ti > 0 && risky(&toks[ti - 1]))
                || (ti + 1 < toks.len() && risky(&toks[ti + 1]))
            {
                continue;
            }
            let ctx = &ids[j.saturating_sub(n - 1)..j];
            if self.lm.match_order(ctx, ids[j]) < n.min(ctx.len() + 1) {
                let lo = toks[(j - 1).saturating_sub(1)].start;
                let hi = toks[(j).min(toks.len() - 1)].end;
                for f in in_region.iter_mut().take(hi + 1).skip(lo) {
                    *f = true;
                }
            }
        }
        if !in_region.iter().any(|b| *b) {
            return Vec::new();
        }
        let base = self.sentence_logp(ids);
        let mut out = Vec::new();
        let try_edit = |start: usize, end: usize, repl: &str, out: &mut Vec<Finding>| {
            let mut s: String = chars[..start].iter().collect();
            s.push_str(repl);
            s.extend(chars[end..].iter());
            let t2 = self.tok.tokenize(&s);
            let ids2 = self.ids_of(&t2);
            if ids2.iter().filter(|&&x| x == UNK).count()
                > ids.iter().filter(|&&x| x == UNK).count()
                || t2
                    .iter()
                    .any(|t| t.pos1 == "固有名詞" && !toks.iter().any(|o| o.surface == t.surface))
            {
                return;
            }
            let delta = self.sentence_logp(&ids2) - base;
            if delta >= th {
                out.push(Finding {
                    start,
                    end,
                    original: chars[start..end].iter().collect(),
                    replacement: repl.to_string(),
                    kind: EditKind::Char,
                    delta,
                    alternatives: Vec::new(),
                });
            }
        };
        for p in 0..chars.len() {
            if !in_region[p] {
                continue;
            }
            let c = chars[p];
            if c.is_alphanumeric() || is_kana(c) {
                try_edit(p, p + 1, "", &mut out);
            }
            if p + 1 < chars.len()
                && in_region[p + 1]
                && chars[p] != chars[p + 1]
                && (is_kana(chars[p]) || is_kana(chars[p + 1]))
            {
                let sw: String = [chars[p + 1], chars[p]].iter().collect();
                try_edit(p, p + 2, &sw, &mut out);
            }
            // かな 1 文字の脱落 (直前か直後がかなの位置だけ)
            if p > 0 && (is_kana(chars[p - 1]) || is_kana(c)) {
                for k in INSERT_KANA {
                    try_edit(p, p, k, &mut out);
                }
            }
        }
        out
    }

    /// 元の文の S[a-1 .. b+1] 付近に、コーパスで見たことのない並びがあるか。
    /// 語彙外の語が隣接している場合は判断できないので false (指摘しない) にする。
    fn is_novel(s: &[u32], novel: &[bool], a: usize, b: usize) -> bool {
        let lo = a.saturating_sub(1);
        let hi = (b + 2).min(s.len());
        if s[lo..hi].contains(&UNK) {
            return false;
        }
        novel[a..hi].iter().any(|v| *v)
    }

    /// S の各位置 j について、直前 n-1 語の文脈で予測したとき完全一致する n-gram が無いか。
    /// 編集箇所の語と、その直後 2 語までのどこかが未出現なら、その候補を評価する。
    fn novel_positions(&self, s: &[u32]) -> Vec<bool> {
        let n = self.cfg.novelty_order;
        (0..s.len())
            .map(|j| {
                if n == 0 {
                    return true;
                }
                let ctx = &s[j.saturating_sub(n - 1)..j];
                self.lm.match_order(ctx, s[j]) < n.min(ctx.len() + 1)
            })
            .collect()
    }

    fn candidates(&self, toks: &[Token]) -> Vec<Cand<'_>> {
        let mut out = Vec::new();
        for (i, t) in toks.iter().enumerate() {
            let is_particle = t.pos == "助詞";
            // 削除・置換の対象は単純な助詞だけ。「によって」「に対し」等の複合助詞は消しても文が成立しやすく誤検出源になる
            let is_simple_particle = is_particle && PARTICLES.contains(&t.surface.as_str());
            // 1 文字の平仮名は、内容語の一部 (「のまネコ」の「ま」など) を消さないよう機能語に限る
            let is_hira1 = t.surface.chars().count() == 1
                && t.surface.chars().all(|c| ('ぁ'..='ん').contains(&c))
                && !matches!(
                    t.pos,
                    "名詞" | "動詞" | "形容詞" | "接頭詞" | "副詞" | "連体詞"
                );
            let dup = i > 0 && toks[i - 1].surface == t.surface && t.pos != "名詞";
            // 空白に接する語は、PDF 由来の改行崩れや見出しの区切りで前後の文脈が切れていることが多く、
            // n-gram の判定が当てにならない (「使用さ れる」「よ う」)。ここは削除・置換・挿入とも出さない
            if touches_space(toks, i, i + 1) {
                continue;
            }
            // 「ユーザもベンダも」「AやB」のような並列は、片方を消したり言い換えたりしても文が成立するので
            // n-gram では誤りに見えやすい。並列の「も」は削除・置換の対象から外す
            let parallel = is_parallel_mo(toks, i) || is_parallel_to(toks, i);
            // 「にも」「については」「ベンダからは」のように、格助詞に係助詞「は」「も」が続く形は
            // どちらを消しても文が成立するので、n-gram では誤りに見えやすい (契約書の誤検出で多かった)。
            // 係助詞側と、「からは」「では」「とも」「よりは」の格助詞側を削除の対象から外す。
            // 「まで」は外さない: 「西口側までは」のように「まで」自体が誤りのことがあり、
            // その場合は「まで」を消す案 (と「に」への置換案) を出したい
            let is_kakari =
                |x: &Token| x.pos == "助詞" && matches!(x.surface.as_str(), "は" | "も");
            let after_particle = is_kakari(t) && i > 0 && toks[i - 1].pos == "助詞";
            let before_kakari = is_particle
                && matches!(t.surface.as_str(), "から" | "で" | "と" | "より")
                && toks.get(i + 1).is_some_and(is_kakari);
            // 「次回までに」「月末までに」の「までに」は期限を表す一続きの言い方なので、「まで」を消す候補にしない
            // (「西口側までは」の「まで」は消す候補に残す)
            // 「違反したとまではいえない」の「まで」は強調 (「とまで」) なので消さない
            let emphatic_made = t.surface == "まで" && i > 0 && toks[i - 1].surface == "と";
            let deadline_madeni =
                t.surface == "まで" && toks.get(i + 1).is_some_and(|x| x.surface == "に");
            let no_delete = after_particle || before_kakari || deadline_madeni || emphatic_made;
            if (is_simple_particle || is_hira1 || dup) && !parallel && !no_delete {
                out.push(Cand {
                    a: i,
                    b: i + 1,
                    repl: None,
                    kind: EditKind::Delete,
                });
            }
            // 置換元は単純な助詞に限る (「によって」→「に」のような複合助詞の言い換えは正しい文でも高得点になる)。
            // 置換先には法令文で使う複合助詞も含める (「市民税[が]経過措置」→「に関する」)
            if is_simple_particle && !parallel {
                for p in PARTICLES.iter().chain(COMPOUND_PARTICLES) {
                    // 並列の「や」と「と」はどちらでも正しいので言い換えを出さない
                    let interchangeable =
                        matches!((t.surface.as_str(), *p), ("や", "と") | ("と", "や"));
                    if *p != t.surface && !interchangeable {
                        out.push(Cand {
                            a: i,
                            b: i + 1,
                            repl: Some(p),
                            kind: EditKind::Substitute,
                        });
                    }
                }
            }
            // 文末の命令形 (判決主文の「支払え。」) と話し言葉の縮約 (「進めてる」「書いとく」) は書き手が選んだ形で、
            // 誤字ではないので活用を直す候補にしない。文中の命令形 (「支払え義務」) は打ち間違いのことがあるので残す
            let sentence_final = toks
                .get(i + 1)
                .is_none_or(|x| x.pos == "記号" && matches!(x.pos1, "句点" | "括弧閉"));
            let intentional_form = (t.conj_form.starts_with("命令") && sentence_final)
                || (t.pos1 == "非自立"
                    && matches!(
                        t.base,
                        "てる" | "でる" | "とく" | "どく" | "ちゃう" | "じゃう"
                    ));
            if matches!(t.pos, "動詞" | "形容詞" | "助動詞")
                && !t.conj_type.is_empty()
                && !intentional_form
                && let Some(forms) = self
                    .inflections
                    .get(&(t.base.to_string(), t.conj_type.to_string()))
            {
                // 活用語 + 後続の助動詞列 (最大 2 つ) をまとめて置き換える
                let mut j = i + 1;
                let mut spans = vec![j];
                while j < toks.len() && j < i + 3 && toks[j].pos == "助動詞" {
                    j += 1;
                    spans.push(j);
                }
                for &b in &spans {
                    // 落とす助動詞は推量の「う」「よう」だけにする (「多くあろう」→「ある」)。
                    // 過去・願望・丁寧・推定 (た / たい / ます / らしい …) を落とす言い換えは、
                    // 元の文も正しいことがほとんどで誤検出になる (「交わした」→「交わす」)。
                    if toks[i + 1..b]
                        .iter()
                        .any(|x| !matches!(x.base, "う" | "よう"))
                    {
                        continue;
                    }
                    // 「〜であろう」「必要があろう」は推量として正しい。誤りになりやすいのは
                    // 「多くあろう」のように活用語の直後に「あろう」が続く形なので、直前が助詞か「で」なら出さない
                    if b > i + 1
                        && i > 0
                        && (toks[i - 1].pos == "助詞" || toks[i - 1].surface == "で")
                    {
                        continue;
                    }
                    for f in forms {
                        if b == i + 1 && *f == t.surface {
                            continue;
                        }
                        out.push(Cand {
                            a: i,
                            b,
                            repl: Some(f.as_str()),
                            kind: EditKind::Inflection,
                        });
                    }
                }
            }

            // 固有名詞 (人名・地名・氏族名) は同音の別表記が正しいことが多く (「加茂」「賀茂」、「一雄」「一夫」)、
            // 文脈からも決められないので同音異字の対象にしない
            if matches!(t.pos, "名詞" | "動詞" | "形容詞" | "副詞")
                && t.pos1 != "固有名詞"
                && !t.reading.is_empty()
                && let Some(alts) = self.readings.get(t.reading)
            {
                // 誤変換 (別の漢字) だけを狙う。かな書き→漢字 (かかる→係る) や送り仮名違い (当り→当たり) は
                // 誤字ではなく表記ゆれなので出さない。1 文字の語は同音が多すぎて誤検出が増えるので除く。
                let kanji_of = |w: &str| w.chars().filter(|c| is_kanji(*c)).collect::<String>();
                let orig_kanji = kanji_of(&t.surface);
                for (alt, cnt) in alts.iter().take(12) {
                    if *alt != t.surface
                        && *cnt >= 20
                        && t.surface.chars().count() >= 2
                        && !orig_kanji.is_empty()
                        && !is_notation_variant(&orig_kanji, &kanji_of(alt))
                    {
                        out.push(Cand {
                            a: i,
                            b: i + 1,
                            repl: Some(alt.as_str()),
                            kind: EditKind::Homophone,
                        });
                    }
                }
            }
            // 名詞 (または閉じ括弧「」」「)」) の直後で、次が助詞でも記号でもない位置か、読点の直前に補う
            // (「この条例□、公布の日から」「『納税義務者』□いう」)
            // IPADIC は「お忙しい」「お美しい」を名詞にしているが、実際は形容詞なので後ろに助詞を補わない
            // (「お忙しい□ところ」に「の」を補う誤検出になる)
            let prev_ok = i > 0
                && ((toks[i - 1].pos == "名詞" && !is_honorific_adjective(&toks[i - 1].surface))
                    || (toks[i - 1].pos == "記号" && toks[i - 1].pos1 == "括弧閉"));
            let next_ok = t.pos != "助詞" && (t.pos != "記号" || t.pos1 == "読点");
            if self.cfg.enable_insert && prev_ok && next_ok {
                for p in INSERT_PARTICLES.iter().chain(COMPOUND_PARTICLES) {
                    out.push(Cand {
                        a: i,
                        b: i,
                        repl: Some(p),
                        kind: EditKind::Insert,
                    });
                }
            }
            if self.cfg.enable_insert {
                for p in extra_insertions(toks, i) {
                    out.push(Cand {
                        a: i,
                        b: i,
                        repl: Some(p),
                        kind: EditKind::Insert,
                    });
                }
            }
        }
        out
    }
}

/// 分かち書き時に集めた活用表 (TSV: 原形 \t 活用型 \t 表層形) を読む。
///
/// `keep` で表層形を絞る (言語モデルの語彙に無い語は候補にしても採点できないので、
/// 読み込み時に落としてメモリを抑える。語彙 1 万語なら 10 分の 1 以下になる)。
/// 意味が同じで、どちらの表記も許容される同音の組 (漢字部分)。
///
/// 誤変換ではなく表記の選び方なので、同音異字の候補にしない (表記の統一は用字用語・表記ゆれの機能が担う)。
/// 「生」「活」のように漢字 1 字の組は、同じ読みの語どうし (生かす/活かす) にだけ効く。
/// 意味の違う異字同訓 (追求/追及、意思/意志、配布/配付、超える/越える など) は入れない。
const NOTATION_VARIANTS: &[(&str, &str)] = &[
    ("交代", "交替"),
    ("稼動", "稼働"),
    ("充分", "十分"),
    ("係", "関"),
    ("拘", "関"),
    ("名字", "苗字"),
    ("収集", "蒐集"),
    ("回", "廻"),
    ("付則", "附則"),
    ("付記", "附記"),
    ("付属", "附属"),
    ("付帯", "附帯"),
    ("寄付", "寄附"),
    ("付置", "附置"),
    ("関数", "函数"),
    ("侵食", "浸食"),
    ("賞賛", "称賛"),
    ("摩耗", "磨耗"),
    ("機運", "気運"),
    ("生", "活"),
    ("広", "拡"),
    ("撹乱", "攪乱"),
    ("車両", "車輛"),
    ("車両", "車輌"),
    ("車輛", "車輌"),
    ("木曽", "木曾"),
    ("陰", "蔭"),
];

fn is_listed_variant(a: &str, b: &str) -> bool {
    NOTATION_VARIANTS
        .iter()
        .any(|&(x, y)| (a == x && b == y) || (a == y && b == x))
}

/// 同音の 2 語の漢字部分が、表記ゆれの関係 (同じ漢字・交ぜ書き・送り仮名違い・許容表記の組) か。
/// 「あん分」⇔「按分」「漏えい」⇔「漏洩」のように一方の漢字が他方に含まれるものは、
/// 誤変換ではなく表記の選び方なので同音異字の候補にしない。
#[must_use]
pub fn is_notation_variant(a: &str, b: &str) -> bool {
    a.chars().all(|c| b.contains(c)) || b.chars().all(|c| a.contains(c)) || is_listed_variant(a, b)
}

/// S の各位置 j の対数確率 (直前 order-1 語の文脈)。候補ごとの「元」の和を使い回すために文ごとに 1 回求める。
pub(crate) fn position_logps(lm: &dyn LanguageModel, s: &[u32]) -> Vec<f32> {
    let order = lm.order();
    (0..s.len())
        .map(|j| {
            if j == 0 {
                0.0
            } else {
                lm.logp(&s[j.saturating_sub(order - 1)..j], s[j])
            }
        })
        .collect()
}

/// 言語モデル `lm` で、S[a..b] を repl に置き換えたときの対数確率の改善幅。
/// 元の文の対数確率は文ごとに求めた `lp` ([`position_logps`]) を使い回す。
pub(crate) fn delta_pre(
    lm: &dyn LanguageModel,
    s: &[u32],
    lp: &[f32],
    a: usize,
    b: usize,
    repl: &[u32],
    buf: &mut Vec<u32>,
) -> f32 {
    let order = lm.order();
    let ctx_start = a.saturating_sub(order - 1);
    let tail_end = (b + order - 1).min(s.len());
    let orig: f32 = lp[a..tail_end].iter().sum();
    buf.clear();
    buf.extend_from_slice(&s[ctx_start..a]);
    let first = buf.len();
    buf.extend_from_slice(repl);
    buf.extend_from_slice(&s[b..tail_end]);
    let mut new = 0.0;
    for j in first..buf.len() {
        new += lm.logp(&buf[j.saturating_sub(order - 1)..j], buf[j]);
    }
    new - orig
}

/// toks[a..b] の前後どちらかに、かな・漢字に挟まれた空白があるか。
/// 「DX は」「第6 条」のような英数字の後ろの空白は Word の文書でも普通に書くので数えない。
/// 引用・思考の動詞 (「…だと言う」「…と考える」の「と」が抜けやすい)。
const QUOTE_VERBS: &[&str] = &[
    "言う",
    "いう",
    "考える",
    "思う",
    "呼ぶ",
    "述べる",
    "語る",
    "話す",
    "答える",
    "感じる",
    "見る",
    "称する",
];

/// 名詞の後ろ以外で、抜けやすい字を補う位置 (`toks[i]` の直前)。JWTD の開発用で、候補に入っていなかった
/// 実際の誤字に多かった形だけを足す。
/// - 「偽者だ□言う」「阻止する□いう目的」: 活用語の終止形と引用・思考の動詞の間の「と」
/// - 「江戸時代に□みられない」: 格助詞の後ろの係助詞「は」「も」
/// - 「普及□、」「辞任□、」: サ変名詞と読点の間の「し」
/// - 「されて□ない」: 「て」と「ない」の間の「い」(い抜き)
fn extra_insertions(toks: &[Token], i: usize) -> &'static [&'static str] {
    let Some(prev) = i.checked_sub(1).map(|j| &toks[j]) else {
        return &[];
    };
    let t = &toks[i];
    if matches!(prev.pos, "動詞" | "助動詞" | "形容詞")
        && prev.conj_form == "基本形"
        && t.pos == "動詞"
        && QUOTE_VERBS.contains(&t.base)
    {
        return &["と"];
    }
    if prev.pos == "助詞"
        && prev.pos1 == "格助詞"
        && matches!(prev.surface.as_str(), "に" | "で" | "まで" | "から" | "へ")
        && !matches!(t.pos, "助詞" | "記号" | "助動詞")
    {
        return &["は", "も"];
    }
    if prev.pos == "名詞" && prev.pos1 == "サ変接続" && t.pos == "記号" && t.pos1 == "読点"
    {
        return &["し"];
    }
    // IPADIC は「されてない」の「て」を「てる」の未然形 (動詞・非自立) にする
    let te = matches!(prev.surface.as_str(), "て" | "で")
        && (prev.pos == "助詞" || (prev.pos1 == "非自立" && matches!(prev.base, "てる" | "でる")));
    if te && t.base == "ない" {
        return &["い"];
    }
    &[]
}

/// 名詞と名詞をつなぐ「の」(「無料のシャトルバス」「宿泊の施設」) か。消しても残しても文が成立するので、
/// 複合語 (「無料シャトルバス」) の n-gram が強いと誤りに見えやすい。ただし実際の誤字 (JWTD) にも余計な「の」は
/// 多い (開発用 4,883 文で 107 件) ので、候補からは外さず判定器の特徴量にする。
fn is_genitive_between_nouns(toks: &[Token], i: usize) -> bool {
    toks[i].surface == "の"
        && toks[i].pos == "助詞"
        && i > 0
        && toks[i - 1].pos == "名詞"
        && toks
            .get(i + 1)
            .is_some_and(|x| x.pos == "名詞" && !matches!(x.pos1, "非自立" | "接尾"))
}

/// 同じ助詞・接頭辞が 2 つ続く打ち間違い (「土産物はは店」「ごご連絡」) を、2 つ目を消す指摘にする。
///
/// 助詞や接頭辞がそのまま 2 回続く正しい文はほぼ無い (「もも」「ここ」のような語は名詞として 1 語になる) ので、
/// n-gram や判定器を通さずに出す (判定器は複合語の n-gram に引きずられて「は」の削除を嫌うことがある)。
/// 種類は実データ由来のパターンと同じ扱い (閾値 0) にする。
fn repeated_function_words(toks: &[Token]) -> Vec<Finding> {
    toks.windows(2)
        .filter(|w| {
            w[0].surface == w[1].surface
                && w[0].pos == w[1].pos
                && matches!(w[0].pos, "助詞" | "接頭詞")
                && w[0].end == w[1].start
        })
        .map(|w| Finding {
            start: w[1].start,
            end: w[1].end,
            original: w[1].surface.clone(),
            replacement: String::new(),
            kind: EditKind::Pattern,
            delta: REPEATED_WORD_SCORE,
            alternatives: Vec::new(),
        })
        .collect()
}

/// 箇所の候補をスコアの高い順に並べ、直した文が同じになる候補は最も高いものだけを残す。
/// 規則とパターンが同じ「を」の削除を出したり、「をを」のどちらの「を」を消すかで別の候補になったりして、
/// 別案に同じものが並ぶため。
fn dedup_site(sent: &str, site: &mut Vec<Finding>) {
    site.sort_by(|x, y| y.delta.total_cmp(&x.delta));
    let mut seen: Vec<String> = Vec::with_capacity(site.len());
    site.retain(|f| {
        let k = apply_finding(sent, f);
        if seen.contains(&k) {
            false
        } else {
            seen.push(k);
            true
        }
    });
}

/// 列全体の対数確率 (先頭の BOS は条件にだけ使う)。
fn seq_logp(lm: &dyn LanguageModel, ids: &[u32]) -> f32 {
    let order = lm.order();
    (1..ids.len())
        .map(|j| lm.logp(&ids[j.saturating_sub(order - 1)..j], ids[j]))
        .sum()
}

/// `a` にあって `b` に無い要素 (重複も数える)。どちらも昇順に並べたもの。
fn sorted_minus(a: &[u32], b: &[u32]) -> Vec<u32> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < a.len() {
        if j < b.len() && b[j] < a[i] {
            j += 1;
        } else if j < b.len() && b[j] == a[i] {
            i += 1;
            j += 1;
        } else {
            out.push(a[i]);
            i += 1;
        }
    }
    out
}

/// 文字単位の指摘を採るか。`char_delta` は文字モデルでの改善幅、`word_delta` は単語モデルでの文の改善幅 (どちらも log10)。
///
/// 直し方の種類ごとに下限を決める (JWTD の開発用 先頭 5000 件と判例要旨・Wikipedia の正しい文で、
/// 正解の候補を多く残しつつ正しい文での候補を正解の 1 割程度に抑える値)。
/// - 1 字の削除は、正しい文でも文字モデルの尤度が上がりやすい (珍しい固有名詞・専門語の字を消すと
///   自然になる) ので、とくに厳しくし、かなだけにする。漢字 (「再更正」の「再」、「各号」の「各」)・
///   英字・数字・括弧 (「A社」「(普通自動車)」) は、正しい文での候補が正解の倍以上あった
/// - 助詞どうしの置き換え (「の → が」) は単語モデルの受け持ちなので、文字モデルでは出さない
fn char_accepted(original: &str, replacement: &str, char_delta: f32, word_delta: f32) -> bool {
    const PARTICLES: &str = "のにがをはでともへやか";
    let is_kanji_str = |s: &str| s.chars().next().is_some_and(is_kanji);
    let is_word_chars = |s: &str| s.chars().all(|c| is_kanji(c) || is_kana(c));
    let is_particle = |s: &str| s.chars().count() == 1 && PARTICLES.contains(s);
    let (min_char, min_word) = match (original.chars().count(), replacement.chars().count()) {
        (0, _) => (4.0, 3.0),                                  // 補う (脱字)
        (_, 0) if original.chars().all(is_kana) => (8.0, 0.0), // 消す (余分な字)
        (2, 2) if is_word_chars(original) => (6.0, 0.0),       // 入れ替え
        (_, 0) | (2, 2) => return false,
        _ if is_kanji_str(original) => (5.0, 3.0), // 同じ読みの漢字への置き換え
        _ if is_particle(original) && is_particle(replacement) => return false,
        _ => (7.0, 0.0), // かなの置き換え
    };
    char_delta >= min_char && word_delta >= min_word
}

/// 判定器の `#tau_kind` に書く、名詞の間の「の」を消す候補の名前 (他の削除より厳しい閾値にする)。
pub const DELETE_GENITIVE: &str = "delete-gen";

/// 判定器の `#tau_kind` に書く、名詞と 1 字の名詞の間の助詞を消す候補 (「和菓子[は]店」「土産物[から]店」) の名前。
/// 複合語の中に紛れ込んだ助詞は判定器の点数が低く出やすいので、他の削除より緩い閾値にする。
pub const DELETE_BEFORE_SHORT_NOUN: &str = "delete-nsfx";

/// 判定器の `#tau_kind` に書く、係助詞の直前の助詞を消す候補 (「西口側[まで]は」) の名前。
pub const DELETE_BEFORE_KAKARI: &str = "delete-pp";

/// 「されてる」「読んでる」の「て」「で」を消す指摘 (→ される・読む) を、抜けた「い」を補う指摘 (→ されている) に替える。
///
/// どちらの直し方でも文は成り立つが、実際の誤字 (JWTD) では「い」の抜けの方が多く (開発用で 14 件対 6 件)、
/// 書き言葉としても「ている」が自然。
fn restore_dropped_i(sent: &str, findings: &mut [Finding]) {
    let chars: Vec<char> = sent.chars().collect();
    for f in findings {
        if f.replacement.is_empty()
            && matches!(f.original.as_str(), "て" | "で")
            && chars.get(f.end) == Some(&'る')
        {
            f.start = f.end;
            f.original.clear();
            f.replacement.push('い');
        }
    }
}

/// 文字モデルの削除のうち、単語モデルが同じ削除を判断済みなので文字モデルでは拾わないもの。
///
/// 敬語の接頭辞 (「[ご]確認」) と名詞の間の「の」(「費用[の]面」) は、消しても文が成り立つので文字モデルの点数が
/// 高く出やすいが、誤りであることは少ない。単語モデルはこれらを厳しい閾値 (delete-gen) や判定器で見ているので、
/// 文字モデルで拾い直さない (JWTD の検出はそのままで、正しい文での誤検出が減った)。
fn char_deletion_left_to_word_model(toks: &[Token], f: &Finding) -> bool {
    if !f.replacement.is_empty() {
        return false;
    }
    toks.iter()
        .position(|t| t.start == f.start && t.end == f.end)
        .is_some_and(|i| toks[i].pos == "接頭詞" || is_genitive_between_nouns(toks, i))
}

/// 1 語を消す候補のうち、delete と別の閾値で決めるものの分類 (`#tau_kind` の名前)。
///
/// 一般文では、正しい文の助詞を消す誤検出 (「被害[が]軽減」「状態[を]関数」) を抑えるために delete の閾値を厳しくする一方、
/// 判定器の点数が低く出やすい実際の誤り (「和菓子[は]店」「西口側[まで]は」) はこの分類で緩い閾値に残す。
fn delete_class(toks: &[Token], i: usize) -> Option<&'static str> {
    if is_genitive_between_nouns(toks, i) {
        return Some(DELETE_GENITIVE);
    }
    if toks[i].pos != "助詞" {
        return None;
    }
    let next = toks.get(i + 1)?;
    if i > 0 && toks[i - 1].pos == "名詞" && next.pos == "名詞" && next.surface.chars().count() == 1
    {
        Some(DELETE_BEFORE_SHORT_NOUN)
    } else if next.pos1 == "係助詞" {
        Some(DELETE_BEFORE_KAKARI)
    } else {
        None
    }
}

/// 判定器の `#exempt` に書く、「活用語 + 推量の助動詞」をまとめて直す活用の候補の名前。
pub const INFLECTION_AUX: &str = "inflection-aux";

/// パターンの採否のスコア = 直した文の n-gram 対数確率 (log10) の改善幅 + この重み × log10(支持数)。
/// 支持数の多いパターン (「をを」「れいる」) は文脈が多少不自然でも採り、支持数の少ないものは改善幅で確かめる。
const PATTERN_SUPPORT_WEIGHT: f32 = 4.0;
/// パターンの指摘を残すスコアの下限 [法令文, 一般文, 契約書]。
/// 一般文は JWTD の開発用 (先頭 5000 件)・判例要旨 dev・wiki2 で決めた値。3.0 に下げると検出と誤検出が
/// ほぼ 1 対 1 で増えるだけだった (パターンを開発用を含む train から作っていたときは 3.0 が良く見えたが、それは漏れのため)。
/// 法令文・契約書は、支持数の少ないパターンが例規・契約書の言い回しに当たりやすいので厳しめにする
/// (一宮市・JEITA 大の原文での誤検出を増やさない値)。
const PATTERN_MIN_SCORE: [f32; 3] = [5.0, 3.5, 5.0];

/// パターンの指摘を採るか (`log_support` は log10(支持数))。
fn pattern_accepted(d: Domain, sentence_delta: f32, log_support: f32) -> bool {
    let score = sentence_delta + PATTERN_SUPPORT_WEIGHT * log_support;
    let min = PATTERN_MIN_SCORE[match d {
        Domain::Legal => 0,
        Domain::General => 1,
        Domain::Contract => 2,
    }];
    score >= min
}

fn apply_finding(sent: &str, f: &Finding) -> String {
    let mut out = String::with_capacity(sent.len());
    for (i, c) in sent.chars().enumerate() {
        if i == f.start {
            out.push_str(&f.replacement);
        }
        if i < f.start || i >= f.end {
            out.push(c);
        }
    }
    if f.start >= sent.chars().count() {
        out.push_str(&f.replacement);
    }
    out
}

/// 必ず「に」をとる動詞 (原形)。「資料を基づき」「業務を従事する」の「を」は誤り。
const NI_VERBS: &[&str] = &[
    "基づく",
    "従う",
    "関する",
    "対する",
    "応じる",
    "準じる",
    "準ずる",
    "反する",
    "属する",
];
/// 必ず「に」をとるサ変名詞 (「〜する」の形で使うとき)。
const NI_SURU_NOUNS: &[&str] = &[
    "従事", "該当", "違反", "貢献", "依存", "起因", "準拠", "抵触", "寄与", "参画",
];

/// 「に」をとる動詞の直前の「を」を「に」に直す (「資料を基づき」「業務を従事する」「指示を従う」)。
///
/// 動詞が決まれば格助詞も決まる (文法の事実) ので、n-gram や判定器を通さずに出す。判定器は助詞の
/// 置き換えを全般に嫌うので、n-gram の差が大きくても落とすことがあった。
fn wrong_case_before_ni_verbs(toks: &[Token]) -> Vec<Finding> {
    let mut out = Vec::new();
    for (i, t) in toks.iter().enumerate() {
        if t.surface != "を" || t.pos != "助詞" {
            continue;
        }
        let Some(next) = toks.get(i + 1) else {
            continue;
        };
        let verb = next.pos == "動詞" && NI_VERBS.contains(&next.base);
        let suru_noun = next.pos == "名詞"
            && NI_SURU_NOUNS.contains(&next.surface.as_str())
            && toks
                .get(i + 2)
                .is_some_and(|x| x.pos == "動詞" && x.base == "する");
        if verb || suru_noun {
            out.push(Finding {
                start: t.start,
                end: t.end,
                original: t.surface.clone(),
                replacement: "に".to_string(),
                kind: EditKind::Pattern,
                delta: REPEATED_WORD_SCORE,
                alternatives: Vec::new(),
            });
        }
    }
    out
}

/// 「に」を伴って複合助詞になる動詞 (原形)。名詞の直後にあれば「に」が抜けている (「契約関して」→「契約に関して」)。
const NI_COMPOUND_VERBS: &[&str] = &[
    "関する",
    "関す",
    "際する",
    "際す",
    "基づく",
    "伴う",
    "応じる",
    "対する",
    "対す",
];

/// 「という」の後ろに来やすい名詞。「と」の後ろの名詞が述語 (「と規定する」) や次の節の主語 (「と取引者が誤認」)
/// のことも多いので、「という」を補うのはこれらの名詞の前だけにする (判決文の要旨で誤検出が多かった)。
const IU_NOUNS: &[&str] = &[
    "内容",
    "趣旨",
    "意味",
    "理由",
    "話",
    "意見",
    "主張",
    "考え",
    "事実",
    "指摘",
    "見解",
    "立場",
    "方針",
    "目的",
    "条件",
    "前提",
    "結論",
    "認識",
    "感覚",
    "ニュアンス",
    "印象",
    "噂",
    "説",
    "名目",
    "意図",
    "経緯",
    "問題",
    "疑い",
    "評価",
    "批判",
    "声",
    "報道",
    "情報",
    "連絡",
    "通知",
    "記載",
    "表現",
    "言葉",
    "発言",
    "趣き",
];

/// 抜けやすい「に」「いう」を補う (文法の事実なので n-gram や判定器を通さずに出す)。
///
/// - 名詞の直後の「対して」「関して」「ついて」「よって」などには「に」が要る (「北朝鮮対して」「この点ついて」)。
///   「に」を補うと分かち書きが変わる (「に対し」「について」) ので、語単位の n-gram では差が出にくかった
/// - 活用語の終止形 +「と」+ 名詞は「という」の「いう」が抜けている (「改めたいとニュアンス」「合意したと内容」)
fn missing_ni_and_iu(toks: &[Token]) -> Vec<Finding> {
    let mut out = Vec::new();
    let insert = |at: usize, repl: &str| Finding {
        start: at,
        end: at,
        original: String::new(),
        replacement: repl.to_string(),
        kind: EditKind::Pattern,
        delta: REPEATED_WORD_SCORE,
        alternatives: Vec::new(),
    };
    for i in 1..toks.len() {
        let (prev, t) = (&toks[i - 1], &toks[i]);
        let next = toks.get(i + 1);
        // 「すべて応じる」「一切応じない」のように副詞として使う名詞の後ろには「に」が要らない
        if prev.pos == "名詞" && prev.pos1 != "副詞可能" && prev.end == t.start {
            let te_follows = next.is_some_and(|x| x.surface == "て");
            let compound = (t.surface == "対"
                && t.pos1 == "接続詞的"
                && next.is_some_and(|x| x.pos == "動詞" && x.base == "する"))
                || (t.pos == "動詞" && NI_COMPOUND_VERBS.contains(&t.base))
                || (t.pos == "動詞" && t.base == "つく" && t.surface == "つい" && te_follows)
                || (t.pos == "動詞" && t.base == "よる" && t.surface == "よっ" && te_follows);
            if compound {
                out.push(insert(t.start, "に"));
            }
        }
        if t.surface == "と"
            && t.pos == "助詞"
            && t.pos1 == "格助詞"
            && matches!(prev.pos, "動詞" | "助動詞" | "形容詞")
            && prev.conj_form == "基本形"
            && let Some(n) = next
            && n.pos == "名詞"
            && IU_NOUNS.contains(&n.surface.as_str())
            // 「と主張して」「と発言した」のように動詞として使うときは出さない
            && !toks
                .get(i + 2)
                .is_some_and(|x| x.pos == "動詞" && matches!(x.base, "する" | "できる" | "出来る"))
        {
            out.push(insert(n.start, "いう"));
        }
    }
    out
}

/// 助詞・接頭辞の重複の指摘のスコア (パターンのスコア = log10(支持数) に合わせ、支持数 100 相当)。
const REPEATED_WORD_SCORE: f32 = 2.0;

/// IPADIC で名詞になっている「お + 形容詞」(お忙しい・お美しい・お寂しい)。
fn is_honorific_adjective(s: &str) -> bool {
    (s.starts_with('お') || s.starts_with('ご')) && s.ends_with("しい") && s.chars().count() >= 4
}

fn touches_space(toks: &[Token], a: usize, b: usize) -> bool {
    let ja = |c: Option<char>| c.is_some_and(|c| is_kana(c) || is_kanji(c));
    let gap = |l: &Token, r: &Token| {
        l.end < r.start && ja(l.surface.chars().last()) && ja(r.surface.chars().next())
    };
    (a > 0 && gap(&toks[a - 1], &toks[a])) || (b < toks.len() && gap(&toks[b - 1], &toks[b]))
}

/// toks[i] が「A も B も」の並列の「も」か (同じ文の近く (前後 6 語以内) に別の「も」がある)。
/// 古い表記 (促音・拗音を小さく書かない「あつた」「なつた」) を直す指摘のうち、文書の中で 2 か所以上あるものは
/// 書き手の表記として出さない (古い判決の要旨などで、文書全体がこの表記になっている)。1 か所だけなら打ち間違いとして残す。
fn drop_old_style_small_kana(found: Vec<Finding>) -> Vec<Finding> {
    let is_small_kana_fix = |f: &Finding| {
        matches!(
            (f.original.as_str(), f.replacement.as_str()),
            ("つ", "っ") | ("や", "ゃ") | ("ゆ", "ゅ") | ("よ", "ょ")
        )
    };
    if found.iter().filter(|f| is_small_kana_fix(f)).count() < 2 {
        return found;
    }
    found
        .into_iter()
        .filter(|f| !is_small_kana_fix(f))
        .collect()
}

/// 並列の「と」(「代金と費用とを」「発症と事故との間」)。近くにもう 1 つ「と」があり、間に句読点が無いもの。
/// どちらの「と」を消しても文が成立するので、n-gram では誤りに見えやすい (判決文の要旨で誤検出が多かった)。
fn is_parallel_to(toks: &[Token], i: usize) -> bool {
    let is_to = |t: &Token| t.surface == "と" && t.pos == "助詞";
    if !is_to(&toks[i]) {
        return false;
    }
    let lo = i.saturating_sub(6);
    let hi = (i + 7).min(toks.len());
    (lo..hi).any(|j| {
        j != i
            && is_to(&toks[j])
            && !toks[j.min(i)..j.max(i)]
                .iter()
                .any(|t| matches!(t.surface.as_str(), "。" | "、"))
    })
}

fn is_parallel_mo(toks: &[Token], i: usize) -> bool {
    let is_mo = |t: &Token| t.surface == "も" && t.pos == "助詞";
    if !is_mo(&toks[i]) {
        return false;
    }
    let lo = i.saturating_sub(6);
    let hi = (i + 7).min(toks.len());
    (lo..hi).any(|j| {
        j != i
            && is_mo(&toks[j])
            // 間に句点・読点を挟むものは別の節なので並列とはみなさない
            && !toks[j.min(i)..j.max(i)]
                .iter()
                .any(|t| matches!(t.surface.as_str(), "。" | "、"))
    })
}

pub fn load_inflections(
    path: &std::path::Path,
    keep: &dyn Fn(&str) -> bool,
) -> anyhow::Result<FxHashMap<(String, String), Vec<String>>> {
    // ファイル全体を文字列にせず 1 行ずつ読む (読み込み時の一時メモリを抑える)
    load_inflections_from_reader(std::io::BufReader::new(std::fs::File::open(path)?), keep)
}

/// [`load_inflections`] の読み込み元を任意の reader にしたもの (ブラウザではバイト列から読む)。
pub fn load_inflections_from_reader(
    reader: impl std::io::BufRead,
    keep: &dyn Fn(&str) -> bool,
) -> anyhow::Result<FxHashMap<(String, String), Vec<String>>> {
    let mut m: FxHashMap<(String, String), Vec<String>> = FxHashMap::default();
    for line in reader.lines() {
        let line = line?;
        let mut it = line.split('\t');
        if let (Some(b), Some(t), Some(s)) = (it.next(), it.next(), it.next())
            && keep(s)
        {
            m.entry((b.to_string(), t.to_string()))
                .or_default()
                .push(s.to_string());
        }
    }
    m.shrink_to_fit();
    Ok(m)
}

/// 同音異字表 (TSV: 読み \t 表層形 \t 出現数) を読む。`keep` は [`load_inflections`] と同じ。
/// 候補に使うのは出現 20 回以上の語だけなので、それ未満もここで落とす。
pub fn load_readings(
    path: &std::path::Path,
    keep: &dyn Fn(&str) -> bool,
) -> anyhow::Result<FxHashMap<String, Vec<(String, u32)>>> {
    load_readings_from_reader(std::io::BufReader::new(std::fs::File::open(path)?), keep)
}

/// [`load_readings`] の読み込み元を任意の reader にしたもの (ブラウザではバイト列から読む)。
pub fn load_readings_from_reader(
    reader: impl std::io::BufRead,
    keep: &dyn Fn(&str) -> bool,
) -> anyhow::Result<FxHashMap<String, Vec<(String, u32)>>> {
    let mut m: FxHashMap<String, Vec<(String, u32)>> = FxHashMap::default();
    for line in reader.lines() {
        let line = line?;
        let mut it = line.split('\t');
        if let (Some(r), Some(s), Some(c)) = (it.next(), it.next(), it.next())
            && keep(s)
            && c.parse::<u32>().unwrap_or(0) >= 20
        {
            m.entry(r.to_string())
                .or_default()
                .push((s.to_string(), c.parse().unwrap_or(0)));
        }
    }
    // 読みに 1 語しか無ければ同音異字の候補にならない
    m.retain(|_, v| v.len() >= 2);
    for v in m.values_mut() {
        v.sort_by_key(|e| std::cmp::Reverse(e.1));
        v.shrink_to_fit();
    }
    m.shrink_to_fit();
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<Token> {
        Tokenizer::new()
            .expect("辞書 (data/ipadic.dic) が必要")
            .tokenize(s)
    }

    fn index_of(t: &[Token], surface: &str, nth: usize) -> usize {
        t.iter()
            .enumerate()
            .filter(|(_, x)| x.surface == surface)
            .nth(nth)
            .map(|(i, _)| i)
            .unwrap()
    }

    #[test]
    fn mo_in_parallel_phrase_is_parallel() {
        let t = toks("ユーザもベンダも責任を負う。");
        assert!(is_parallel_mo(&t, index_of(&t, "も", 0)));
        assert!(is_parallel_mo(&t, index_of(&t, "も", 1)));
    }

    #[test]
    fn single_mo_is_not_parallel() {
        let t = toks("ベンダも責任を負う。");
        assert!(!is_parallel_mo(&t, index_of(&t, "も", 0)));
        // 読点を挟んだ別の節の「も」は並列ではない
        let t = toks("ユーザも、ベンダも責任を負う。");
        assert!(!is_parallel_mo(&t, index_of(&t, "も", 0)));
    }

    /// 候補生成だけを見るための最小の Checker (言語モデルは 1 文から作る)。
    fn tiny_checker() -> anyhow::Result<Checker> {
        let dir = std::env::temp_dir().join(format!("celso-checker-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let corpus = dir.join("corpus.txt");
        std::fs::write(&corpus, "宿泊 施設 が ある 。\n")?;
        let lm = crate::lm::build(
            &[&corpus],
            &crate::lm::BuildConfig {
                order: 3,
                min_word_count: 1,
                min_count: [1; crate::lm::MAX_ORDER],
                vocab: None,
                keep_words: None,
                keep_min_count: [1; crate::lm::MAX_ORDER],
            },
        )?;
        Ok(Checker::new(
            Tokenizer::new()?,
            Box::new(lm),
            Config::default(),
            FxHashMap::default(),
            FxHashMap::default(),
        ))
    }

    fn has_delete(c: &Checker, t: &[Token], surface: &str) -> bool {
        let i = index_of(t, surface, 0);
        c.candidates(t)
            .iter()
            .any(|x| x.a == i && x.b == i + 1 && x.kind == EditKind::Delete)
    }

    #[test]
    fn kakari_after_case_particle_is_not_deleted() {
        let Ok(c) = tiny_checker() else { return };
        let t = toks("瑕疵についてもベンダは責任を負う。");
        assert!(!has_delete(&c, &t, "も"));
        let t = toks("ベンダからは回答がない。");
        assert!(!has_delete(&c, &t, "から"));
        assert!(!has_delete(&c, &t, "は"));
        // 「西口側までは」の「まで」は消す候補に残す
        let t = toks("西口側までは宿泊施設がある。");
        assert!(has_delete(&c, &t, "まで"));
        // 期限の「までに」は消さない
        let t = toks("次回までに案を示す。");
        assert!(!has_delete(&c, &t, "まで"));
        // 名詞の直後の余計な「は」は従来どおり削除候補にする (「飲食は店」)
        let t = toks("飲食は店がある。");
        assert!(has_delete(&c, &t, "は"));
    }

    fn has_insert_before(c: &Checker, t: &[Token], surface: &str) -> bool {
        let i = index_of(t, surface, 0);
        c.candidates(t)
            .iter()
            .any(|x| x.a == i && x.b == i && x.kind == EditKind::Insert)
    }

    #[test]
    fn no_particle_is_inserted_after_honorific_adjective() {
        let Ok(c) = tiny_checker() else { return };
        // IPADIC では「お忙しい」が名詞だが、「お忙しいのところ」にはしない
        let t = toks("お忙しいところ恐縮です。");
        assert!(!has_insert_before(&c, &t, "ところ"));
        // 普通の名詞の後ろには従来どおり補う候補を出す (「宿泊□施設」)
        let t = toks("宿泊施設がある。");
        assert!(has_insert_before(&c, &t, "施設"));
    }

    fn has_inflection(c: &Checker, t: &[Token], surface: &str) -> bool {
        let i = index_of(t, surface, 0);
        c.candidates(t)
            .iter()
            .any(|x| x.a == i && x.kind == EditKind::Inflection)
    }

    /// 活用表に `text` の活用語 (原形, 活用型) と、ダミーの活用形を 1 つずつ入れた Checker。
    fn checker_with_inflections_of(text: &str) -> anyhow::Result<(Checker, Vec<Token>)> {
        let mut c = tiny_checker()?;
        let t = toks(text);
        for x in &t {
            if !x.conj_type.is_empty() {
                c.inflections.insert(
                    (x.base.to_string(), x.conj_type.to_string()),
                    vec![x.base.to_string(), "ダミー".into()],
                );
            }
        }
        Ok((c, t))
    }

    #[test]
    fn imperative_and_colloquial_forms_are_not_inflection_errors() {
        // 判決主文の命令形
        let Ok((c, t)) = checker_with_inflections_of("被告は原告に金100万円を支払え。")
        else {
            return;
        };
        assert!(!has_inflection(&c, &t, "支払え"));
        // 文中の命令形は打ち間違いの可能性があるので候補に残す
        let Ok((c, t)) = checker_with_inflections_of("金100万円を支払え義務を負う。")
        else {
            return;
        };
        assert!(has_inflection(&c, &t, "支払え"));
        // い抜き (話し言葉の縮約)
        let Ok((c, t)) = checker_with_inflections_of("整備を進めてる。") else {
            return;
        };
        assert!(!has_inflection(&c, &t, "てる"));
        // 「多くあろう」は従来どおり活用の候補にする
        let Ok((c, t)) = checker_with_inflections_of("飲食店など多くあろう") else {
            return;
        };
        assert!(has_inflection(&c, &t, "あろ"));
    }

    #[test]
    fn genitive_no_between_nouns_is_a_reranker_feature() {
        let Ok(c) = tiny_checker() else { return };
        let t = toks("駅から無料のシャトルバスが出る。");
        // 候補には残し (実際の誤字にも余計な「の」は多い)、判定器に「名詞の間の『の』」と伝える
        assert!(has_delete(&c, &t, "の"));
        let i = index_of(&t, "の", 0);
        let cand = Cand {
            a: i,
            b: i + 1,
            repl: None,
            kind: EditKind::Delete,
        };
        let sig = Signals {
            ngram: 1.0,
            aux: None,
            cooc: None,
            novel: 1,
            sent_novel: 0.0,
        };
        let f = c.features(&t, &cand, Domain::General, &sig);
        assert!(f.iter().any(|(k, _)| k == "delete:gen_nn"), "{f:?}");
        // 動詞の後ろの余計な「の」(「課するのもの」) は従来どおり消す候補にする
        let t = toks("個人に対して課するのものとする。");
        assert!(has_delete(&c, &t, "の"));
    }

    #[test]
    fn deletions_are_classified_for_their_own_thresholds() {
        let t = toks("駅から無料のシャトルバスが出る。");
        assert_eq!(
            delete_class(&t, index_of(&t, "の", 0)),
            Some(DELETE_GENITIVE)
        );
        // 名詞と 1 字の名詞の間の余計な助詞 (「和菓子は店」) は、他の削除より緩い閾値で拾う
        let t = toks("和菓子は店や酒蔵が多い。");
        assert_eq!(
            delete_class(&t, index_of(&t, "は", 0)),
            Some(DELETE_BEFORE_SHORT_NOUN)
        );
        // 係助詞の直前の余計な助詞 (「西口側までは」)
        let t = toks("西口側までは宿泊施設が多い。");
        assert_eq!(
            delete_class(&t, index_of(&t, "まで", 0)),
            Some(DELETE_BEFORE_KAKARI)
        );
        // それ以外の削除 (「被害が軽減」) は delete の閾値のまま
        let t = toks("被害が軽減できる。");
        assert_eq!(delete_class(&t, index_of(&t, "が", 0)), None);
    }

    #[test]
    fn char_model_leaves_prefix_and_genitive_deletions_to_word_model() {
        let del = |s: &str, start: usize, end: usize, original: &str| {
            let f = Finding {
                start,
                end,
                original: original.to_string(),
                replacement: String::new(),
                kind: EditKind::Pattern,
                delta: 0.0,
                alternatives: Vec::new(),
            };
            char_deletion_left_to_word_model(&toks(s), &f)
        };
        // 敬語の接頭辞 (「ご確認」の「ご」) と名詞の間の「の」(「費用の面」) は、単語モデルの判断に任せる
        assert!(del("内容をご確認のうえ、ご返信ください。", 3, 4, "ご"));
        assert!(del("いずれの案も、費用の面で課題がある。", 9, 10, "の"));
        // 語の中の余分な字 (「離脱抜した」の「抜」) は文字モデルで拾う
        assert!(!del("親子でも離脱抜した場合は", 6, 7, "抜"));
    }

    #[test]
    fn dropped_i_after_te_is_restored_instead_of_deleting_te() {
        let del = |start: usize, original: &str| Finding {
            start,
            end: start + 1,
            original: original.to_string(),
            replacement: String::new(),
            kind: EditKind::Pattern,
            delta: 0.0,
            alternatives: Vec::new(),
        };
        // 「挙行されてる」の「て」を消す (される) より、抜けた「い」を補う (されている) 方を示す
        let mut fs = vec![del(10, "て")];
        restore_dropped_i("日本武道館で挙行されてる。", &mut fs);
        assert_eq!((fs[0].start, fs[0].end), (11, 11));
        assert_eq!(
            (fs[0].original.as_str(), fs[0].replacement.as_str()),
            ("", "い")
        );
        // 「読んでる」の「で」も同じ
        let mut fs = vec![del(4, "で")];
        restore_dropped_i("本を読んでる。", &mut fs);
        assert_eq!(fs[0].replacement, "い");
        // 後ろが「る」でなければそのまま (「されて、」)
        let mut fs = vec![del(4, "て")];
        restore_dropped_i("挙行されて、", &mut fs);
        assert_eq!(fs[0].original, "て");
    }

    #[test]
    fn repeated_particle_or_prefix_is_removed() {
        let f = repeated_function_words(&toks("土産物はは店が並ぶ。"));
        assert_eq!(f.len(), 1);
        assert_eq!(
            (
                f[0].start,
                f[0].end,
                f[0].original.as_str(),
                f[0].replacement.as_str()
            ),
            (4, 5, "は", "")
        );
        let f = repeated_function_words(&toks("折り返しごご連絡いたします。"));
        assert_eq!(f.len(), 1);
        assert_eq!((f[0].start, f[0].end), (5, 6));
        // 重複の無い文や、同じ字が語の一部として続くだけの文には出さない
        assert!(
            repeated_function_words(&toks("ここには桃もももある。"))
                .iter()
                .all(|x| x.original != "こ")
        );
        assert!(repeated_function_words(&toks("ごみを捨てて帰る。")).is_empty());
        assert!(repeated_function_words(&toks("会議では予算について議論した。")).is_empty());
    }

    #[test]
    fn pattern_acceptance_weighs_sentence_delta_and_support() {
        let g = Domain::General;
        // 支持数 1000 件 (log10 = 3) なら、文の尤度が少し下がっても採る
        assert!(pattern_accepted(g, -2.0, 3.0));
        // 支持数 2 件 (log10 ≒ 0.3) は、文の尤度が十分に上がるときだけ採る
        assert!(!pattern_accepted(g, 1.0, 0.3));
        assert!(pattern_accepted(g, 3.0, 0.3));
        // 一般文の下限は 3.5 (改善幅 2.4 + 4 × 0.3 = 3.6 は採り、法令文では捨てる)
        assert!(pattern_accepted(g, 2.4, 0.3));
        assert!(!pattern_accepted(g, 2.2, 0.3));
        assert!(!pattern_accepted(Domain::Legal, 2.4, 0.3));
        // 「いたしまします → いたしまいます」(文の Δ が大きく負) はどの文書でも捨てる
        assert!(!pattern_accepted(g, -12.0, 0.78));
        assert!(!pattern_accepted(Domain::Legal, -12.0, 0.78));
        // 法令文・契約書は一般文より厳しい
        assert!(pattern_accepted(g, 4.0, 0.0));
        assert!(!pattern_accepted(Domain::Legal, 4.0, 0.0));
        assert!(pattern_accepted(Domain::Contract, 0.0, 1.5));
    }

    #[test]
    fn char_findings_use_thresholds_per_edit_type() {
        // 熟語の中の同じ読みの漢字 (「骨董品 300 店 → 点」): 文字・単語の両方で改善するときだけ
        assert!(char_accepted("店", "点", 6.0, 3.5));
        assert!(!char_accepted("店", "点", 6.0, 1.0));
        // 脱字 (「すると[い]うような」)
        assert!(char_accepted("", "い", 5.0, 4.0));
        assert!(!char_accepted("", "い", 3.5, 4.0));
        // かなの余分な字は大きく改善するときだけ。漢字・英字・括弧は消さない
        assert!(char_accepted("れ", "", 9.0, 0.0));
        assert!(!char_accepted("れ", "", 7.0, 0.0));
        assert!(!char_accepted("再", "", 12.0, 8.0));
        assert!(!char_accepted("A", "", 12.0, 8.0));
        assert!(!char_accepted(")", "", 12.0, 8.0));
        // 助詞どうしの置き換えは単語モデルに任せる
        assert!(!char_accepted("の", "が", 9.0, 5.0));
        assert!(char_accepted("れ", "て", 8.0, 0.0));
        // 入れ替えはかな・漢字だけ
        assert!(char_accepted("オデ", "デオ", 7.0, 0.0));
        assert!(!char_accepted(")年", "年)", 7.0, 3.0));
    }

    #[test]
    fn same_edit_from_different_sources_is_listed_once() {
        let f = |delta: f32, kind: EditKind| Finding {
            start: 3,
            end: 4,
            original: "を".into(),
            replacement: String::new(),
            kind,
            delta,
            alternatives: Vec::new(),
        };
        let mut site = vec![
            f(2.0, EditKind::Pattern),
            f(3.8, EditKind::Pattern),
            f(1.6, EditKind::Delete),
        ];
        // 1 つ目の「を」を消す候補も、直した文は同じ
        site.push(Finding {
            start: 2,
            end: 3,
            ..f(1.0, EditKind::Delete)
        });
        dedup_site("資料をを読む。", &mut site);
        assert_eq!(site.len(), 1);
        assert!((site[0].delta - 3.8).abs() < 1e-6);
    }

    #[test]
    fn apply_finding_replaces_by_char_offsets() {
        let f = |start, end, r: &str| Finding {
            start,
            end,
            original: String::new(),
            replacement: r.into(),
            kind: EditKind::Pattern,
            delta: 0.0,
            alternatives: Vec::new(),
        };
        assert_eq!(apply_finding("されいる", &f(2, 2, "て")), "されている");
        assert_eq!(apply_finding("資料をを読む", &f(3, 4, "")), "資料を読む");
        assert_eq!(apply_finding("利点ある", &f(2, 2, "が")), "利点がある");
        // 文末への挿入
        assert_eq!(
            apply_finding("目的としてい", &f(6, 6, "る")),
            "目的としている"
        );
    }

    #[test]
    fn homophone_features_include_frequency_prior_and_shared_kanji() {
        let Ok(mut c) = tiny_checker() else { return };
        let t = toks("対象を比較する。");
        let reading = t[0].reading;
        assert_ne!(reading, "");
        // 「対照」は「対象」より 10 倍少ない
        c.readings.insert(
            reading.to_string(),
            vec![("対象".into(), 1000), ("対照".into(), 99)],
        );
        let cand = Cand {
            a: 0,
            b: 1,
            repl: Some("対照"),
            kind: EditKind::Homophone,
        };
        let sig = Signals {
            ngram: 1.0,
            aux: None,
            cooc: None,
            novel: 1,
            sent_novel: 0.25,
        };
        let f = c.features(&t, &cand, Domain::General, &sig);
        let get = |name: &str| f.iter().find(|(k, _)| k == name).map(|(_, v)| *v);
        // log10((99+1)/(1000+1)) ≒ -1
        assert!((get("homophone:freq").unwrap() + 1.0).abs() < 0.01, "{f:?}");
        assert_eq!(get("homophone:share"), Some(1.0));
        assert_eq!(get("homophone:sn2"), Some(1.0));
    }

    fn has_insert(c: &Checker, t: &[Token], before: &str, repl: &str) -> bool {
        let i = index_of(t, before, 0);
        c.candidates(t)
            .iter()
            .any(|x| x.a == i && x.b == i && x.kind == EditKind::Insert && x.repl == Some(repl))
    }

    #[test]
    fn inserts_common_missing_characters_outside_noun_positions() {
        let Ok(c) = tiny_checker() else { return };
        // 引用の「と」
        let t = toks("彼が偽者だ言う場面がある。");
        assert!(has_insert(&c, &t, "言う", "と"));
        // 格助詞の後ろの「は」
        let t = toks("江戸時代にみられない。");
        assert!(has_insert(&c, &t, "み", "は"));
        // サ変名詞と読点の間の「し」
        let t = toks("技術が急速に発展、世界に広まった。");
        assert!(has_insert(&c, &t, "、", "し"));
        // い抜き
        let t = toks("まだ公開されてない。");
        assert!(has_insert(&c, &t, "ない", "い"));
        // 正しい形には余計な候補を出さない (「と言う」の「と」の後ろに「と」は補わない)
        let t = toks("彼が偽者だと言う。");
        assert!(!has_insert(&c, &t, "言う", "と"));
    }

    #[test]
    fn wo_before_verbs_that_take_ni_is_replaced() {
        let fixes = |s: &str| -> Vec<(usize, String)> {
            wrong_case_before_ni_verbs(&toks(s))
                .into_iter()
                .map(|f| (f.start, f.replacement))
                .collect()
        };
        assert_eq!(fixes("資料を基づき説明した。"), [(2, "に".to_string())]);
        assert_eq!(
            fixes("他の業務を従事してはならない。"),
            [(4, "に".to_string())]
        );
        assert_eq!(fixes("甲の指示を従う。"), [(4, "に".to_string())]);
        assert_eq!(fixes("法令を違反した。"), [(2, "に".to_string())]);
        // 正しい文や、サ変名詞が「する」を伴わない形には出さない
        assert_eq!(fixes("資料に基づき説明した。"), []);
        assert_eq!(fixes("違反を是正する。"), []);
        assert_eq!(fixes("従事を命じる。"), []);
        assert_eq!(fixes("部下を従える。"), []);
    }

    #[test]
    fn parallel_to_and_emphatic_made_are_not_deleted() {
        let Ok(c) = tiny_checker() else { return };
        // 並列の「と」(「AとBとを」「AとBとの間」) はどちらも消さない
        let t = toks("代金と費用とを提供した。");
        assert!(!has_delete(&c, &t, "と"));
        let t = toks("発症と事故との間に因果関係がある。");
        assert!(!has_delete(&c, &t, "と"));
        // 「〜とまで」の「まで」は強調なので消さない
        let t = toks("違反したとまでいうことはできない。");
        assert!(!has_delete(&c, &t, "まで"));
        // 並列でない「と」は従来どおり消す候補にする
        let t = toks("彼と話した。");
        assert!(has_delete(&c, &t, "と"));
    }

    #[test]
    fn old_style_large_tsu_is_kept_when_used_throughout_the_document() {
        let f = |start: usize| Finding {
            start,
            end: start + 1,
            original: "つ".into(),
            replacement: "っ".into(),
            kind: EditKind::Pattern,
            delta: 1.0,
            alternatives: Vec::new(),
        };
        // 文書の中で 2 か所以上あれば、古い表記として指摘しない
        assert!(drop_old_style_small_kana(vec![f(3), f(10)]).is_empty());
        // 1 か所だけなら打ち間違いの可能性があるので残す
        assert_eq!(drop_old_style_small_kana(vec![f(3)]).len(), 1);
    }

    #[test]
    fn missing_ni_before_compound_particles_is_inserted() {
        let fixes = |s: &str| -> Vec<(usize, String)> {
            missing_ni_and_iu(&toks(s))
                .into_iter()
                .map(|f| (f.start, f.replacement))
                .collect()
        };
        assert_eq!(fixes("北朝鮮対して融和的だ。"), [(3, "に".to_string())]);
        assert_eq!(fixes("契約関して定める。"), [(2, "に".to_string())]);
        assert_eq!(fixes("この点ついて説明する。"), [(3, "に".to_string())]);
        // 「に」がある文や、「つく」の別の意味には出さない
        assert_eq!(fixes("北朝鮮に対して融和的だ。"), []);
        assert_eq!(fixes("この点について説明する。"), []);
        assert_eq!(fixes("傷がついている。"), []);
        // 副詞のように使う名詞 (「すべて」「一切」) の後ろは「に」が要らない
        assert_eq!(fixes("要求にすべて応じることはできない。"), []);
        assert_eq!(fixes("和解にも一切応じなかった。"), []);
    }

    #[test]
    fn missing_iu_after_quotative_to_is_inserted() {
        let fixes = |s: &str| -> Vec<(usize, String)> {
            missing_ni_and_iu(&toks(s))
                .into_iter()
                .map(|f| (f.start, f.replacement))
                .collect()
        };
        // 「改めたいとニュアンス」「合意したと内容」は「という」の「いう」が抜けている
        assert_eq!(
            fixes("改めたいとニュアンスがある。"),
            [(5, "いう".to_string())]
        );
        assert_eq!(
            fixes("合意したと内容を確認した。"),
            [(5, "いう".to_string())]
        );
        // 正しい言い方には出さない
        assert_eq!(fixes("合意したという内容を確認した。"), []);
        assert_eq!(fixes("到着すると同時に連絡した。"), []);
        assert_eq!(fixes("彼と内容を確認した。"), []);
        // 「と」の後ろの名詞が述語 (「と規定する」「と信頼する」) や次の節の主語 (「と取引者が誤認」) のときは出さない
        assert_eq!(fixes("手当金を支給すると規定するにとどまる。"), []);
        assert_eq!(fixes("真実であると取引者が誤認する。"), []);
        assert_eq!(fixes("義務を負うと主張して提訴した。"), []);
    }

    #[test]
    fn mixed_kana_spelling_is_notation_variant() {
        assert!(is_notation_variant("分", "按分"));
        assert!(is_notation_variant("漏洩", "漏"));
        assert!(is_notation_variant("当", "当"));
        assert!(!is_notation_variant("改訂", "改定"));
        assert!(!is_notation_variant("追求", "追及"));
    }

    #[test]
    fn listed_synonymous_spellings_are_notation_variants() {
        assert!(is_notation_variant("交代", "交替"));
        assert!(is_notation_variant("稼働", "稼動"));
        assert!(is_notation_variant("生", "活"));
        // 意味の違う異字同訓は誤変換として扱う
        assert!(!is_notation_variant("意思", "意志"));
        assert!(!is_notation_variant("配布", "配付"));
    }

    #[test]
    fn edit_next_to_space_touches_space() {
        let t = toks("使用さ れる画面");
        let i = index_of(&t, "さ", 0);
        assert!(touches_space(&t, i, i + 1));
        let t = toks("使用される画面");
        let i = index_of(&t, "さ", 0);
        assert!(!touches_space(&t, i, i + 1));
        // 英字の後ろの空白は Word の文書でも普通なので対象外
        let t = toks("IPA から具体的対策が");
        let i = index_of(&t, "から", 0);
        assert!(!touches_space(&t, i, i + 1));
    }
}
