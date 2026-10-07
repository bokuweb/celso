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
    if hits * 1000 >= chars * 2 {
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
    /// 文書の種類を固定する (None なら文書ごとに自動判定)
    pub domain: Option<Domain>,
    pub enable_insert: bool,
    /// 元の文の編集箇所まわりに、この次数の n-gram として「未出現」の並びがあるときだけ指摘する。
    /// 正しい文どうしの言い換え (「期間が」⇔「期間の」) を拾わないための足切り。0 で無効。
    pub novelty_order: usize,
    /// 指摘箇所の前後を含む文字列が文書内にこの回数以上あれば指摘しない (0 で無効)。
    pub doc_repeat_limit: usize,
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
        t.insert(EditKind::Delete, 3.0);
        t.insert(EditKind::Substitute, 3.0);
        t.insert(EditKind::Inflection, 2.0);
        t.insert(EditKind::Insert, 1.5);
        t.insert(EditKind::Homophone, 3.0);
        // 文字単位の編集は遅く誤検出も多いので既定では無効 (README 参照)
        t.insert(EditKind::Char, f32::INFINITY);
        Self {
            general_thresholds: t.iter().map(|(k, v)| (*k, v + 1.0)).collect(),
            thresholds: t,
            domain: None,
            enable_insert: true,
            novelty_order: 3,
            doc_repeat_limit: 2,
            mlm_weight: 1.0,
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
const INSERT_PARTICLES: &[&str] = &["の", "を", "に", "が", "は", "で", "と"];

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
    /// 文単位の結果キャッシュ (正規化済みの文のハッシュ → その文の指摘)。
    /// 指摘は文の中身だけで決まる (文書内繰り返しの抑制は文書全体で毎回かけ直す) ので、
    /// 編集されていない文は再計算しなくてよい。
    cache: Option<std::sync::RwLock<FxHashMap<u64, Vec<Finding>>>>,
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

struct Cand {
    a: usize,
    b: usize,
    repl: Vec<String>,
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
            cache: None,
        }
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
        found
            .into_iter()
            .filter(|f| self.doc_repeats(&normalized, &chars, f, 0) < self.cfg.doc_repeat_limit)
            .collect()
    }

    /// 複数のテキストをまとめて検査する。MLM の採点要求を全テキスト分まとめて大きなバッチで流すので、
    /// 1 つずつ `check` するより速い (評価用)。
    pub fn check_many(&self, texts: &[&str]) -> Vec<Vec<Finding>> {
        use rayon::prelude::*;
        let dbg = std::env::var_os("CELSO_DEBUG").is_some();
        let t0 = std::time::Instant::now();
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
                // 空白も区切りにする: 条例の「第41条　固定資産税は」のようにラベルと本文を空白で分ける書き方が多く、
                // ひと続きの文として採点すると「第41条」直後の語が不自然に見えてしまう。
                let end_here = i == chars.len()
                    || chars[i] == '\n'
                    || chars[i] == '。'
                    || chars[i].is_whitespace();
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
                t0.elapsed(),
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
            done[i] = Some(self.finalize(sites, sents[i].3));
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
            eprintln!("  compute+store {:.2?}", t0.elapsed());
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
    fn stage1(&self, sent: &str, d: Domain) -> Vec<Vec<Finding>> {
        let toks = self.tok.tokenize(sent);
        if toks.is_empty() {
            return Vec::new();
        }
        let ids = self.ids_of(&toks);
        let slack = if self.mlm.is_some() {
            self.cfg.stage1_slack
        } else {
            0.0
        };

        let mut cands: Vec<Finding> = Vec::new();
        let mut buf: Vec<u32> = Vec::with_capacity(32);
        for c in self.candidates(&toks) {
            let repl_ids: Vec<u32> = c.repl.iter().map(|w| self.lm.word_id(w)).collect();
            if repl_ids.contains(&UNK) {
                continue;
            }
            let th = self.threshold(c.kind, d) - slack;
            if !th.is_finite() || !self.is_novel(&ids, c.a + 1, c.b + 1) {
                continue;
            }
            let delta = self.delta(&ids, c.a + 1, c.b + 1, &repl_ids, &mut buf);
            if delta >= th {
                let start = toks
                    .get(c.a)
                    .map(|t| t.start)
                    .unwrap_or_else(|| toks.last().unwrap().end);
                let end = if c.b > c.a { toks[c.b - 1].end } else { start };
                cands.push(Finding {
                    start,
                    end,
                    original: toks[c.a..c.b].iter().map(|t| t.surface.as_str()).collect(),
                    replacement: c.repl.concat(),
                    kind: c.kind,
                    delta,
                    alternatives: Vec::new(),
                });
            }
        }
        if self.threshold(EditKind::Char, d).is_finite() {
            cands.extend(self.char_edits(sent, &toks, &ids, d));
        }
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
            site.sort_by(|x, y| y.delta.total_cmp(&x.delta));
            site.truncate(self.cfg.top_k);
        }
        sites
    }

    /// 箇所ごとに最良案のスコアが閾値を超えたら指摘し、残りは別案として添える。
    fn finalize(&self, sites: Vec<Vec<Finding>>, d: Domain) -> Vec<Finding> {
        let mut out = Vec::new();
        for mut site in sites {
            site.sort_by(|x, y| y.delta.total_cmp(&x.delta));
            let best = &site[0];
            if best.delta < self.threshold(best.kind, d) {
                continue;
            }
            let mut f = site[0].clone();
            f.alternatives = site[1..]
                .iter()
                .filter(|a| a.delta >= self.threshold(a.kind, d) - 1.0)
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
        let t = match d {
            Domain::Legal => &self.cfg.thresholds,
            Domain::General => &self.cfg.general_thresholds,
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
        let order = self.lm.order();
        (1..ids.len())
            .map(|j| self.lm.logp(&ids[j.saturating_sub(order - 1)..j], ids[j]))
            .sum()
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
    fn is_novel(&self, s: &[u32], a: usize, b: usize) -> bool {
        let n = self.cfg.novelty_order;
        if n == 0 {
            return true;
        }
        let lo = a.saturating_sub(1);
        let hi = (b + 2).min(s.len());
        if s[lo..hi].contains(&UNK) {
            return false;
        }
        // 編集箇所の語と、その直後 2 語までを、直前 n-1 語の文脈で予測したときに完全一致があるか
        (a..hi).any(|j| {
            let ctx = &s[j.saturating_sub(n - 1)..j];
            self.lm.match_order(ctx, s[j]) < n.min(ctx.len() + 1)
        })
    }

    /// S[a..b] を repl に置き換えたときの対数確率の改善幅。
    fn delta(&self, s: &[u32], a: usize, b: usize, repl: &[u32], buf: &mut Vec<u32>) -> f32 {
        let order = self.lm.order();
        let ctx_start = a.saturating_sub(order - 1);
        // 元: S[a .. b + order - 1) を採点 (文末で打ち切り)
        let tail_end = (b + order - 1).min(s.len());
        let mut orig = 0.0;
        for j in a..tail_end {
            orig += self
                .lm
                .logp(&s[ctx_start.max(j.saturating_sub(order - 1))..j], s[j]);
        }
        // 編集後: 文脈 + repl + S[b .. tail_end)
        buf.clear();
        buf.extend_from_slice(&s[ctx_start..a]);
        let first = buf.len();
        buf.extend_from_slice(repl);
        buf.extend_from_slice(&s[b..tail_end]);
        let mut new = 0.0;
        for j in first..buf.len() {
            new += self.lm.logp(&buf[j.saturating_sub(order - 1)..j], buf[j]);
        }
        new - orig
    }

    fn candidates(&self, toks: &[Token]) -> Vec<Cand> {
        let mut out = Vec::new();
        for (i, t) in toks.iter().enumerate() {
            let is_particle = t.pos == "助詞";
            // 削除・置換の対象は単純な助詞だけ。「によって」「に対し」等の複合助詞は消しても文が成立しやすく誤検出源になる
            let is_simple_particle = is_particle && PARTICLES.contains(&t.surface.as_str());
            // 1 文字の平仮名は、内容語の一部 (「のまネコ」の「ま」など) を消さないよう機能語に限る
            let is_hira1 = t.surface.chars().count() == 1
                && t.surface.chars().all(|c| ('ぁ'..='ん').contains(&c))
                && !matches!(
                    t.pos.as_str(),
                    "名詞" | "動詞" | "形容詞" | "接頭詞" | "副詞" | "連体詞"
                );
            let dup = i > 0 && toks[i - 1].surface == t.surface && t.pos != "名詞";
            if is_simple_particle || is_hira1 || dup {
                out.push(Cand {
                    a: i,
                    b: i + 1,
                    repl: vec![],
                    kind: EditKind::Delete,
                });
            }
            // 置換は単純な助詞どうしに限る (「によって」→「に」のような複合助詞の言い換えは正しい文でも高得点になる)
            if is_simple_particle {
                for p in PARTICLES {
                    if *p != t.surface {
                        out.push(Cand {
                            a: i,
                            b: i + 1,
                            repl: vec![p.to_string()],
                            kind: EditKind::Substitute,
                        });
                    }
                }
            }
            if matches!(t.pos.as_str(), "動詞" | "形容詞" | "助動詞")
                && !t.conj_type.is_empty()
                && let Some(forms) = self.inflections.get(&(t.base.clone(), t.conj_type.clone()))
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
                        .any(|x| !matches!(x.base.as_str(), "う" | "よう"))
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
                            repl: vec![f.clone()],
                            kind: EditKind::Inflection,
                        });
                    }
                }
            }

            if matches!(t.pos.as_str(), "名詞" | "動詞" | "形容詞" | "副詞")
                && !t.reading.is_empty()
                && let Some(alts) = self.readings.get(&t.reading)
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
                        && kanji_of(alt) != orig_kanji
                    {
                        out.push(Cand {
                            a: i,
                            b: i + 1,
                            repl: vec![alt.clone()],
                            kind: EditKind::Homophone,
                        });
                    }
                }
            }
            if self.cfg.enable_insert
                && i > 0
                && toks[i - 1].pos == "名詞"
                && t.pos != "助詞"
                && t.pos != "記号"
            {
                for p in INSERT_PARTICLES {
                    out.push(Cand {
                        a: i,
                        b: i,
                        repl: vec![p.to_string()],
                        kind: EditKind::Insert,
                    });
                }
            }
        }
        out
    }
}

/// 分かち書き時に集めた活用表 (TSV: 原形 \t 活用型 \t 表層形) を読む。
pub fn load_inflections(
    path: &std::path::Path,
) -> anyhow::Result<FxHashMap<(String, String), Vec<String>>> {
    let mut m: FxHashMap<(String, String), Vec<String>> = FxHashMap::default();
    for line in std::fs::read_to_string(path)?.lines() {
        let mut it = line.split('\t');
        if let (Some(b), Some(t), Some(s)) = (it.next(), it.next(), it.next()) {
            m.entry((b.to_string(), t.to_string()))
                .or_default()
                .push(s.to_string());
        }
    }
    Ok(m)
}

/// 同音異字表 (TSV: 読み \t 表層形 \t 出現数) を読む。
pub fn load_readings(
    path: &std::path::Path,
) -> anyhow::Result<FxHashMap<String, Vec<(String, u32)>>> {
    let mut m: FxHashMap<String, Vec<(String, u32)>> = FxHashMap::default();
    for line in std::fs::read_to_string(path)?.lines() {
        let mut it = line.split('\t');
        if let (Some(r), Some(s), Some(c)) = (it.next(), it.next(), it.next()) {
            m.entry(r.to_string())
                .or_default()
                .push((s.to_string(), c.parse().unwrap_or(0)));
        }
    }
    for v in m.values_mut() {
        v.sort_by_key(|e| std::cmp::Reverse(e.1));
    }
    Ok(m)
}
