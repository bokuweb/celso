//! 修正候補を作って言語モデルで比べるチェッカー。
//!
//! 単語としては正しいが並びがおかしい誤り (余計な助詞・助詞の取り違え・活用の誤り) を狙う。
//! 各位置で「よくある誤りを元に戻す編集」を候補として作り、編集の前後で
//! 影響を受ける範囲 (編集箇所 + 後続 order-1 語) の対数確率を比べる。
//! 改善幅 Δ (log10) が種類ごとの閾値を超えたものを指摘する。

use rustc_hash::FxHashMap;

use crate::lm::{BOS, EOS, Model, UNK};
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
    /// 改善幅 (log10)。
    pub delta: f32,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub thresholds: FxHashMap<EditKind, f32>,
    pub enable_insert: bool,
    /// 元の文の編集箇所まわりに、この次数の n-gram として「未出現」の並びがあるときだけ指摘する。
    /// 正しい文どうしの言い換え (「期間が」⇔「期間の」) を拾わないための足切り。0 で無効。
    pub novelty_order: usize,
    /// 指摘箇所の前後を含む文字列が文書内にこの回数以上あれば指摘しない (0 で無効)。
    pub doc_repeat_limit: usize,
}

impl Default for Config {
    fn default() -> Self {
        let mut t = FxHashMap::default();
        t.insert(EditKind::Delete, 2.5);
        t.insert(EditKind::Substitute, 2.0);
        t.insert(EditKind::Inflection, 2.0);
        t.insert(EditKind::Insert, 2.5);
        t.insert(EditKind::Homophone, 3.0);
        // 文字単位の編集は遅く誤検出も多いので既定では無効 (README 参照)
        t.insert(EditKind::Char, f32::INFINITY);
        Self {
            thresholds: t,
            enable_insert: true,
            novelty_order: 3,
            doc_repeat_limit: 2,
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
    pub lm: Model,
    pub cfg: Config,
    /// (原形, 活用型) → その語の活用した表層形の一覧
    inflections: FxHashMap<(String, String), Vec<String>>,
    /// 読み → (表層形, 出現数) 出現数の多い順
    readings: FxHashMap<String, Vec<(String, u32)>>,
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
        lm: Model,
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
        }
    }

    /// テキスト全体を検査する。文 (句点・改行) ごとに独立に処理する。
    pub fn check(&self, text: &str) -> Vec<Finding> {
        let normalized = norm(text);
        let chars: Vec<char> = normalized.chars().collect();
        let mut out = Vec::new();
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
                let sent: String = chars[s..e].iter().collect();
                for mut f in self.check_sentence(&sent) {
                    f.start += s;
                    f.end += s;
                    out.push(f);
                }
            }
            s = i + 1;
        }
        // original は元テキスト (正規化前) から切り出し直す
        let orig_chars: Vec<char> = text.chars().collect();
        for f in &mut out {
            f.original = orig_chars[f.start..f.end].iter().collect();
        }
        out
    }

    /// 文書全体を検査する。行単位で並列に処理し、最後に文書内の繰り返しで誤検出を抑える。
    pub fn check_document(&self, text: &str) -> Vec<Finding> {
        use rayon::prelude::*;
        let mut lines = Vec::new();
        let mut off = 0;
        for l in text.split('\n') {
            lines.push((off, l));
            off += l.chars().count() + 1;
        }
        let found: Vec<Finding> = lines
            .par_iter()
            .flat_map_iter(|(off, l)| {
                self.check(l).into_iter().map(move |mut f| {
                    f.start += off;
                    f.end += off;
                    f
                })
            })
            .collect();
        let normalized = norm(text);
        let chars: Vec<char> = normalized.chars().collect();
        found
            .into_iter()
            .filter(|f| self.doc_repeats(&normalized, &chars, f, 0) < self.cfg.doc_repeat_limit)
            .collect()
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
        doc.matches(ctx.as_str()).count() + extra
    }

    /// 正規化済みの 1 文を検査する。
    pub fn check_sentence(&self, sent: &str) -> Vec<Finding> {
        let toks = self.tok.tokenize(sent);
        if toks.is_empty() {
            return Vec::new();
        }
        // S = <s> w1 .. wn </s>
        let ids = self.ids_of(&toks);

        // (Δ, Finding) を集めてから重なりを解消する
        let mut scored: Vec<Finding> = Vec::new();
        let mut buf: Vec<u32> = Vec::with_capacity(32);
        for c in self.candidates(&toks) {
            let repl_ids: Vec<u32> = c.repl.iter().map(|w| self.lm.word_id(w)).collect();
            if repl_ids.contains(&UNK) {
                continue;
            }
            let th = self
                .cfg
                .thresholds
                .get(&c.kind)
                .copied()
                .unwrap_or(f32::INFINITY);
            if !self.is_novel(&ids, c.a + 1, c.b + 1) {
                continue;
            }
            let delta = self.delta(&ids, c.a + 1, c.b + 1, &repl_ids, &mut buf);
            if delta >= th {
                let start = toks
                    .get(c.a)
                    .map(|t| t.start)
                    .unwrap_or_else(|| toks.last().unwrap().end);
                let end = if c.b > c.a { toks[c.b - 1].end } else { start };
                scored.push(Finding {
                    start,
                    end,
                    original: toks[c.a..c.b].iter().map(|t| t.surface.as_str()).collect(),
                    replacement: c.repl.concat(),
                    kind: c.kind,
                    delta,
                });
            }
        }
        if self
            .cfg
            .thresholds
            .get(&EditKind::Char)
            .is_some_and(|t| t.is_finite())
        {
            scored.extend(self.char_edits(sent, &toks, &ids));
        }

        // 重なる候補は Δ が最大のものだけ残す (挿入は幅 1 とみなす)。
        // 隣接 (1 文字以内) も重なり扱いにする: 「飲食は店」で「は」を消す候補と「店」を消す候補は
        // 同じ 1 つの誤りに対する別解なので、両方出すと誤検出になる。
        scored.sort_by(|x, y| y.delta.total_cmp(&x.delta));
        let mut taken: Vec<(usize, usize)> = Vec::new();
        let mut out = Vec::new();
        for f in scored {
            let (a, b) = (f.start.saturating_sub(1), f.end.max(f.start + 1) + 1);
            if taken.iter().any(|&(x, y)| a < y && x < b) {
                continue;
            }
            taken.push((a, b));
            out.push(f);
        }
        out.sort_by_key(|f| f.start);
        out
    }

    fn ids_of(&self, toks: &[Token]) -> Vec<u32> {
        let mut ids: Vec<u32> = Vec::with_capacity(toks.len() + 2);
        ids.push(BOS);
        ids.extend(toks.iter().map(|t| self.lm.word_id(t.key())));
        ids.push(EOS);
        ids
    }

    fn sentence_logp(&self, ids: &[u32]) -> f32 {
        let order = self.lm.order;
        (1..ids.len())
            .map(|j| self.lm.logp(&ids[j.saturating_sub(order - 1)..j], ids[j]))
            .sum()
    }

    /// 文字単位の編集 (1 文字削除・隣接入れ替え・かな 1 文字補完) を、
    /// 「コーパスに出てこない並び」がある語の周辺だけで試す。編集後は分かち書きし直して文全体で比べる。
    fn char_edits(&self, sent: &str, toks: &[Token], ids: &[u32]) -> Vec<Finding> {
        let th = self.cfg.thresholds[&EditKind::Char];
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
                || self.lm.word_id(t.key()) == UNK
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
        let order = self.lm.order;
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
            let is_hira1 = t.surface.chars().count() == 1
                && t.surface.chars().all(|c| ('ぁ'..='ん').contains(&c));
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
                && t.conj_type != "*"
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
