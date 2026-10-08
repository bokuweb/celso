//! 文字単位の言語モデルで、語の中の 1 字の誤り (「性意式 → 性意識」「骨董品 300 店 → 点」「離脱抜した → 離脱した」) を拾う。
//!
//! 単語 n-gram は語の中を見ないので、熟語の 1 字の変換ミスや、語の中の脱字・余分な字に届かない。
//! そこで文字の 5-gram を持ち、
//!
//! 1. 文字ごとの対数確率を出し、低い (コーパスで見かけない並びの) 位置だけを怪しい位置とする
//! 2. 怪しい位置で、1 字の削除・かなの挿入・置換 (漢字は同じ読みの漢字、かなは別のかな)・隣との入れ替えを試す
//! 3. 前後の文字を含めた対数確率が大きく上がる直しだけを指摘する
//!
//! 怪しい位置にだけ候補を作るので、文書全体の文字ごとに全候補を試すより桁違いに少ない計算で済む。

use std::path::Path;

use anyhow::Result;
use rustc_hash::FxHashMap;

use crate::checker::{Domain, EditKind, Finding, delta_pre, position_logps};
use crate::lm::{BOS, EOS, LanguageModel, UNK};
use crate::rerank::{Features, Reranker};

/// 補う・置き換える候補にするひらがな。
const KANA: &str = "あいうえおかきくけこがぎぐげごさしすせそざじずぜぞたちつてとだぢづでどなにぬねのはひふへほばびぶべぼぱぴぷぺぽまみむめもやゆよらりるれろわをんっゃゅょー";

/// 助詞どうしの置き換えを表す特徴量 (`partpair`) に使う助詞。
const PARTICLES: &str = "のにがをはでともへやか";

/// 補う・置き換える候補に残すひらがなの、実際の誤字での出現数の下限 (補う・置き換えの合計)。
const KANA_MIN_PRIOR: u32 = 200;

/// 文字モデルが見つけた直しと、採否の判定に使う値。
#[derive(Debug, Clone)]
pub struct CharCandidate {
    pub finding: Finding,
    /// 直したときの文字モデルの対数確率の改善幅 (log10)
    pub delta: f32,
    /// 直す位置の文字 (とその次の文字) の対数確率の小さいほう
    pub logp: f32,
    /// 同じ位置の 2 番目によい直しとの改善幅の差 (他に候補がなければ改善幅そのもの)
    pub margin: f32,
    /// 文の中の怪しい位置の数 (文全体が珍しい並びばかりのときは個々の候補を信用しにくい)
    pub suspicious_count: usize,
}

pub struct CharChecker {
    lm: Box<dyn LanguageModel>,
    /// 漢字 → 置き換える候補の漢字と文字モデルでの ID (語彙外の字は除く)
    homo: FxHashMap<char, Vec<(char, u32)>>,
    /// この対数確率 (log10) より低い文字の位置だけを怪しい位置として調べる
    pub suspicious: f32,
    /// 直したときの対数確率の改善幅 (log10) がこれ以上のものだけを候補にする
    /// (採否は呼び出し側が判定器か、直し方の種類ごとの下限で決める)
    pub min_delta: f32,
    /// 採否の判定器。無ければ直し方の種類ごとの下限で決める
    pub rank: Option<CharRanker>,
    /// 1 つの位置から出す直しの数 (改善幅の大きい順)
    pub per_position: usize,
    /// 置換・挿入で、後ろの文脈まで含めて採点する候補の数 (直後の 1 字の確率の上位)
    pub beam: usize,
    /// 補う・置き換える候補にするひらがな (と文字モデルでの ID)
    kana: Vec<(char, u32)>,
}

/// 文字単位の直しの採否を決める判定器 (ロジスティック回帰)。
///
/// 直し方の種類ごとの下限だけでは、「語の中のカタカナの余分な字 (ッ・ュ) は誤りが多い」
/// 「『も』を消す直しは正しい文に多い」のような字ごとの癖を表せない。そこで文字モデル・単語モデルの
/// 改善幅に、直す字とその前後の字の種類、実際の誤字での同じ直しの出現数 (誤りの起きやすさ) などを
/// 足した特徴量で、誤りである対数オッズを出す (JWTD の学習用の文と判例要旨の正しい文で学習)。
///
/// TSV は [`Reranker`] と同じ重みの行と、`#prior \t 誤り \t 正しい \t 出現数` の行からなる。
pub struct CharRanker {
    rank: Reranker,
    prior: FxHashMap<(Box<str>, Box<str>), u32>,
}

/// 直し方の種類 (判定器の特徴量の名前に使う)。
fn edit_type(original: &str, replacement: &str) -> &'static str {
    let o: Vec<char> = original.chars().collect();
    let r: Vec<char> = replacement.chars().collect();
    if o.is_empty() {
        "ins"
    } else if r.is_empty() {
        "del"
    } else if o.len() == 2 && r.len() == 2 && o[0] == r[1] && o[1] == r[0] {
        "swap"
    } else if is_kanji(o[0]) {
        "subk"
    } else {
        "subn"
    }
}

/// 字の種類 (漢字 k・ひらがな h・カタカナ K・その他 o・無し none)。
fn char_class(c: Option<char>) -> &'static str {
    match c {
        None => "none",
        Some(c) if is_kanji(c) => "k",
        Some(c) if is_hiragana(c) => "h",
        Some(c) if ('ァ'..='ヺ').contains(&c) => "K",
        Some(_) => "o",
    }
}

/// 単語モデル側で求めた、直した文についての値。
pub struct WordSignals {
    /// 単語モデルでの文の対数確率の改善幅
    pub delta: f32,
    /// 直して減った未知語の数
    pub unknown_removed: i32,
    /// 直して増えた語の数
    pub token_count_change: i32,
    /// 文法モデル (品詞と機能語の 5-gram) での文の対数確率の改善幅
    pub aux_delta: f32,
    /// 文全体の語との共起の改善 (直して入る語 − 直して消える語)
    pub cooc_delta: f32,
}

impl CharRanker {
    pub fn from_tsv(text: &str) -> Result<Self> {
        let mut prior = FxHashMap::default();
        let mut rest = String::new();
        for line in text.lines() {
            if let Some(x) = line.strip_prefix("#prior\t") {
                let mut cols = x.split('\t');
                let (Some(o), Some(r), Some(n)) = (cols.next(), cols.next(), cols.next()) else {
                    continue;
                };
                prior.insert((o.into(), r.into()), n.trim().parse()?);
            } else {
                rest.push_str(line);
                rest.push('\n');
            }
        }
        Ok(Self {
            rank: Reranker::from_tsv(&rest)?,
            prior,
        })
    }

    pub fn load(path: &Path) -> Result<Self> {
        Self::from_tsv(&std::fs::read_to_string(path)?)
    }

    /// 実際の誤字で、補う・置き換え後の字として `min` 回以上出てくるひらがな (KANA の順)。
    #[must_use]
    pub fn frequent_kana(&self, min: u32) -> Vec<char> {
        let mut n: FxHashMap<char, u32> = FxHashMap::default();
        for ((o, r), &cnt) in &self.prior {
            let mut rc = r.chars();
            if let (Some(c), None) = (rc.next(), rc.next())
                && o.chars().count() <= 1
                && is_hiragana(c)
            {
                *n.entry(c).or_default() += cnt;
            }
        }
        KANA.chars()
            .filter(|c| n.get(c).copied().unwrap_or(0) >= min)
            .collect()
    }

    /// 採用の閾値 (対数オッズ)。
    #[must_use]
    pub fn tau(&self, d: Domain) -> f32 {
        self.rank.tau(d)
    }

    /// 誤りである対数オッズ。`fixed` は直した文 (文字単位)。
    #[must_use]
    pub fn score(&self, c: &CharCandidate, fixed: &[char], w: &WordSignals) -> f32 {
        self.rank.score(&self.features(c, fixed, w))
    }

    /// 判定器の特徴量 (学習に使ったスクリプトと同じ名前・同じ値にする)。
    #[must_use]
    #[allow(clippy::many_single_char_names)] // 学習スクリプト (scripts/charrank/feat.py) と同じ短い名前で突き合わせやすくする
    pub fn features(&self, c: &CharCandidate, fixed: &[char], w: &WordSignals) -> Features {
        let f = &c.finding;
        let (o, r) = (f.original.as_str(), f.replacement.as_str());
        let t = edit_type(o, r);
        let prior = self
            .prior
            .get(&(Box::<str>::from(o), Box::<str>::from(r)))
            .copied()
            .unwrap_or(0);
        let prior_log = (prior as f32).ln_1p();
        let (cd, wd) = (c.delta, w.delta);
        let ns = c.suspicious_count as f32;
        let mut out: Features = Vec::with_capacity(48);
        let mut add = |k: String, v: f32| out.push((k, v));
        for pre in ["", t] {
            let p = if pre.is_empty() {
                String::new()
            } else {
                format!("{pre}:")
            };
            add(format!("{p}bias"), 1.0);
            add(format!("{p}cd"), cd);
            add(format!("{p}wd"), wd);
            add(format!("{p}min"), cd.min(wd));
            add(format!("{p}lp"), c.logp);
            add(format!("{p}mg"), c.margin.min(10.0));
            add(format!("{p}prior"), prior_log);
            add(format!("{p}prior0"), if prior == 0 { 1.0 } else { 0.0 });
            add(format!("{p}unk"), w.unknown_removed as f32);
            add(format!("{p}nt"), w.token_count_change as f32);
            add(format!("{p}ns"), ns.ln_1p());
            add(format!("{p}ad"), w.aux_delta);
            add(format!("{p}co"), w.cooc_delta);
        }
        let left = f.start.checked_sub(1).and_then(|i| fixed.get(i)).copied();
        let right = fixed.get(f.start + r.chars().count()).copied();
        add(format!("{t}:oc={}", char_class(o.chars().next())), 1.0);
        add(format!("{t}:rc={}", char_class(r.chars().next())), 1.0);
        add(format!("{t}:lc={}", char_class(left)), 1.0);
        add(format!("{t}:Rc={}", char_class(right)), 1.0);
        add(
            format!("{t}:lc={}:rc={}", char_class(left), char_class(right)),
            1.0,
        );
        if o.chars().count() <= 1
            && r.chars().count() <= 1
            && !o.chars().next().is_some_and(is_kanji)
        {
            add(format!("{t}:o={o}"), 1.0);
            add(format!("{t}:r={r}"), 1.0);
        }
        let is_particle = |s: &str| s.chars().count() == 1 && PARTICLES.contains(s);
        if t == "subn" && is_particle(o) && is_particle(r) {
            add("partpair".to_string(), 1.0);
        }
        add(format!("{t}:cdwd"), cd * wd / 10.0);
        // 値の区間 (線形だけでは表せない形を補う)
        let bin = |v: f32, step: f32, lo: f32, hi: f32| (v.clamp(lo, hi) / step).floor() as i64;
        add(format!("{t}:cdb={}", bin(cd, 1.0, 0.0, 12.0)), 1.0);
        add(format!("{t}:wdb={}", bin(wd, 1.0, -3.0, 10.0)), 1.0);
        add(
            format!(
                "{t}:cdb={}:wdb={}",
                bin(cd, 2.0, 0.0, 12.0),
                bin(wd, 2.0, -2.0, 10.0)
            ),
            1.0,
        );
        add(format!("{t}:lpb={}", bin(c.logp, 1.0, -8.0, -3.0)), 1.0);
        add(format!("{t}:mgb={}", bin(c.margin, 1.0, 0.0, 8.0)), 1.0);
        add(format!("{t}:pb={}", bin(prior_log, 1.0, 0.0, 8.0)), 1.0);
        add(format!("{t}:nsb={}", bin(ns, 2.0, 0.0, 12.0)), 1.0);
        add(format!("{t}:adb={}", bin(w.aux_delta, 1.0, -4.0, 6.0)), 1.0);
        add(
            format!("{t}:cob={}", bin(w.cooc_delta, 0.5, -2.0, 2.0)),
            1.0,
        );
        out
    }
}

fn is_kanji(c: char) -> bool {
    ('\u{4E00}'..='\u{9FFF}').contains(&c) || c == '々'
}

fn is_hiragana(c: char) -> bool {
    ('ぁ'..='ゖ').contains(&c) || c == 'ー'
}

impl CharChecker {
    #[must_use]
    pub fn new(lm: Box<dyn LanguageModel>, homo: FxHashMap<char, Vec<char>>) -> Self {
        let kana = Self::char_ids(lm.as_ref(), KANA.chars());
        let homo = homo
            .into_iter()
            .map(|(k, alts)| (k, Self::char_ids(lm.as_ref(), alts.into_iter())))
            .collect();
        Self {
            kana,
            lm,
            homo,
            suspicious: -3.0,
            min_delta: 1.0,
            rank: None,
            per_position: 3,
            beam: 6,
        }
    }

    /// 同じ読みの漢字の表 (TSV: 漢字 \t 同じ読みの漢字を並べた文字列) を読む。
    #[must_use]
    pub fn parse_homo(text: &str) -> FxHashMap<char, Vec<char>> {
        let mut m = FxHashMap::default();
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let Some((k, alts)) = line.split_once('\t') else {
                continue;
            };
            let Some(k) = k.chars().next() else { continue };
            m.insert(k, alts.chars().collect());
        }
        m
    }

    pub fn load(model: &Path, homo: &Path) -> Result<Self> {
        let lm = crate::lm::load_any(model)?;
        let homo = Self::parse_homo(&std::fs::read_to_string(homo)?);
        Ok(Self::new(lm, homo))
    }

    fn char_ids(lm: &dyn LanguageModel, chars: impl Iterator<Item = char>) -> Vec<(char, u32)> {
        let mut b = [0u8; 4];
        chars
            .map(|c| (c, lm.word_id(c.encode_utf8(&mut b))))
            .filter(|&(_, x)| x != UNK)
            .collect()
    }

    /// 判定器を付ける。補う・置き換えるひらがなは、実際の誤字でよく直されている字に絞る
    /// (候補が減って速くなり、出現数の少ない字の候補は判定器がほとんど採らない)。
    #[must_use]
    pub fn with_rank(mut self, rank: CharRanker) -> Self {
        let frequent = rank.frequent_kana(KANA_MIN_PRIOR);
        if !frequent.is_empty() {
            self.kana = Self::char_ids(self.lm.as_ref(), frequent.into_iter());
        }
        self.rank = Some(rank);
        self
    }

    fn id(&self, c: char) -> u32 {
        let mut b = [0u8; 4];
        self.lm.word_id(c.encode_utf8(&mut b))
    }

    /// 文の中の、語の中の 1 字の誤りらしい箇所を返す (オフセットは文字単位)。
    #[must_use]
    pub fn find(&self, sent: &str) -> Vec<Finding> {
        self.candidates(sent)
            .into_iter()
            .map(|c| c.finding)
            .collect()
    }

    /// [`Self::find`] と同じ直しを、採否の判定に使う値つきで返す。
    #[must_use]
    #[allow(clippy::too_many_lines)] // 怪しい位置 → 直し方ごとの候補 → 位置ごとの上位、の流れを 1 か所で追えるようにしている
    pub fn candidates(&self, sent: &str) -> Vec<CharCandidate> {
        let chars: Vec<char> = sent.chars().collect();
        // 空白は分かち書きと同じく飛ばす (文字の位置は元の文のまま持つ)
        let pos: Vec<usize> = (0..chars.len())
            .filter(|&i| !chars[i].is_whitespace())
            .collect();
        if pos.len() < 3 {
            return Vec::new();
        }
        let mut ids: Vec<u32> = Vec::with_capacity(pos.len() + 2);
        ids.push(BOS);
        ids.extend(pos.iter().map(|&i| self.id(chars[i])));
        ids.push(EOS);
        let lp = position_logps(self.lm.as_ref(), &ids);
        let order = self.lm.order();
        let mut buf = Vec::with_capacity(16);
        let kana = &self.kana;
        let suspicious_count = (1..ids.len()).filter(|&j| lp[j] < self.suspicious).count();
        let mut out: Vec<CharCandidate> = Vec::new();
        // j: ids の位置 (1..=pos.len())。文字 pos[j-1] に対応する
        for j in 1..ids.len() - 1 {
            // その文字か次の文字が見かけない並びなら調べる (挿入・削除は後ろの文字の確率に出る)
            if lp[j] >= self.suspicious && lp[j + 1] >= self.suspicious {
                continue;
            }
            if ids[j] == UNK || ids[j - 1] == UNK || ids.get(j + 1) == Some(&UNK) {
                continue;
            }
            let c = chars[pos[j - 1]];
            // この位置で試した直し (改善幅, a, b, 置き換える文字列)
            let mut tried: Vec<(f32, usize, usize, String)> = Vec::new();
            let mut consider = |d: f32, a: usize, b: usize, repl: String| {
                if d.is_finite() {
                    tried.push((d, a, b, repl));
                }
            };
            // 削除 (語の中の余分な字)
            let d = delta_pre(self.lm.as_ref(), &ids, &lp, j, j + 1, &[], &mut buf);
            consider(d, j, j + 1, String::new());
            // 置換・挿入は、まず直後の 1 字の確率 (前の文脈だけで決まる) で候補を beam 件に絞り、
            // 絞った候補だけ後ろの文脈まで含めて採点する (全候補を後ろまで採点すると、
            // 5-gram では 1 候補あたり 5 回の引き当てになり、文字モデルの検査が単語モデルの数倍かかった)
            // 絞り込みは 3-gram (前の 2 字) で見る (長い文脈は後退の引き当てが増えて遅い)
            let ctx_start = j.saturating_sub(order.min(3) - 1);
            let mut first: Vec<f32> = Vec::new();
            let mut top = |cands: &[(char, u32)]| -> Vec<(char, u32)> {
                self.lm.logp_many(
                    &ids[ctx_start..j],
                    &cands.iter().map(|x| x.1).collect::<Vec<u32>>(),
                    &mut first,
                );
                let mut v: Vec<(f32, char, u32)> = cands
                    .iter()
                    .zip(&first)
                    .filter(|((_, k), _)| *k != UNK)
                    .map(|(&(ch, k), &l)| (l, ch, k))
                    .collect();
                let n = self.beam.min(v.len());
                if n < v.len() {
                    v.select_nth_unstable_by(n, |x, y| y.0.total_cmp(&x.0));
                    v.truncate(n);
                }
                v.into_iter().map(|(_, ch, k)| (ch, k)).collect()
            };
            // 置換: 漢字は同じ読みの漢字、ひらがなは別のひらがな
            if is_kanji(c)
                && let Some(alts) = self.homo.get(&c)
            {
                for (a, k) in top(alts) {
                    let d = delta_pre(self.lm.as_ref(), &ids, &lp, j, j + 1, &[k], &mut buf);
                    consider(d, j, j + 1, a.to_string());
                }
            }
            let kana_top = top(kana);
            if is_hiragana(c) {
                for &(kc, k) in &kana_top {
                    if k == ids[j] {
                        continue;
                    }
                    let d = delta_pre(self.lm.as_ref(), &ids, &lp, j, j + 1, &[k], &mut buf);
                    consider(d, j, j + 1, kc.to_string());
                }
            }
            // 挿入 (この文字の前にかなを 1 字補う)
            for &(kc, k) in &kana_top {
                let d = delta_pre(self.lm.as_ref(), &ids, &lp, j, j, &[k], &mut buf);
                consider(d, j, j, kc.to_string());
            }
            // 隣との入れ替え
            if j + 1 < ids.len() - 1 && ids[j] != ids[j + 1] {
                let r = [ids[j + 1], ids[j]];
                let d = delta_pre(self.lm.as_ref(), &ids, &lp, j, j + 2, &r, &mut buf);
                let swapped: String = [chars[pos[j]], chars[pos[j - 1]]].iter().collect();
                consider(d, j, j + 2, swapped);
            }
            // 改善幅の大きい順に per_position 件まで (差は次によい直しとの改善幅の差)
            tried.sort_by(|x, y| y.0.total_cmp(&x.0));
            for (k, (d, a, b, replacement)) in tried.iter().enumerate().take(self.per_position) {
                let (d, a, b) = (*d, *a, *b);
                if d < self.min_delta {
                    break;
                }
                let start = pos[a - 1];
                let end = if b > a { pos[b - 2] + 1 } else { start };
                out.push(CharCandidate {
                    finding: Finding {
                        start,
                        end,
                        original: chars[start..end].iter().collect(),
                        replacement: replacement.clone(),
                        kind: EditKind::Pattern,
                        delta: d - self.min_delta,
                        alternatives: Vec::new(),
                    },
                    delta: d,
                    logp: lp[j].min(lp[j + 1]),
                    margin: tried.get(k + 1).map_or(d, |x| d - x.0),
                    suspicious_count,
                });
            }
        }
        // 同じ箇所を複数の位置から見つけたときは 1 つにする (最初のものを残す)
        out.dedup_by(|a, b| {
            let (a, b) = (&a.finding, &b.finding);
            a.start == b.start && a.end == b.end && a.replacement == b.replacement
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny(lines: &[&str], homo: &str) -> CharChecker {
        let dir = std::env::temp_dir().join(format!("celso-charcheck-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let corpus = dir.join("chars.txt");
        let mut text = String::new();
        for _ in 0..50 {
            for l in lines {
                let cs: Vec<String> = l.chars().map(String::from).collect();
                text.push_str(&cs.join(" "));
                text.push('\n');
            }
        }
        std::fs::write(&corpus, text).unwrap();
        let lm = crate::lm::build(
            &[&corpus],
            &crate::lm::BuildConfig {
                order: 5,
                min_word_count: 1,
                min_count: [1; crate::lm::MAX_ORDER],
                vocab: None,
                keep_words: None,
                keep_min_count: [1; crate::lm::MAX_ORDER],
            },
        )
        .unwrap();
        CharChecker::new(Box::new(lm), CharChecker::parse_homo(homo))
    }

    #[test]
    fn fixes_one_character_errors_inside_words() {
        let c = tiny(
            &[
                "男性の意識を改める必要がある。",
                "古い意識が残っている。",
                "式典は中止された。",
            ],
            "式\t識\n識\t式\n",
        );
        // 熟語の中の同じ読みの漢字の誤り
        let f = c.find("男性の意式を改める必要がある。");
        assert!(
            f.iter()
                .any(|x| x.original == "式" && x.replacement == "識"),
            "{:?}",
            f.iter()
                .map(|x| (&x.original, &x.replacement))
                .collect::<Vec<_>>()
        );
        // 正しい文には出さない
        assert!(c.find("男性の意識を改める必要がある。").is_empty());
    }

    #[test]
    fn ranker_scores_with_weights_and_prior_counts() {
        let rk = CharRanker::from_tsv(
            "# コメント\n#tau\t1,1,1\nbias\t-1\nsubk:cd\t0.5\nsubk:prior\t1\n#prior\t店\t点\t20\n",
        )
        .unwrap();
        let cand = |o: &str, r: &str| CharCandidate {
            finding: Finding {
                start: 3,
                end: 4,
                original: o.to_string(),
                replacement: r.to_string(),
                kind: EditKind::Pattern,
                delta: 0.0,
                alternatives: Vec::new(),
            },
            delta: 4.0,
            logp: -5.0,
            margin: 2.0,
            suspicious_count: 1,
        };
        let fixed: Vec<char> = "骨董品点が盗難".chars().collect();
        let w = WordSignals {
            delta: 2.0,
            unknown_removed: 0,
            token_count_change: 0,
            aux_delta: 0.0,
            cooc_delta: 0.0,
        };
        // 実際の誤字で 20 回あった直しは、同じ改善幅でも出現数の分だけ高くなる
        let seen = rk.score(&cand("店", "点"), &fixed, &w);
        let unseen = rk.score(&cand("店", "天"), &fixed, &w);
        assert!((seen - (-1.0 + 2.0 + 21f32.ln())).abs() < 1e-4, "{seen}");
        assert!((unseen - 1.0).abs() < 1e-4, "{unseen}");
        assert!((rk.tau(Domain::General) - 1.0).abs() < 1e-6);
    }
}
