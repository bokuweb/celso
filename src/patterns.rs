//! 実際の誤字から集めた書き換えパターン (「をを → を」「れいる → れている」「いづれも → いずれも」) の照合。
//!
//! enno.jp のような「誤りのパターンを蓄積して照合する」方式を、JWTD (日本語 Wikipedia 入力誤り
//! データセット、CC BY-SA 3.0) の train から自前で作った表で行う (scripts/patterns/)。
//! n-gram の候補生成では拾えない文字単位の誤字 (重複・脱落・入れ替え・かな違い) を、
//! 文書を 1 回なめるだけで見つける (n-gram の検査に比べて無視できる時間)。
//!
//! 照合は「先頭の 2 文字 → その 2 文字で始まるパターン (長い順)」の索引で行う。1 万件余りの
//! パターンを Aho-Corasick のオートマトンにすると常駐が 8MB ほど増えたが、この索引なら
//! パターン本体だけで済む。先頭 1 文字の索引だと「に」「の」で始まるパターンが
//! 数百件あって遅かったので 2 文字にしている (パターンは必ず 2 文字以上)。
//!
//! パターン (3.5 万件) を 1 件ずつ String で持つと確保の分だけで数 MB になるので、誤り・正しい側を
//! 1 本の文字列につなげ、各パターンは位置と長さだけを持つ。索引も先頭 2 文字ごとの範囲だけにする。

use anyhow::Result;
use rustc_hash::FxHashMap;

use crate::checker::{EditKind, Finding};

/// パターン 1 件 (`text` の中の位置と長さ)。
struct Rule {
    /// 誤り側の先頭のバイト位置 (正しい側はその直後に続く)
    off: u32,
    /// 誤り側・正しい側のバイト長
    wlen: u16,
    rlen: u16,
    /// log10(支持数)
    score: f32,
}

pub struct Patterns {
    /// 先頭の 2 文字 → その 2 文字で始まるパターンの範囲 (`rules` の添字。中は誤り側の長い順)
    index: FxHashMap<(char, char), (u32, u32)>,
    /// 先頭 2 文字・長さの順に並べたパターン
    rules: Vec<Rule>,
    /// 全パターンの誤り側と正しい側をつなげた文字列
    text: String,
}

impl Patterns {
    /// TSV (誤り \t 正しい \t 支持数 …) を読む。スコアは log10(支持数)。
    pub fn from_tsv(src: &str) -> Result<Self> {
        let mut rows: Vec<(&str, &str, f32)> = Vec::new();
        for line in src.lines() {
            let mut it = line.split('\t');
            let (Some(w), Some(r)) = (it.next(), it.next()) else {
                continue;
            };
            let support: f32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(1.0);
            if w.chars().count() < 2
                || w == r
                || w.len() > usize::from(u16::MAX)
                || r.len() > usize::from(u16::MAX)
            {
                continue;
            }
            rows.push((w, r, support.max(1.0).log10()));
        }
        let first2 = |w: &str| {
            let mut cs = w.chars();
            (cs.next().unwrap_or('\0'), cs.next().unwrap_or('\0'))
        };
        // 先頭 2 文字ごとにまとめ、その中は誤り側の長い順 (同じ長さなら元の並び)
        rows.sort_by(|a, b| {
            first2(a.0)
                .cmp(&first2(b.0))
                .then(b.0.len().cmp(&a.0.len()))
        });
        let mut text = String::with_capacity(rows.iter().map(|r| r.0.len() + r.1.len()).sum());
        let mut rules = Vec::with_capacity(rows.len());
        let mut index: FxHashMap<(char, char), (u32, u32)> = FxHashMap::default();
        for (i, (w, r, score)) in rows.iter().enumerate() {
            let off = text.len() as u32;
            text.push_str(w);
            text.push_str(r);
            rules.push(Rule {
                off,
                wlen: w.len() as u16,
                rlen: r.len() as u16,
                score: *score,
            });
            let e = index.entry(first2(w)).or_insert((i as u32, i as u32));
            e.1 = i as u32 + 1;
        }
        index.shrink_to_fit();
        Ok(Self { index, rules, text })
    }

    fn wrong(&self, r: &Rule) -> &str {
        &self.text[r.off as usize..r.off as usize + usize::from(r.wlen)]
    }

    fn right(&self, r: &Rule) -> &str {
        let start = r.off as usize + usize::from(r.wlen);
        &self.text[start..start + usize::from(r.rlen)]
    }

    pub fn load(path: &std::path::Path) -> Result<Self> {
        Self::from_tsv(&std::fs::read_to_string(path)?)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// 文の中のパターンを探す。直す範囲は誤りと正しい側の共通の前後を除いた最小の範囲にする
    /// (「れいる → れている」は「れ|い」の間に「て」を補う指摘になる)。
    ///
    /// 重なるパターンもすべて返す (採否は呼び出し側が文脈で決める)。一番長いものだけを採ると、
    /// 支持数の少ない「しなない → しかない」が「なない → ない」の位置を奪い、それが文脈で落ちると何も残らない。
    #[must_use]
    pub fn find(&self, sent: &str) -> Vec<Finding> {
        let mut out = Vec::new();
        let mut ch = 0usize;
        for (byte, c) in sent.char_indices() {
            let rest = &sent[byte..];
            let c2 = rest[c.len_utf8()..].chars().next().unwrap_or('\0');
            let Some(&(lo, hi)) = self.index.get(&(c, c2)) else {
                ch += 1;
                continue;
            };
            for rule in &self.rules[lo as usize..hi as usize] {
                let (w, r, score) = (self.wrong(rule), self.right(rule), &rule.score);
                if !rest.starts_with(w) {
                    continue;
                }
                let wc: Vec<char> = w.chars().collect();
                let rc: Vec<char> = r.chars().collect();
                let mut pre = 0;
                while pre < wc.len() && pre < rc.len() && wc[pre] == rc[pre] {
                    pre += 1;
                }
                let mut suf = 0;
                while suf < wc.len() - pre
                    && suf < rc.len() - pre
                    && wc[wc.len() - 1 - suf] == rc[rc.len() - 1 - suf]
                {
                    suf += 1;
                }
                out.push(Finding {
                    start: ch + pre,
                    end: ch + wc.len() - suf,
                    original: wc[pre..wc.len() - suf].iter().collect(),
                    replacement: rc[pre..rc.len() - suf].iter().collect(),
                    kind: EditKind::Pattern,
                    delta: *score,
                    alternatives: Vec::new(),
                });
            }
            ch += 1;
        }
        // 文脈の幅が違う同じ直し (「をを」と「資料をを」) は 1 つにまとめる (支持数の多い方を残す)
        out.sort_by(|a, b| {
            (a.start, a.end, &a.replacement)
                .cmp(&(b.start, b.end, &b.replacement))
                .then(b.delta.total_cmp(&a.delta))
        });
        out.dedup_by(|a, b| a.start == b.start && a.end == b.end && a.replacement == b.replacement);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_minimal_edit_of_a_pattern() {
        let p = Patterns::from_tsv("れいる\tれている\t1223\nをを\tを\t6837\n").unwrap();
        let f = p.find("記録されいる資料をを読む。");
        assert_eq!(f.len(), 2);
        // 「されいる」の「れ」と「い」の間に「て」を補う
        assert_eq!((f[0].start, f[0].end), (4, 4));
        assert_eq!(f[0].replacement, "て");
        // 「をを」は共通の先頭を除いた 2 つ目の「を」を消す
        assert_eq!(
            (
                f[1].start,
                f[1].end,
                f[1].original.as_str(),
                f[1].replacement.as_str()
            ),
            (9, 10, "を", "")
        );
    }

    #[test]
    fn reports_overlapping_patterns_for_the_caller_to_choose() {
        let p = Patterns::from_tsv("しなない\tしかない\t2\nなない\tない\t14\n").unwrap();
        let f = p.find("適用しなない。");
        let edits: Vec<(usize, usize, &str)> = f
            .iter()
            .map(|x| (x.start, x.end, x.replacement.as_str()))
            .collect();
        // 長い方 (「な」→「か」) だけでなく、重なる短い方 (「な」を消す) も返す
        assert!(edits.contains(&(3, 4, "か")), "{edits:?}");
        assert!(
            edits.contains(&(3, 4, "")) || edits.contains(&(4, 5, "")),
            "{edits:?}"
        );
    }

    #[test]
    fn merges_the_same_edit_found_by_patterns_of_different_widths() {
        let p = Patterns::from_tsv("をを\tを\t6837\n資料をを\t資料を\t3\n").unwrap();
        let f = p.find("資料をを読む。");
        assert_eq!(f.len(), 1);
    }
}
