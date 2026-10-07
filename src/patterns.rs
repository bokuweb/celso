//! 実際の誤字から集めた書き換えパターン (「をを → を」「れいる → れている」「いづれも → いずれも」) の照合。
//!
//! enno.jp のような「誤りのパターンを蓄積して照合する」方式を、JWTD (日本語 Wikipedia 入力誤り
//! データセット、CC BY-SA 3.0) の train から自前で作った表で行う (scripts/patterns/)。
//! n-gram の候補生成では拾えない文字単位の誤字 (重複・脱落・入れ替え・かな違い) を、
//! 文書を 1 回なめるだけで見つける (n-gram の検査に比べて無視できる時間)。
//!
//! 照合は「先頭の 2 文字 → その 2 文字で始まるパターン (長い順)」の索引で行う。1 万件余りの
//! パターンを Aho-Corasick のオートマトンにすると常駐が 8MB ほど増えたが、この索引なら
//! パターン本体 (約 0.5MB) だけで済む。先頭 1 文字の索引だと「に」「の」で始まるパターンが
//! 数百件あって遅かったので 2 文字にしている (パターンは必ず 2 文字以上)。

use anyhow::Result;
use rustc_hash::FxHashMap;

use crate::checker::{EditKind, Finding};

pub struct Patterns {
    /// 先頭の 2 文字 → その 2 文字で始まるパターンの番号 (誤り側の長い順)
    index: FxHashMap<(char, char), Vec<u32>>,
    /// パターン番号 → (誤り, 正しい, スコア)
    rules: Vec<(String, String, f32)>,
}

impl Patterns {
    /// TSV (誤り \t 正しい \t 支持数 …) を読む。スコアは log10(支持数)。
    pub fn from_tsv(text: &str) -> Result<Self> {
        let mut rules = Vec::new();
        for line in text.lines() {
            let mut it = line.split('\t');
            let (Some(w), Some(r)) = (it.next(), it.next()) else {
                continue;
            };
            let support: f32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(1.0);
            if w.chars().count() < 2 || w == r {
                continue;
            }
            rules.push((w.to_string(), r.to_string(), support.max(1.0).log10()));
        }
        let mut index: FxHashMap<(char, char), Vec<u32>> = FxHashMap::default();
        for (i, (w, _, _)) in rules.iter().enumerate() {
            let mut cs = w.chars();
            if let (Some(a), Some(b)) = (cs.next(), cs.next()) {
                index.entry((a, b)).or_default().push(i as u32);
            }
        }
        for v in index.values_mut() {
            v.sort_by_key(|&i| std::cmp::Reverse(rules[i as usize].0.len()));
            v.shrink_to_fit();
        }
        Ok(Self { index, rules })
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
    #[must_use]
    pub fn find(&self, sent: &str) -> Vec<Finding> {
        let mut out = Vec::new();
        // (バイト位置, 文字位置) を進めながら、各位置で一番長いパターンを探す (重なりは取らない)
        let mut byte = 0usize;
        let mut ch = 0usize;
        while byte < sent.len() {
            let rest = &sent[byte..];
            let mut cs = rest.chars();
            let c = cs.next().unwrap_or('\0');
            let c2 = cs.next().unwrap_or('\0');
            let hit = self.index.get(&(c, c2)).and_then(|ids| {
                ids.iter()
                    .map(|&i| &self.rules[i as usize])
                    .find(|(w, _, _)| rest.starts_with(w.as_str()))
            });
            let Some((w, r, score)) = hit else {
                byte += c.len_utf8();
                ch += 1;
                continue;
            };
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
            byte += w.len();
            ch += wc.len();
        }
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
}
