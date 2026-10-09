//! カタカナ語の打ち間違い (「オブジェクト → オプジェクト」「キャラクター → キャラクー」) の検出。
//!
//! 言語モデルの語彙はカタカナ語の多くを品詞クラス (`<名詞-一般>`) にまとめるので、n-gram では
//! カタカナ語の中の 1 字の誤りを見分けられない。そこで、コーパスのカタカナ語の出現数の表を持ち、
//! **表に (ほぼ) 無いカタカナ語**が、**1 字違いのよく使われる語**に直せるときだけ指摘する。
//!
//! 固有名詞 (「ノアール」「ダンスク」) を短い別の語 (「アール」「ダンス」) に直す誤検出が多かったので、
//! 先頭・末尾の字を消す直し方と、5 字未満の語は対象にしない。表記の揺れ (小書きの仮名・ヴ・末尾の長音) も出さない。
//! JWTD の開発用 (先頭 5000 件) で、正解 124 件に対して正しい文での指摘 26 件 (うち一部は本物の誤り)。
//!
//! 表は「語 \t 出現数」の TSV (`scripts/katakana_lexicon.py`)。メモリを抑えるため、語の順に並べた 1 本の文字列と
//! 行頭の位置だけを持ち、二分探索で引く (8 万語で約 2MB)。

use anyhow::{Result, bail};

use crate::checker::{EditKind, Finding};

/// 対象にするカタカナ語の最短の長さ (文字数)。
const MIN_LEN: usize = 5;
/// この長さ未満の語は、字を補う直し (「キャラクー → キャラクター」) のときだけ指摘する。
/// 5 字の語を同じ長さ・短い語に直す候補は固有名詞 (「プチペイド → プリペイド」「バイロック → バロック」) の誤検出が多い
/// (Wikipedia の正しい文で 1 万字あたり約 1 件)。
const MIN_LEN_NON_INSERT: usize = 6;
/// この回数以上出てくる語は正しい語とみなす。
const KNOWN_COUNT: u32 = 3;
/// 直し先の語の最少の出現数。
const MIN_TARGET_COUNT: u32 = 30;
/// 直し先の語は、元の語の (出現数 + 1) のこの倍以上出てくること。
const MIN_RATIO: u32 = 10;

pub struct Katakana {
    /// 語の順に並べた「語 \t 出現数 \n」の連結
    text: String,
    /// 各行の先頭のバイト位置
    lines: Vec<u32>,
}

fn is_katakana(c: char) -> bool {
    ('ァ'..='ヺ').contains(&c) || c == 'ー'
}

/// 小書きの仮名と並字の対応 (表記の揺れとして扱う)。
fn small_to_large(c: char) -> Option<char> {
    let small = "ァィゥェォャュョッヮヵヶ";
    let large = "アイウエオヤユヨツワカケ";
    small
        .chars()
        .position(|x| x == c)
        .and_then(|i| large.chars().nth(i))
}

/// 表記の揺れ (「クオーク / クォーク」「ヴァイオリン / バイオリン」「エレベータ / エレベーター」) か。
fn is_notation_variant(a: &[char], b: &[char]) -> bool {
    let trim = |x: &[char]| -> Vec<char> {
        let mut v = x.to_vec();
        while v.last() == Some(&'ー') {
            v.pop();
        }
        v
    };
    if trim(a) == trim(b) {
        return true;
    }
    if a.len() == b.len() {
        let diffs: Vec<(char, char)> = a
            .iter()
            .zip(b)
            .filter(|(x, y)| x != y)
            .map(|(x, y)| (*x, *y))
            .collect();
        if let [(x, y)] = diffs[..] {
            return small_to_large(x) == Some(y)
                || small_to_large(y) == Some(x)
                || x == 'ヴ'
                || y == 'ヴ';
        }
    }
    false
}

impl Katakana {
    /// TSV (語 \t 出現数) を読む。並びは問わない。
    pub fn from_tsv(src: &str) -> Result<Self> {
        let mut rows: Vec<(&str, u32)> = Vec::new();
        for line in src.lines() {
            let Some((w, c)) = line.split_once('\t') else {
                continue;
            };
            let c: u32 = c.trim().parse()?;
            if !w.chars().all(is_katakana) {
                bail!("カタカナ語の表にカタカナ以外の語がある: {w}");
            }
            rows.push((w, c));
        }
        rows.sort_unstable_by(|a, b| a.0.cmp(b.0));
        rows.dedup_by(|a, b| a.0 == b.0);
        let mut text = String::with_capacity(src.len());
        let mut lines = Vec::with_capacity(rows.len());
        for (w, c) in rows {
            lines.push(text.len() as u32);
            text.push_str(w);
            text.push('\t');
            text.push_str(&c.to_string());
            text.push('\n');
        }
        Ok(Self { text, lines })
    }

    /// 語の順に並んだ TSV (`scripts/katakana_lexicon.py` の出力) なら、読んだ文字列をそのまま使う
    /// (並べ直し用の表と文字列の複製を作らないので、読み込み時のメモリが増えない)。並んでいなければ [`Self::from_tsv`]。
    pub fn from_sorted_string(src: String) -> Result<Self> {
        let mut lines = Vec::new();
        let mut prev: Option<&str> = None;
        let mut sorted = true;
        let mut off = 0usize;
        for line in src.split_inclusive('\n') {
            let body = line.trim_end_matches('\n');
            let Some((w, c)) = body.split_once('\t') else {
                sorted = false;
                break;
            };
            if c.trim().parse::<u32>().is_err()
                || !w.chars().all(is_katakana)
                || prev.is_some_and(|p| p >= w)
                || !line.ends_with('\n')
            {
                sorted = false;
                break;
            }
            lines.push(off as u32);
            off += line.len();
            prev = Some(w);
        }
        if !sorted {
            return Self::from_tsv(&src);
        }
        let mut text = src;
        text.shrink_to_fit();
        Ok(Self { text, lines })
    }

    pub fn load(path: &std::path::Path) -> Result<Self> {
        Self::from_sorted_string(std::fs::read_to_string(path)?)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.lines.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    fn row(&self, i: usize) -> (&str, u32) {
        let start = self.lines[i] as usize;
        let end = self
            .lines
            .get(i + 1)
            .map_or(self.text.len(), |x| *x as usize);
        let line = &self.text[start..end - 1];
        let (w, c) = line.split_once('\t').unwrap_or((line, "0"));
        (w, c.parse().unwrap_or(0))
    }

    /// 語の出現数 (表に無ければ 0)。
    #[must_use]
    pub fn count(&self, word: &str) -> u32 {
        self.lines
            .binary_search_by(|&off| {
                let start = off as usize;
                let tab = self.text[start..]
                    .find('\t')
                    .map_or(self.text.len(), |p| start + p);
                self.text[start..tab].cmp(word)
            })
            .map_or(0, |i| self.row(i).1)
    }

    /// 1 字違い (挿入・置換・隣接の入れ替え・語の中の 1 字の削除) の語のうち、条件を満たす最も多い語。
    fn best_neighbor(&self, w: &[char], own: u32) -> Option<String> {
        let mut best: Option<(String, u32)> = None;
        let mut buf = String::with_capacity(w.len() * 3 + 3);
        let mut consider = |cand: &[char], best: &mut Option<(String, u32)>| {
            if cand.len() < MIN_LEN - 1 || is_notation_variant(w, cand) {
                return;
            }
            buf.clear();
            buf.extend(cand.iter());
            let n = self.count(&buf);
            if n >= MIN_TARGET_COUNT
                && n >= MIN_RATIO * (own + 1)
                && best.as_ref().is_none_or(|b| n > b.1)
            {
                *best = Some((buf.clone(), n));
            }
        };
        let alphabet: Vec<char> = ('ァ'..='ヺ').chain(std::iter::once('ー')).collect();
        let mut cand: Vec<char> = Vec::with_capacity(w.len() + 1);
        for i in 0..=w.len() {
            for &ch in &alphabet {
                cand.clear();
                cand.extend_from_slice(&w[..i]);
                cand.push(ch);
                cand.extend_from_slice(&w[i..]);
                consider(&cand, &mut best);
            }
        }
        for i in 0..w.len() {
            for &ch in &alphabet {
                if ch == w[i] {
                    continue;
                }
                cand.clear();
                cand.extend_from_slice(w);
                cand[i] = ch;
                consider(&cand, &mut best);
            }
        }
        // 削除は語の中の字だけ (先頭・末尾を消すと「ノアール → アール」のような別の短い語になりやすい)
        for i in 1..w.len().saturating_sub(1) {
            cand.clear();
            cand.extend_from_slice(&w[..i]);
            cand.extend_from_slice(&w[i + 1..]);
            consider(&cand, &mut best);
        }
        for i in 0..w.len().saturating_sub(1) {
            if w[i] == w[i + 1] {
                continue;
            }
            cand.clear();
            cand.extend_from_slice(w);
            cand.swap(i, i + 1);
            consider(&cand, &mut best);
        }
        best.map(|b| b.0)
    }

    /// 文の中のカタカナ語を調べ、打ち間違いらしいものを指摘する (オフセットは文字単位)。
    #[must_use]
    pub fn find(&self, sent: &str) -> Vec<Finding> {
        let chars: Vec<char> = sent.chars().collect();
        let mut out = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            if !is_katakana(chars[i]) {
                i += 1;
                continue;
            }
            let start = i;
            while i < chars.len() && is_katakana(chars[i]) {
                i += 1;
            }
            let w = &chars[start..i];
            if w.len() < MIN_LEN {
                continue;
            }
            let word: String = w.iter().collect();
            let own = self.count(&word);
            if own >= KNOWN_COUNT {
                continue;
            }
            let Some(fix) = self.best_neighbor(w, own) else {
                continue;
            };
            if w.len() < MIN_LEN_NON_INSERT && fix.chars().count() <= w.len() {
                continue;
            }
            // 直す範囲は共通の前後を除いた最小の範囲にする
            let f: Vec<char> = fix.chars().collect();
            let mut pre = 0;
            while pre < w.len() && pre < f.len() && w[pre] == f[pre] {
                pre += 1;
            }
            let mut suf = 0;
            while suf < w.len() - pre
                && suf < f.len() - pre
                && w[w.len() - 1 - suf] == f[f.len() - 1 - suf]
            {
                suf += 1;
            }
            out.push(Finding {
                start: start + pre,
                end: start + w.len() - suf,
                original: w[pre..w.len() - suf].iter().collect(),
                replacement: f[pre..f.len() - suf].iter().collect(),
                kind: EditKind::Pattern,
                delta: (self.count(&fix) as f32).log10() - 1.0,
                alternatives: Vec::new(),
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lexicon() -> Katakana {
        Katakana::from_tsv(
            "オブジェクト\t1184\nキャラクター\t7571\nモロッコ\t721\nアール\t665\nダンス\t1134\nクウェート\t383\nブランド\t4509\nペンシルベニア\t493\n",
        )
        .unwrap()
    }

    fn fixes(k: &Katakana, s: &str) -> Vec<(String, String)> {
        k.find(s)
            .into_iter()
            .map(|f| (f.original, f.replacement))
            .collect()
    }

    #[test]
    fn sorted_input_is_used_as_is_and_unsorted_input_is_sorted() {
        let sorted = "アイデア\t5\nキャラクター\t7571\n".to_string();
        let k = Katakana::from_sorted_string(sorted.clone()).unwrap();
        assert_eq!(k.text, sorted);
        assert_eq!(k.count("キャラクター"), 7571);
        // 並んでいない入力・最終行に改行が無い入力も読める
        let k =
            Katakana::from_sorted_string("キャラクター\t7571\nアイデア\t5".to_string()).unwrap();
        assert_eq!(k.count("アイデア"), 5);
        assert_eq!(k.count("キャラクター"), 7571);
    }

    #[test]
    fn looks_up_counts_by_binary_search() {
        let k = lexicon();
        assert_eq!(k.count("モロッコ"), 721);
        assert_eq!(k.count("モッロコ"), 0);
        assert_eq!(k.len(), 8);
    }

    #[test]
    fn fixes_one_character_typos_of_frequent_words() {
        let k = lexicon();
        // 置換
        assert_eq!(
            fixes(&k, "オプジェクト指向"),
            [("プ".to_string(), "ブ".to_string())]
        );
        // 語末の脱字 (5 字以上)
        assert_eq!(
            fixes(&k, "キャラクーの設定"),
            [(String::new(), "タ".to_string())]
        );
        // 入れ替え (「ベル」→「ルベ」は最小の範囲で示す)
        let f = k.find("州都はペンシベルニア州にある");
        assert_eq!(f.len(), 1);
        assert_eq!(
            (f[0].start, f[0].end, f[0].replacement.as_str()),
            (6, 8, "ルベ")
        );
    }

    #[test]
    fn five_letter_words_are_fixed_only_by_inserting_a_letter() {
        let k =
            Katakana::from_tsv("キャラクター\t7571\nプリペイド\t3000\nバロック\t2000\n").unwrap();
        // 5 字の語の脱字は直す
        assert_eq!(
            fixes(&k, "キャラクーの設定"),
            [(String::new(), "タ".to_string())]
        );
        // 5 字の語を同じ長さ・短い語に直すのは、固有名詞 (「プチペイド」「バイロック」) の誤検出が多いので出さない
        assert!(k.find("プチペイドが存在した").is_empty());
        assert!(k.find("バイロックは降参した").is_empty());
    }

    #[test]
    fn leaves_known_words_short_words_and_proper_noun_shortening_alone() {
        let k = lexicon();
        // 表にある語
        assert!(k.find("オブジェクト指向").is_empty());
        // 4 字以下
        assert!(k.find("ダンヌ").is_empty());
        // 先頭・末尾を消して短い別の語にする直し方はしない (固有名詞の誤検出が多い)
        assert!(k.find("ノアールの作品").is_empty());
        assert!(k.find("ラブランドの作品").is_empty());
        // 表記の揺れ (小書き) は出さない
        assert!(
            k.find("クウェートとクェート")
                .iter()
                .all(|f| f.original != "ェ")
        );
    }
}
