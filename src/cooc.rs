//! 同音異字の判定に使う、文内共起の小さなモデル。
//!
//! n-gram は前後 2 語しか見ないので「衛生 / 衛星」「障害 / 傷害」の違いが出にくい。
//! 同音異字の組になる語 (約 9 千語) について、同じ文に出やすい語を PMI で上位 K 語だけ持ち、
//! 候補の語と元の語で「文中の語との相性」の差を足す。サイズは数 MB に収まる。
//!
//! 手がかりにする文中の語は、言語モデルの語彙 (約 1.5 万語) とは別に持てる (CELSOCO3)。
//! 「祭祀」「献上」のような分野の語は言語モデルの語彙に入らないが、誤変換の判断には効く。

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{Result, bail};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::lm::{LanguageModel, UNK, corpus_key};

/// PMI (自然対数) を 8bit に詰めるときの 1 段の幅
const SCALE: f32 = 1.0 / 24.0;

fn has_kanji(s: &str) -> bool {
    s.chars().any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c))
}

/// モデルの中身。配布物は mmap で開き、作った直後はメモリ上のバイト列をそのまま使う。
enum Backing {
    Owned(Vec<u8>),
    Mapped(memmap2::Mmap),
}

impl Backing {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Owned(v) => v,
            Self::Mapped(m) => m,
        }
    }
}

/// 同音異字の文内共起モデル。
///
/// 形式 (CELSOCO2 / CELSOCO3):
/// - magic 8 バイト
/// - (CELSOCO3 のみ) 手がかりの語彙: 語数 u32、続いて「長さ u8 + UTF-8」をバイト順に昇順で並べる。
///   並び順がそのまま共起語 ID になるので、引くときは二分探索でよい (ハッシュ表に展開しない)
/// - 同音異字の語数 u32、続いて語ごとに「語 ID u16、件数 u16、(共起語 ID u16, 値 u8) × 件数」
///   (共起語 ID の昇順)
///
/// 表をメモリへ展開せず、ファイル上を直接探すので、常駐するのは語の位置の索引だけで済む。
pub struct Cooc {
    data: Backing,
    /// 同音異字の語 ID (言語モデルの語 ID) → 行の開始バイト位置と件数
    heads: FxHashMap<u32, (u32, u16)>,
    /// 手がかりの語彙の各語の開始バイト位置 (長さのバイト)。無ければ言語モデルの語 ID を使う
    ctx: Option<Vec<u32>>,
}

impl Cooc {
    /// `h` と文中の語 `ctx` との相性 (PMI の和, 自然対数)。`skip` (置き換える元の語) は数えない。
    #[must_use]
    pub fn score(&self, h: u32, ctx: &[u32], skip: Option<u32>) -> f32 {
        let Some(&(start, len)) = self.heads.get(&h) else {
            return 0.0;
        };
        let start = start as usize;
        let row = &self.data.bytes()[start..start + len as usize * 3];
        let id_at = |i: usize| u16::from_le_bytes([row[i * 3], row[i * 3 + 1]]);
        let mut s = 0.0;
        for &w in ctx {
            if Some(w) == skip || w > u32::from(u16::MAX) {
                continue;
            }
            let w = w as u16;
            // 行は共起語 ID の昇順
            let (mut lo, mut hi) = (0usize, len as usize);
            while lo < hi {
                let mid = (lo + hi) / 2;
                match id_at(mid).cmp(&w) {
                    std::cmp::Ordering::Less => lo = mid + 1,
                    std::cmp::Ordering::Greater => hi = mid,
                    std::cmp::Ordering::Equal => {
                        s += f32::from(row[mid * 3 + 2]) * SCALE;
                        break;
                    }
                }
            }
        }
        s
    }

    #[must_use]
    pub fn contains(&self, h: u32) -> bool {
        self.heads.contains_key(&h)
    }

    /// 表層形 1 つの共起語 ID (手がかりにしない語は None)。
    #[must_use]
    pub fn ctx_id(&self, lm: &dyn LanguageModel, s: &str) -> Option<u32> {
        if !has_kanji(s) {
            return None;
        }
        let Some(offsets) = &self.ctx else {
            return Some(lm.word_id(s)).filter(|&id| id != UNK);
        };
        let bytes = self.data.bytes();
        let word_at = |o: u32| {
            let o = o as usize;
            &bytes[o + 1..o + 1 + bytes[o] as usize]
        };
        offsets
            .binary_search_by(|&o| word_at(o).cmp(s.as_bytes()))
            .ok()
            .map(|i| i as u32)
    }

    /// 文中の語のうち、共起の手がかりにする語の ID (昇順・重複なし)。
    pub fn context(
        &self,
        lm: &dyn LanguageModel,
        surfaces: impl Iterator<Item = impl AsRef<str>>,
    ) -> Vec<u32> {
        let mut v: Vec<u32> = surfaces
            .filter_map(|s| self.ctx_id(lm, s.as_ref()))
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// 行 (共起語 ID の昇順) と手がかりの語彙 (バイト順に昇順) からモデルを作る。
    fn from_rows(mut rows: Vec<(u32, Vec<(u32, u8)>)>, ctx_words: &[String]) -> Result<Self> {
        rows.sort_by_key(|r| r.0);
        let mut b: Vec<u8> = Vec::new();
        if ctx_words.is_empty() {
            b.extend_from_slice(b"CELSOCO2");
        } else {
            if !ctx_words.windows(2).all(|w| w[0] < w[1]) {
                bail!("context vocabulary must be sorted and unique");
            }
            b.extend_from_slice(b"CELSOCO3");
            b.extend_from_slice(&(ctx_words.len() as u32).to_le_bytes());
            for w in ctx_words {
                b.push(u8::try_from(w.len())?);
                b.extend_from_slice(w.as_bytes());
            }
        }
        b.extend_from_slice(&(rows.len() as u32).to_le_bytes());
        for (h, row) in &rows {
            // 語彙は 6.5 万語未満なので語 ID は 16bit で足りる (1 件 3 バイト)
            if *h > u32::from(u16::MAX) || row.iter().any(|e| e.0 > u32::from(u16::MAX)) {
                bail!("vocabulary too large for CELSOCO2/3");
            }
            b.extend_from_slice(&(*h as u16).to_le_bytes());
            b.extend_from_slice(&u16::try_from(row.len())?.to_le_bytes());
            for &(w, v) in row {
                b.extend_from_slice(&(w as u16).to_le_bytes());
                b.push(v);
            }
        }
        Self::parse(Backing::Owned(b))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::write(path, self.data.bytes())?;
        Ok(())
    }

    /// mmap で開く (行の中身は読み込まず、語の位置の索引だけを作る)。
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: 読み取り専用で開いたモデルファイルを mmap する。実行中に書き換えないこと。
        let map = unsafe { memmap2::Mmap::map(&file)? };
        Self::parse(Backing::Mapped(map))
    }

    fn parse(data: Backing) -> Result<Self> {
        let bytes = data.bytes();
        let truncated = || anyhow::anyhow!("truncated co-occurrence model");
        let u16_at = |p: usize| -> Result<u16> {
            Ok(u16::from_le_bytes(
                bytes.get(p..p + 2).ok_or_else(truncated)?.try_into()?,
            ))
        };
        let u32_at = |p: usize| -> Result<u32> {
            Ok(u32::from_le_bytes(
                bytes.get(p..p + 4).ok_or_else(truncated)?.try_into()?,
            ))
        };
        let mut p = 8;
        let ctx = match bytes.get(..8) {
            Some(b"CELSOCO2") => None,
            Some(b"CELSOCO3") => {
                let n = u32_at(p)? as usize;
                p += 4;
                let mut offsets = Vec::with_capacity(n);
                for _ in 0..n {
                    offsets.push(u32::try_from(p)?);
                    p += 1 + *bytes.get(p).ok_or_else(truncated)? as usize;
                }
                Some(offsets)
            }
            _ => bail!("not a celso co-occurrence model (CELSOCO2/3)"),
        };
        let n = u32_at(p)? as usize;
        p += 4;
        let mut heads = FxHashMap::default();
        heads.reserve(n);
        for _ in 0..n {
            let h = u32::from(u16_at(p)?);
            let len = u16_at(p + 2)?;
            p += 4;
            if p + len as usize * 3 > bytes.len() {
                return Err(truncated());
            }
            heads.insert(h, (u32::try_from(p)?, len));
            p += len as usize * 3;
        }
        Ok(Self { data, heads, ctx })
    }
}

/// コーパスから数えた共起の集計 (選び方を変えて何度も作り直せるように保存できる)。
pub struct Stats {
    /// 手がかりの語を 1 つ以上含む文の数
    pub nsent: u64,
    /// 同音異字の語 (言語モデルの語 ID) → その語を含む文の数
    pub heads: FxHashMap<u32, u32>,
    /// 共起語 ID → その語を含む文の数
    pub ctx: FxHashMap<u32, u32>,
    /// 手がかりの語彙 (共起語 ID 順)。空なら共起語 ID は言語モデルの語 ID
    pub ctx_words: Vec<String>,
    /// (同音異字の語, 共起語, 両方を含む文の数)。`min_pair` 未満は捨ててある
    pub pairs: Vec<(u32, u32, u32)>,
}

/// 分かち書き済みコーパス (`--with-class` 形式) から共起を数える。
///
/// `ctx_words` を渡すと、手がかりの語をその語彙 (漢字を含む語だけ使う) で数える。
/// 空なら言語モデルの語彙で数える。
pub fn count(
    paths: &[impl AsRef<Path>],
    lm: &dyn LanguageModel,
    vocab_filter: Option<&FxHashSet<String>>,
    homophones: &FxHashSet<String>,
    ctx_words: &[String],
    min_pair: u32,
) -> Result<Stats> {
    let hset: FxHashSet<u32> = homophones
        .iter()
        .map(|w| lm.word_id(w))
        .filter(|&id| id != UNK)
        .collect();
    // 共起語 ID は語彙のバイト順の位置にする (モデルの語彙を二分探索で引けるように)
    let mut ctx_words: Vec<String> = ctx_words.iter().filter(|w| has_kanji(w)).cloned().collect();
    ctx_words.sort_unstable();
    ctx_words.dedup();
    if ctx_words.len() > usize::from(u16::MAX) {
        bail!("context vocabulary too large (max 65535)");
    }
    let ctx_map: FxHashMap<&str, u32> = ctx_words
        .iter()
        .enumerate()
        .map(|(i, w)| (w.as_str(), i as u32))
        .collect();
    let mut heads: FxHashMap<u32, u32> = FxHashMap::default();
    let mut ctx: FxHashMap<u32, u32> = FxHashMap::default();
    let mut pair: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    let mut nsent: u64 = 0;
    let mut hs: Vec<(u32, u32)> = Vec::new();
    let mut cs: Vec<u32> = Vec::new();
    for p in paths {
        for line in BufReader::new(File::open(p.as_ref())?).lines() {
            let line = line?;
            hs.clear();
            cs.clear();
            for w in line.split(' ').filter(|w| !w.is_empty()) {
                let surface = w.split_once('\x1f').map_or(w, |(s, _)| s);
                if !has_kanji(surface) {
                    continue;
                }
                let lm_id = lm.word_id(corpus_key(w, vocab_filter));
                let cid = if ctx_map.is_empty() {
                    Some(lm_id).filter(|&id| id != UNK)
                } else {
                    ctx_map.get(surface).copied()
                };
                if let Some(c) = cid {
                    cs.push(c);
                }
                if hset.contains(&lm_id) {
                    // 自分自身は共起に数えない
                    hs.push((lm_id, cid.unwrap_or(u32::MAX)));
                }
            }
            if cs.is_empty() {
                continue;
            }
            cs.sort_unstable();
            cs.dedup();
            hs.sort_unstable();
            hs.dedup_by_key(|e| e.0);
            nsent += 1;
            for &c in &cs {
                *ctx.entry(c).or_default() += 1;
            }
            for &(h, self_id) in &hs {
                *heads.entry(h).or_default() += 1;
                for &c in &cs {
                    if c != self_id {
                        *pair.entry((h, c)).or_default() += 1;
                    }
                }
            }
        }
        eprintln!(
            "{}: {} sentences, {} pairs",
            p.as_ref().display(),
            nsent,
            pair.len()
        );
    }
    let mut pairs: Vec<(u32, u32, u32)> = pair
        .into_iter()
        .filter(|&(_, c)| c >= min_pair)
        .map(|((h, w), c)| (h, w, c))
        .collect();
    pairs.sort_unstable();
    Ok(Stats {
        nsent,
        heads,
        ctx,
        ctx_words,
        pairs,
    })
}

impl Stats {
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(b"CELSOCS2")?;
        w.write_all(&self.nsent.to_le_bytes())?;
        for m in [&self.heads, &self.ctx] {
            w.write_all(&(m.len() as u64).to_le_bytes())?;
            for (k, v) in m {
                w.write_all(&k.to_le_bytes())?;
                w.write_all(&v.to_le_bytes())?;
            }
        }
        w.write_all(&(self.ctx_words.len() as u64).to_le_bytes())?;
        for s in &self.ctx_words {
            w.write_all(&[u8::try_from(s.len())?])?;
            w.write_all(s.as_bytes())?;
        }
        w.write_all(&(self.pairs.len() as u64).to_le_bytes())?;
        for (h, c, n) in &self.pairs {
            w.write_all(&h.to_le_bytes())?;
            w.write_all(&c.to_le_bytes())?;
            w.write_all(&n.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let b = std::fs::read(path)?;
        if b.len() < 16 || &b[..8] != b"CELSOCS2" {
            bail!("not a celso co-occurrence stats file (CELSOCS2)");
        }
        let u64_at = |p: usize| u64::from_le_bytes(b[p..p + 8].try_into().unwrap());
        let u32_at = |p: usize| u32::from_le_bytes(b[p..p + 4].try_into().unwrap());
        let nsent = u64_at(8);
        let mut p = 16;
        let mut maps = [FxHashMap::default(), FxHashMap::default()];
        for m in &mut maps {
            let n = u64_at(p) as usize;
            p += 8;
            m.reserve(n);
            for _ in 0..n {
                m.insert(u32_at(p), u32_at(p + 4));
                p += 8;
            }
        }
        let [heads, ctx] = maps;
        let nw = u64_at(p) as usize;
        p += 8;
        let mut ctx_words = Vec::with_capacity(nw);
        for _ in 0..nw {
            let len = b[p] as usize;
            ctx_words.push(std::str::from_utf8(&b[p + 1..p + 1 + len])?.to_string());
            p += 1 + len;
        }
        let np = u64_at(p) as usize;
        p += 8;
        let mut pairs = Vec::with_capacity(np);
        for _ in 0..np {
            pairs.push((u32_at(p), u32_at(p + 4), u32_at(p + 8)));
            p += 12;
        }
        Ok(Self {
            nsent,
            heads,
            ctx,
            ctx_words,
            pairs,
        })
    }

    /// 語ごとに PMI の高い共起語を上位 `top_k` 語持つ。
    ///
    /// `weight_by_count` なら「共起回数 × PMI」の大きい順に選ぶ (まれな語ばかりにならないように)。
    pub fn select_pmi(&self, top_k: usize, min_pair: u32, weight_by_count: bool) -> Result<Cooc> {
        let n = self.nsent as f64;
        let mut rows: FxHashMap<u32, Vec<(u32, f32, f32)>> = FxHashMap::default();
        for &(h, w, c) in &self.pairs {
            if c < min_pair {
                continue;
            }
            let pmi = ((f64::from(c) * n) / (f64::from(self.heads[&h]) * f64::from(self.ctx[&w])))
                .ln() as f32;
            if pmi > 0.0 {
                let rank = if weight_by_count { pmi * c as f32 } else { pmi };
                rows.entry(h).or_default().push((w, pmi, rank));
            }
        }
        let mut out = Vec::new();
        for (h, mut v) in rows {
            v.sort_by(|a, b| b.2.total_cmp(&a.2));
            v.truncate(top_k);
            if h > u32::from(u16::MAX) || v.iter().any(|e| e.0 > u32::from(u16::MAX)) {
                bail!("vocabulary too large for CELSOCO2");
            }
            let mut q: Vec<(u32, u8)> = v
                .into_iter()
                .map(|(w, p, _)| (w, (p / SCALE).round().clamp(1.0, 255.0) as u8))
                .collect();
            q.sort_by_key(|e| e.0);
            out.push((h, q));
        }
        Cooc::from_rows(out, &self.ctx_words)
    }
}

/// 分かち書き済みコーパス (`--with-class` 形式) から作る (PMI 上位 `top_k` 語)。
pub fn build(
    paths: &[impl AsRef<Path>],
    lm: &dyn LanguageModel,
    vocab_filter: Option<&FxHashSet<String>>,
    homophones: &FxHashSet<String>,
    ctx_words: &[String],
    top_k: usize,
    min_pair: u32,
) -> Result<Cooc> {
    count(paths, lm, vocab_filter, homophones, ctx_words, min_pair)?
        .select_pmi(top_k, min_pair, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 手がかりの語彙を持つモデルでは言語モデルを引かないので、空の言語モデルで足りる。
    struct NoLm;
    impl LanguageModel for NoLm {
        fn order(&self) -> usize {
            3
        }
        fn word_id(&self, _: &str) -> u32 {
            UNK
        }
        fn logp(&self, _: &[u32], _: u32) -> f32 {
            0.0
        }
        fn match_order(&self, _: &[u32], _: u32) -> usize {
            0
        }
        fn vocab_len(&self) -> usize {
            0
        }
        fn words(&self) -> Vec<(&str, u32)> {
            Vec::new()
        }
    }

    /// 手がかりの語彙つきのモデルを、保存 → mmap で読み直しても同じ採点になる。
    #[test]
    fn context_vocabulary_model_round_trips_through_file() {
        let ctx_words: Vec<String> = ["人工衛星", "打ち上げ", "食品", "管理"]
            .iter()
            .map(|s| (*s).to_string())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let id = |w: &str| ctx_words.iter().position(|x| x == w).unwrap() as u32;
        // 語 ID 1 =「衛星」、2 =「衛生」とする
        let rows = vec![
            (1, {
                let mut r = vec![(id("人工衛星"), 48), (id("打ち上げ"), 24)];
                r.sort_unstable();
                r
            }),
            (2, {
                let mut r = vec![(id("食品"), 48), (id("管理"), 24)];
                r.sort_unstable();
                r
            }),
        ];
        let built = Cooc::from_rows(rows, &ctx_words).unwrap();
        let path = std::env::temp_dir().join(format!("celso-cooc-test-{}.bin", std::process::id()));
        built.save(&path).unwrap();
        let loaded = Cooc::load(&path).unwrap();
        let lm = NoLm;
        for m in [&built, &loaded] {
            assert_eq!(m.ctx_id(&lm, "食品"), Some(id("食品")));
            assert_eq!(m.ctx_id(&lm, "未知語"), None);
            // 平仮名だけの語は手がかりにしない
            assert_eq!(m.ctx_id(&lm, "たべもの"), None);
            let ctx = m.context(&lm, ["食品", "の", "管理"].iter());
            assert!((m.score(2, &ctx, None) - 3.0).abs() < 1e-6);
            assert!(m.score(1, &ctx, None).abs() < 1e-6);
            // 置き換える元の語は数えない
            assert!((m.score(2, &ctx, Some(id("食品"))) - 1.0).abs() < 1e-6);
        }
        std::fs::remove_file(path).ok();
    }
}
