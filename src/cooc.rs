//! 同音異字の判定に使う、文内共起の小さなモデル。
//!
//! n-gram は前後 2 語しか見ないので「衛生 / 衛星」「障害 / 傷害」の違いが出にくい。
//! 同音異字の組になる語 (約 9 千語) について、同じ文に出やすい語を PMI で上位 K 語だけ持ち、
//! 候補の語と元の語で「文中の語との相性」の差を足す。サイズは数 MB に収まる。

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{Result, bail};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::lm::{LanguageModel, UNK, corpus_key};

/// PMI (自然対数) を 8bit に詰めるときの 1 段の幅
const SCALE: f32 = 1.0 / 24.0;

pub struct Cooc {
    /// 同音異字の語 ID → `ids` / `vals` の中の範囲 (開始, 長さ)
    heads: FxHashMap<u32, (u32, u16)>,
    /// 共起語 ID (語ごとに昇順) と、PMI を量子化した値。メモリを抑えるため平坦な配列で持つ
    ids: Vec<u16>,
    vals: Vec<u8>,
}

impl Cooc {
    /// `h` と文中の語 `ctx` との相性 (PMI の和, 自然対数)。
    #[must_use]
    pub fn score(&self, h: u32, ctx: &[u32]) -> f32 {
        let Some(&(start, len)) = self.heads.get(&h) else {
            return 0.0;
        };
        let (start, len) = (start as usize, len as usize);
        let row = &self.ids[start..start + len];
        let mut s = 0.0;
        for &w in ctx {
            if w == h || w > u32::from(u16::MAX) {
                continue;
            }
            if let Ok(i) = row.binary_search(&(w as u16)) {
                s += f32::from(self.vals[start + i]) * SCALE;
            }
        }
        s
    }

    #[must_use]
    pub fn contains(&self, h: u32) -> bool {
        self.heads.contains_key(&h)
    }

    fn from_rows(mut rows: Vec<(u32, Vec<(u32, u8)>)>) -> Self {
        rows.sort_by_key(|r| r.0);
        let mut heads = FxHashMap::default();
        heads.reserve(rows.len());
        let mut ids = Vec::new();
        let mut vals = Vec::new();
        for (h, row) in rows {
            heads.insert(h, (ids.len() as u32, row.len() as u16));
            for (w, v) in row {
                ids.push(w as u16);
                vals.push(v);
            }
        }
        Self { heads, ids, vals }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(b"CELSOCO2")?;
        w.write_all(&(self.heads.len() as u32).to_le_bytes())?;
        let mut keys: Vec<(&u32, &(u32, u16))> = self.heads.iter().collect();
        keys.sort();
        for (h, &(start, len)) in keys {
            // 語彙は 6.5 万語未満なので語 ID は 16bit で足りる (1 件 3 バイト)
            if *h > u32::from(u16::MAX) {
                bail!("vocabulary too large for CELSOCO2");
            }
            w.write_all(&(*h as u16).to_le_bytes())?;
            w.write_all(&len.to_le_bytes())?;
            for i in start as usize..start as usize + len as usize {
                w.write_all(&self.ids[i].to_le_bytes())?;
                w.write_all(&[self.vals[i]])?;
            }
        }
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        // 一時的な行ごとの配列を作らず、最終形の平坦な配列へ直接読む (読み込み時のメモリを抑える)
        let bytes = std::fs::read(path)?;
        if bytes.len() < 12 || &bytes[..8] != b"CELSOCO2" {
            bail!("not a celso co-occurrence model (CELSOCO2)");
        }
        let n = u32::from_le_bytes(bytes[8..12].try_into()?) as usize;
        let mut heads = FxHashMap::default();
        heads.reserve(n);
        let total = (bytes.len() - 12 - n * 4) / 3;
        let mut ids = Vec::with_capacity(total);
        let mut vals = Vec::with_capacity(total);
        let mut p = 12;
        for _ in 0..n {
            if p + 4 > bytes.len() {
                bail!("truncated co-occurrence model");
            }
            let h = u32::from(u16::from_le_bytes([bytes[p], bytes[p + 1]]));
            let len = u16::from_le_bytes([bytes[p + 2], bytes[p + 3]]);
            p += 4;
            if p + len as usize * 3 > bytes.len() {
                bail!("truncated co-occurrence model");
            }
            heads.insert(h, (ids.len() as u32, len));
            for _ in 0..len {
                ids.push(u16::from_le_bytes([bytes[p], bytes[p + 1]]));
                vals.push(bytes[p + 2]);
                p += 3;
            }
        }
        Ok(Self { heads, ids, vals })
    }
}

/// 文中の語のうち、共起の手がかりにする語 (漢字を含む語彙内の語) の ID。
pub fn context_ids(
    lm: &dyn LanguageModel,
    surfaces: impl Iterator<Item = impl AsRef<str>>,
) -> Vec<u32> {
    let mut v: Vec<u32> = surfaces
        .filter(|s| {
            s.as_ref()
                .chars()
                .any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c))
        })
        .map(|s| lm.word_id(s.as_ref()))
        .filter(|&id| id != UNK)
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// 分かち書き済みコーパス (`--with-class` 形式) から作る。語 ID は `lm` の語彙に合わせる。
pub fn build(
    paths: &[impl AsRef<Path>],
    lm: &dyn LanguageModel,
    vocab_filter: Option<&FxHashSet<String>>,
    homophones: &FxHashSet<String>,
    top_k: usize,
    min_pair: u32,
) -> Result<Cooc> {
    let hset: FxHashSet<u32> = homophones
        .iter()
        .map(|w| lm.word_id(w))
        .filter(|&id| id != UNK)
        .collect();
    let mut unigram: FxHashMap<u32, u32> = FxHashMap::default();
    let mut pair: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    let mut nsent: u64 = 0;
    for p in paths {
        for line in BufReader::new(File::open(p.as_ref())?).lines() {
            let line = line?;
            let ids = context_ids(
                lm,
                line.split(' ')
                    .filter(|w| !w.is_empty())
                    .map(|w| corpus_key(w, vocab_filter)),
            );
            if ids.is_empty() {
                continue;
            }
            nsent += 1;
            for &w in &ids {
                *unigram.entry(w).or_default() += 1;
            }
            for &h in ids.iter().filter(|h| hset.contains(h)) {
                for &w in &ids {
                    if w != h {
                        *pair.entry((h, w)).or_default() += 1;
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
    let n = nsent as f64;
    let mut rows: FxHashMap<u32, Vec<(u32, f32)>> = FxHashMap::default();
    for ((h, w), c) in pair {
        if c < min_pair {
            continue;
        }
        let pmi =
            ((f64::from(c) * n) / (f64::from(unigram[&h]) * f64::from(unigram[&w]))).ln() as f32;
        if pmi > 0.0 {
            rows.entry(h).or_default().push((w, pmi));
        }
    }
    let mut out = Vec::new();
    for (h, mut v) in rows {
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v.truncate(top_k);
        if h > u32::from(u16::MAX) || v.iter().any(|e| e.0 > u32::from(u16::MAX)) {
            bail!("vocabulary too large for CELSOCO2");
        }
        let mut q: Vec<(u32, u8)> = v
            .into_iter()
            .map(|(w, p)| (w, (p / SCALE).round().clamp(1.0, 255.0) as u8))
            .collect();
        q.sort_by_key(|e| e.0);
        out.push((h, q));
    }
    Ok(Cooc::from_rows(out))
}
