//! 軽量版の言語モデル: n-gram の「有無」だけを XOR 系フィルタ (binary fuse, 1 件約 9bit) に持つ。
//!
//! 確率と backoff を持たないので、同じ n-gram 数なら確率モデルの約 1/4 の大きさになる。
//! スコアは「実在する最長 n-gram の次数」から作る擬似対数確率で、候補の比較 (Δ) と
//! 未出現ゲートはそのまま使える。精度は確率モデルより落ちる前提で、検出 (誤りの疑いの提示) 用。

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use anyhow::{Result, bail};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use xorf::{BinaryFuse8, Filter};

use crate::lm::{BOS, EOS, LanguageModel, MAX_ORDER, UNK, corpus_key, key_hash, pack};

/// 1 次数下がるごとの減点 (log10 相当)。3-gram があれば 0、2-gram だけなら -STEP、…
const STEP: f32 = 1.5;

pub struct NgramSet {
    order: usize,
    vocab: FxHashMap<String, u32>,
    /// filters[n - 2] が n-gram (n = 2..=order) の集合
    filters: Vec<BinaryFuse8>,
}

impl NgramSet {
    fn contains(&self, ids: &[u32]) -> bool {
        let n = ids.len();
        n >= 2 && n <= self.order && self.filters[n - 2].contains(&key_hash(pack(ids)))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(b"CELSONS1")?;
        w.write_all(&(self.order as u32).to_le_bytes())?;
        let mut words: Vec<(&String, &u32)> = self.vocab.iter().collect();
        words.sort_by_key(|(_, id)| **id);
        w.write_all(&(words.len() as u32).to_le_bytes())?;
        for (word, id) in words {
            w.write_all(&id.to_le_bytes())?;
            w.write_all(&(word.len() as u16).to_le_bytes())?;
            w.write_all(word.as_bytes())?;
        }
        for f in &self.filters {
            let bytes = bincode::encode_to_vec(f, bincode::config::standard())?;
            w.write_all(&(bytes.len() as u64).to_le_bytes())?;
            w.write_all(&bytes)?;
        }
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let mut r = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != b"CELSONS1" {
            bail!("not a celso n-gram set");
        }
        let mut b4 = [0u8; 4];
        let mut b2 = [0u8; 2];
        let mut b8 = [0u8; 8];
        r.read_exact(&mut b4)?;
        let order = u32::from_le_bytes(b4) as usize;
        r.read_exact(&mut b4)?;
        let nwords = u32::from_le_bytes(b4) as usize;
        let mut vocab = FxHashMap::default();
        for _ in 0..nwords {
            r.read_exact(&mut b4)?;
            let id = u32::from_le_bytes(b4);
            r.read_exact(&mut b2)?;
            let mut s = vec![0u8; u16::from_le_bytes(b2) as usize];
            r.read_exact(&mut s)?;
            vocab.insert(String::from_utf8(s)?, id);
        }
        let mut filters = Vec::new();
        for _ in 2..=order {
            r.read_exact(&mut b8)?;
            let mut buf = vec![0u8; u64::from_le_bytes(b8) as usize];
            r.read_exact(&mut buf)?;
            let (f, _): (BinaryFuse8, usize) =
                bincode::decode_from_slice(&buf, bincode::config::standard())?;
            filters.push(f);
        }
        Ok(Self {
            order,
            vocab,
            filters,
        })
    }
}

impl LanguageModel for NgramSet {
    fn order(&self) -> usize {
        self.order
    }

    fn word_id(&self, w: &str) -> u32 {
        self.vocab.get(w).copied().unwrap_or(UNK)
    }

    fn match_order(&self, ctx: &[u32], w: u32) -> usize {
        if w == UNK {
            return 0;
        }
        let n_max = self.order.min(ctx.len() + 1);
        let mut buf = [0u32; MAX_ORDER];
        for n in (2..=n_max).rev() {
            buf[..n - 1].copy_from_slice(&ctx[ctx.len() - (n - 1)..]);
            buf[n - 1] = w;
            if self.contains(&buf[..n]) {
                return n;
            }
        }
        1
    }

    fn logp(&self, ctx: &[u32], w: u32) -> f32 {
        let m = self.match_order(ctx, w);
        // 文頭付近 (文脈が短い) では、取りうる最長の次数を基準にする
        let full = self.order.min(ctx.len() + 1);
        -(full.saturating_sub(m.max(1)) as f32) * STEP - if m == 0 { STEP } else { 0.0 }
    }

    fn vocab_len(&self) -> usize {
        self.vocab.len()
    }

    fn words(&self) -> Vec<(&str, u32)> {
        self.vocab.iter().map(|(w, id)| (w.as_str(), *id)).collect()
    }
}

/// 分かち書き済みコーパスから n-gram の集合を作る。`min_count[n-1]` 回未満の n-gram は入れない。
pub fn build(
    paths: &[impl AsRef<Path>],
    order: usize,
    min_word_count: u32,
    min_count: [u32; MAX_ORDER],
    vocab_filter: Option<&FxHashSet<String>>,
) -> Result<NgramSet> {
    assert!((2..=MAX_ORDER).contains(&order));
    let mut wc: FxHashMap<String, u32> = FxHashMap::default();
    for p in paths {
        for line in BufReader::new(File::open(p.as_ref())?).lines() {
            for w in line?.split(' ').filter(|w| !w.is_empty()) {
                *wc.entry(corpus_key(w, vocab_filter).to_string())
                    .or_default() += 1;
            }
        }
    }
    let mut words: Vec<(String, u32)> = wc
        .into_iter()
        .filter(|(_, c)| *c >= min_word_count)
        .collect();
    words.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut vocab: FxHashMap<String, u32> = FxHashMap::default();
    vocab.insert("<unk>".into(), UNK);
    vocab.insert("<s>".into(), BOS);
    vocab.insert("</s>".into(), EOS);
    for (i, (w, _)) in words.into_iter().enumerate() {
        vocab.insert(w, i as u32 + 4);
    }
    eprintln!("vocab: {}", vocab.len());
    let mut corpus: Vec<u32> = Vec::new();
    for p in paths {
        for line in BufReader::new(File::open(p.as_ref())?).lines() {
            let line = line?;
            corpus.push(BOS);
            for w in line.split(' ').filter(|w| !w.is_empty()) {
                corpus.push(
                    vocab
                        .get(corpus_key(w, vocab_filter))
                        .copied()
                        .unwrap_or(UNK),
                );
            }
            corpus.push(EOS);
        }
    }
    let mut filters = Vec::new();
    for n in 2..=order {
        let mut keys: Vec<u128> = (0..corpus.len().saturating_sub(n - 1))
            .into_par_iter()
            .filter(|&i| !corpus[i..i + n - 1].contains(&EOS))
            .map(|i| pack(&corpus[i..i + n]))
            .collect();
        keys.par_sort_unstable();
        let mut hashes: Vec<u64> = Vec::new();
        let mut i = 0;
        while i < keys.len() {
            let mut j = i;
            while j < keys.len() && keys[j] == keys[i] {
                j += 1;
            }
            if (j - i) as u32 >= min_count[n - 1] {
                hashes.push(key_hash(keys[i]));
            }
            i = j;
        }
        hashes.par_sort_unstable();
        hashes.dedup();
        eprintln!("order {n}: {} kept", hashes.len());
        let f = BinaryFuse8::try_from(&hashes).map_err(|e| anyhow::anyhow!("{e}"))?;
        filters.push(f);
    }
    Ok(NgramSet {
        order,
        vocab,
        filters,
    })
}
