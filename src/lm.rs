//! 単語 n-gram 言語モデル (補間 modified Kneser-Ney)。
//!
//! * 構築: 分かち書き済みコーパスから次数ごとに n-gram を列挙 → ソート → ランレングスで数える。
//!   KN の低次は「左に何種類の語が来たか」(continuation count) を使う。
//! * 保存形式: ARPA と同じく「補間済み確率 + backoff 重み」を n-gram ごとに持たせ、
//!   全次数を 1 枚のオープンアドレス法ハッシュ表に詰める。問い合わせはハッシュ参照数回で済む。
//!
//! n-gram のキーは語 ID (24bit) を最大 4 つ詰めた u128 に次数を上位ビットで足したもの。

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

pub const MAX_ORDER: usize = 5;
const ID_BITS: u32 = 24;
const ID_MASK: u128 = (1 << ID_BITS) - 1;

pub const UNK: u32 = 1;
pub const BOS: u32 = 2;
pub const EOS: u32 = 3;

#[inline]
fn pack(ids: &[u32]) -> u128 {
    let mut k: u128 = 0;
    for &id in ids {
        k = (k << ID_BITS) | id as u128;
    }
    k | ((ids.len() as u128) << 124)
}

#[inline]
fn unpack_len(key: u128) -> usize {
    (key >> 124) as usize
}

#[inline]
fn strip_order(key: u128) -> u128 {
    key & !(0xFu128 << 124)
}

/// ハッシュ表の 1 枠。キーは n-gram (u128) の 64bit 指紋で持つ (0 は空き)。
/// 4,800 万 n-gram 規模でも指紋の衝突で別の n-gram を誤って引く確率は無視できる。
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Slot {
    key: u64,
    logp: f32,
    bow: f32,
}

#[inline]
fn fingerprint(key: u128) -> u64 {
    let lo = key as u64;
    let hi = (key >> 64) as u64;
    let mut h = hi ^ lo.wrapping_mul(0xC2B2_AE3D_27D4_EB4F).rotate_left(29);
    h ^= h >> 31;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 29;
    h | 1
}

/// 完成済みモデル。
pub struct Model {
    pub order: usize,
    vocab: FxHashMap<String, u32>,
    slots: Vec<Slot>,
    mask: usize,
}

#[inline]
fn hash(key: u128) -> u64 {
    let lo = key as u64;
    let hi = (key >> 64) as u64;
    let mut h = lo ^ hi.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h
}

impl Model {
    pub fn word_id(&self, w: &str) -> u32 {
        self.vocab.get(w).copied().unwrap_or(UNK)
    }

    pub fn vocab_len(&self) -> usize {
        self.vocab.len()
    }

    #[inline]
    fn get(&self, key: u128) -> Option<&Slot> {
        let fp = fingerprint(key);
        let mut i = hash(key) as usize & self.mask;
        loop {
            let s = &self.slots[i];
            if s.key == fp {
                return Some(s);
            }
            if s.key == 0 {
                return None;
            }
            i = (i + 1) & self.mask;
        }
    }

    /// log10 P(w | ctx)。ctx は直前の語 (末尾が直前)。必要な分だけ末尾から使う。
    pub fn logp(&self, ctx: &[u32], w: u32) -> f32 {
        let n_max = self.order.min(ctx.len() + 1);
        let mut buf = [0u32; MAX_ORDER];
        let mut bow = 0.0f32;
        for n in (1..=n_max).rev() {
            let h = &ctx[ctx.len() - (n - 1)..];
            buf[..n - 1].copy_from_slice(h);
            buf[n - 1] = w;
            if let Some(s) = self.get(pack(&buf[..n])) {
                return s.logp + bow;
            }
            if n > 1
                && let Some(s) = self.get(pack(h))
            {
                bow += s.bow;
            }
        }
        // 語彙外
        self.get(pack(&[UNK])).map(|s| s.logp).unwrap_or(-7.0) + bow
    }

    /// (ctx, w) について、モデルに実在する最長の n-gram の次数 (1..=order)。
    /// 0 は語彙外。「この並びはコーパスに出てこない」ことの判定に使う。
    pub fn match_order(&self, ctx: &[u32], w: u32) -> usize {
        let n_max = self.order.min(ctx.len() + 1);
        let mut buf = [0u32; MAX_ORDER];
        for n in (1..=n_max).rev() {
            buf[..n - 1].copy_from_slice(&ctx[ctx.len() - (n - 1)..]);
            buf[n - 1] = w;
            if w != UNK && self.get(pack(&buf[..n])).is_some() {
                return n;
            }
        }
        0
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(b"CELSOLM2")?;
        w.write_all(&(self.order as u32).to_le_bytes())?;
        let mut words: Vec<(&String, &u32)> = self.vocab.iter().collect();
        words.sort_by_key(|(_, id)| **id);
        w.write_all(&(words.len() as u32).to_le_bytes())?;
        for (word, id) in words {
            w.write_all(&id.to_le_bytes())?;
            w.write_all(&(word.len() as u16).to_le_bytes())?;
            w.write_all(word.as_bytes())?;
        }
        w.write_all(&(self.slots.len() as u64).to_le_bytes())?;
        // SAFETY: Slot は repr(C) の POD
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.slots.as_ptr() as *const u8,
                self.slots.len() * size_of::<Slot>(),
            )
        };
        w.write_all(bytes)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let mut r = BufReader::with_capacity(
            1 << 20,
            File::open(path).with_context(|| format!("{path:?}"))?,
        );
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != b"CELSOLM2" {
            bail!("not a celso model");
        }
        let mut b4 = [0u8; 4];
        let mut b2 = [0u8; 2];
        let mut b8 = [0u8; 8];
        r.read_exact(&mut b4)?;
        let order = u32::from_le_bytes(b4) as usize;
        r.read_exact(&mut b4)?;
        let nwords = u32::from_le_bytes(b4) as usize;
        let mut vocab = FxHashMap::default();
        vocab.reserve(nwords);
        for _ in 0..nwords {
            r.read_exact(&mut b4)?;
            let id = u32::from_le_bytes(b4);
            r.read_exact(&mut b2)?;
            let mut s = vec![0u8; u16::from_le_bytes(b2) as usize];
            r.read_exact(&mut s)?;
            vocab.insert(String::from_utf8(s)?, id);
        }
        r.read_exact(&mut b8)?;
        let nslots = u64::from_le_bytes(b8) as usize;
        let mut slots = vec![Slot::default(); nslots];
        // SAFETY: 同上
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                slots.as_mut_ptr() as *mut u8,
                nslots * size_of::<Slot>(),
            )
        };
        r.read_exact(bytes)?;
        Ok(Self {
            order,
            vocab,
            slots,
            mask: nslots - 1,
        })
    }
}

/// 構築パラメータ。
pub struct BuildConfig {
    pub order: usize,
    /// これ未満の出現回数の語は <unk> にする。
    pub min_word_count: u32,
    /// 次数ごとの足切り (index = 次数-1)。生の出現回数がこれ未満の n-gram はモデルに入れない。
    pub min_count: [u32; MAX_ORDER],
}

/// 分かち書き済みファイル (1 行 1 文, 空白区切り) からモデルを作る。
pub fn build(paths: &[impl AsRef<Path>], cfg: &BuildConfig) -> Result<Model> {
    assert!(cfg.order >= 1 && cfg.order <= MAX_ORDER);
    // --- 語彙 ---
    let mut wc: FxHashMap<String, u32> = FxHashMap::default();
    for p in paths {
        for line in BufReader::new(File::open(p.as_ref())?).lines() {
            for w in line?.split(' ').filter(|w| !w.is_empty()) {
                if let Some(c) = wc.get_mut(w) {
                    *c += 1;
                } else {
                    wc.insert(w.to_string(), 1);
                }
            }
        }
    }
    let mut words: Vec<(String, u32)> = wc
        .into_iter()
        .filter(|(_, c)| *c >= cfg.min_word_count)
        .collect();
    words.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    if words.len() as u128 + 4 > ID_MASK {
        bail!("vocab too large");
    }
    let mut vocab: FxHashMap<String, u32> = FxHashMap::default();
    vocab.insert("<unk>".into(), UNK);
    vocab.insert("<s>".into(), BOS);
    vocab.insert("</s>".into(), EOS);
    for (i, (w, _)) in words.into_iter().enumerate() {
        vocab.insert(w, i as u32 + 4);
    }
    eprintln!("vocab: {}", vocab.len());

    // --- コーパスを ID 列に (文頭 BOS / 文末 EOS 付き) ---
    let mut corpus: Vec<u32> = Vec::new();
    for p in paths {
        for line in BufReader::new(File::open(p.as_ref())?).lines() {
            let line = line?;
            corpus.push(BOS);
            for w in line.split(' ').filter(|w| !w.is_empty()) {
                corpus.push(vocab.get(w).copied().unwrap_or(UNK));
            }
            corpus.push(EOS);
        }
    }
    eprintln!("tokens (with markers): {}", corpus.len());

    // 文境界 (EOS の直後に BOS) をまたがない n-gram の開始位置か
    let valid_start = |i: usize, n: usize| -> bool {
        if i + n > corpus.len() {
            return false;
        }
        // 窓の内側 (末尾以外) に EOS があったら文をまたいでいる
        !corpus[i..i + n - 1].contains(&EOS)
    };

    // --- 次数ごとに数える ---
    // raw[n-1]: (key, raw count) を key 昇順で
    let mut raw: Vec<Vec<(u128, u32)>> = Vec::new();
    for n in 1..=cfg.order {
        let mut keys: Vec<u128> = (0..corpus.len())
            .into_par_iter()
            .filter(|&i| valid_start(i, n) && !(n == 1 && corpus[i] == BOS))
            .map(|i| pack(&corpus[i..i + n]))
            .collect();
        keys.par_sort_unstable();
        let mut rle: Vec<(u128, u32)> = Vec::new();
        for k in keys {
            match rle.last_mut() {
                Some((lk, c)) if *lk == k => *c += 1,
                _ => rle.push((k, 1)),
            }
        }
        eprintln!("order {n}: {} distinct", rle.len());
        raw.push(rle);
    }
    // BOS 単独 (文脈として必要)
    raw[0].push((pack(&[BOS]), 0));
    raw[0].sort_unstable_by_key(|e| e.0);

    // --- KN 用の調整済みカウント ---
    // 最高次と BOS 始まりは生カウント、それ以外は continuation count
    let mut adj: Vec<Vec<u32>> = raw
        .iter()
        .map(|t| t.iter().map(|e| e.1).collect())
        .collect();
    for n in 1..cfg.order {
        // 次数 n+1 の各 n-gram の接尾 (先頭語を落としたもの) を数える
        let mask: u128 = (1u128 << (ID_BITS as usize * n)) - 1;
        let mut suffixes: Vec<u128> = raw[n]
            .par_iter()
            .map(|(k, _)| strip_order(*k) & mask)
            .collect();
        suffixes.par_sort_unstable();
        let mut cont: FxHashMap<u128, u32> = FxHashMap::default();
        for s in suffixes {
            *cont.entry(s).or_insert(0) += 1;
        }
        let lower = &raw[n - 1];
        for (j, (k, _)) in lower.iter().enumerate() {
            let body = strip_order(*k);
            let first = (body >> (ID_BITS as usize * (n - 1))) as u32;
            if first == BOS {
                continue;
            }
            adj[n - 1][j] = cont.get(&body).copied().unwrap_or(0);
        }
    }

    // --- 次数ごとの割引 (Chen & Goodman の modified KN) ---
    let discounts: Vec<[f64; 3]> = adj
        .iter()
        .map(|a| {
            let mut nk = [0f64; 5];
            for &c in a {
                if (1..=4).contains(&c) {
                    nk[c as usize] += 1.0;
                }
            }
            if nk[1] == 0.0 || nk[2] == 0.0 || nk[3] == 0.0 {
                return [0.5, 1.0, 1.5];
            }
            let y = nk[1] / (nk[1] + 2.0 * nk[2]);
            let d1 = (1.0 - 2.0 * y * nk[2] / nk[1]).clamp(0.05, 0.95);
            let d2 = (2.0 - 3.0 * y * nk[3] / nk[2]).clamp(0.1, 1.9);
            let d3 = (3.0 - 4.0 * y * nk[4] / nk[3]).clamp(0.1, 2.9);
            [d1, d2, d3]
        })
        .collect();
    eprintln!("discounts: {discounts:?}");
    let disc = |n: usize, c: u32| -> f64 {
        match c {
            0 => 0.0,
            1 => discounts[n - 1][0],
            2 => discounts[n - 1][1],
            _ => discounts[n - 1][2],
        }
    };

    // --- 足切り後に残す n-gram ---
    let keep: Vec<Vec<bool>> = raw
        .iter()
        .enumerate()
        .map(|(i, t)| {
            t.iter()
                .map(|(k, c)| *c >= cfg.min_count[i] || (i == 0) || unpack_len(*k) == 0)
                .collect()
        })
        .collect();

    // --- 確率と backoff を下位から計算 ---
    let mut model = Model {
        order: cfg.order,
        vocab,
        slots: Vec::new(),
        mask: 0,
    };
    let total: usize = keep.iter().map(|k| k.iter().filter(|b| **b).count()).sum();
    // 線形探索で充填率 75% 程度までは問い合わせ速度がほぼ落ちない
    let cap = (total + total / 8).next_power_of_two().max(1024);
    model.slots = vec![Slot::default(); cap];
    model.mask = cap - 1;
    eprintln!("kept n-grams: {total}, slots: {cap}");

    let vocab_size = model.vocab.len() as f64;
    for n in 1..=cfg.order {
        let table = &raw[n - 1];
        let a = &adj[n - 1];
        // 文脈 (先頭 n-1 語) ごとの集計: 分母と、割引で浮いた質量
        let ctx_shift = ID_BITS as usize;
        let mut ctx_stats: FxHashMap<u128, (f64, f64)> = FxHashMap::default();
        for (j, (k, _)) in table.iter().enumerate() {
            let ctx = strip_order(*k) >> ctx_shift;
            let e = ctx_stats.entry(ctx).or_insert((0.0, 0.0));
            e.0 += a[j] as f64;
            e.1 += disc(n, a[j]);
        }
        // 確率 (並列に計算してから挿入)
        let probs: Vec<(u128, f32)> = table
            .par_iter()
            .enumerate()
            .filter(|(j, _)| keep[n - 1][*j])
            .map(|(j, (k, _))| {
                let body = strip_order(*k);
                let ctx = body >> ctx_shift;
                let w = (body & ID_MASK) as u32;
                let (den, freed) = ctx_stats[&ctx];
                let lower = if n == 1 {
                    1.0 / vocab_size
                } else {
                    let ids = unpack_ids(body, n);
                    10f64.powf(model.logp(&ids[1..n - 1], w) as f64)
                };
                let p = if den > 0.0 {
                    ((a[j] as f64 - disc(n, a[j])).max(0.0) + freed * lower) / den
                } else {
                    lower
                };
                (*k, p.max(1e-12).log10() as f32)
            })
            .collect();
        for (k, lp) in probs {
            model.insert(k, lp);
        }
        if n == 1 {
            let unk = pack(&[UNK]);
            if model.get(unk).is_none() {
                let (den, freed) = ctx_stats.get(&0).copied().unwrap_or((1.0, 1.0));
                model.insert(unk, ((freed / den) / vocab_size).log10() as f32);
            }
        }
        // 次数 n の文脈 (= 次数 n-1 の n-gram) に backoff 重みを付ける
        if n >= 2 {
            for (ctx, (den, freed)) in ctx_stats {
                let key = ctx | (((n - 1) as u128) << 124);
                if den > 0.0
                    && let Some(i) = model.find_index(key)
                {
                    model.slots[i].bow = (freed / den).max(1e-12).log10() as f32;
                }
            }
        }
        eprintln!("order {n}: probabilities done");
    }
    Ok(model)
}

fn unpack_ids(body: u128, n: usize) -> [u32; MAX_ORDER] {
    let mut out = [0u32; MAX_ORDER];
    for (i, o) in out.iter_mut().enumerate().take(n) {
        *o = ((body >> (ID_BITS as usize * (n - 1 - i))) & ID_MASK) as u32;
    }
    out
}

impl Model {
    fn find_index(&self, key: u128) -> Option<usize> {
        let fp = fingerprint(key);
        let mut i = hash(key) as usize & self.mask;
        loop {
            let s = &self.slots[i];
            if s.key == fp {
                return Some(i);
            }
            if s.key == 0 {
                return None;
            }
            i = (i + 1) & self.mask;
        }
    }

    fn insert(&mut self, key: u128, logp: f32) {
        let fp = fingerprint(key);
        let mut i = hash(key) as usize & self.mask;
        loop {
            let s = &mut self.slots[i];
            if s.key == 0 || s.key == fp {
                s.key = fp;
                s.logp = logp;
                return;
            }
            i = (i + 1) & self.mask;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_frequent_bigram() {
        let dir = std::env::temp_dir().join("celso-lm-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("c.txt");
        let mut f = File::create(&p).unwrap();
        for _ in 0..50 {
            writeln!(f, "宿泊 施設 が ある").unwrap();
            writeln!(f, "駅 から 施設 まで 歩く").unwrap();
            writeln!(f, "飲食 店 が ある").unwrap();
        }
        let m = build(
            &[&p],
            &BuildConfig {
                order: 3,
                min_word_count: 1,
                min_count: [1; MAX_ORDER],
            },
        )
        .unwrap();
        let id = |w| m.word_id(w);
        assert!(m.logp(&[BOS, id("宿泊")], id("施設")) > m.logp(&[BOS, id("宿泊")], id("から")));
        // 文末の確率がまともに引ける
        assert!(m.logp(&[id("が"), id("ある")], EOS) > -0.5);
    }
}
