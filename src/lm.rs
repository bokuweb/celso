//! 単語 n-gram 言語モデル (補間 modified Kneser-Ney)。
//!
//! * 構築: 分かち書き済みコーパスから次数ごとに n-gram を列挙 → ソート → ランレングスで数える。
//!   KN の低次は「左に何種類の語が来たか」(continuation count) を使う。
//! * 保存形式: ARPA と同じく「補間済み確率 + backoff 重み」を n-gram ごとに持たせ、
//!   全次数を 1 枚のオープンアドレス法ハッシュ表に詰める。問い合わせはハッシュ参照数回で済む。
//!
//! n-gram のキーは語 ID (24bit) を最大 4 つ詰めた u128 に次数を上位ビットで足したもの。

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
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
pub(crate) fn pack(ids: &[u32]) -> u128 {
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

/// 構築時のハッシュ表の 1 枠。`h` は n-gram (u128) の 64bit ハッシュ (0 は空き)。
/// 位置は `h` の上位ビット ([`slot_index`])、配布形式の指紋は `h` の下位ビットから取る。
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Slot {
    h: u64,
    logp: f32,
    bow: f32,
}

#[inline]
pub(crate) fn key_hash(key: u128) -> u64 {
    let lo = key as u64;
    let hi = (key >> 64) as u64;
    let mut h = hi ^ lo.wrapping_mul(0xC2B2_AE3D_27D4_EB4F).rotate_left(29);
    h ^= h >> 31;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 29;
    h = h.wrapping_add(lo.rotate_left(17) ^ hi);
    h ^= h >> 32;
    h = h.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    h ^= h >> 32;
    h | 1
}

/// 容量を 2 のべき乗に揃えずに済むよう、`h * cap` の上位 64 bit を位置にする (fastrange)。
#[inline]
fn slot_index(h: u64, cap: usize) -> usize {
    ((u128::from(h) * cap as u128) >> 64) as usize
}

/// 配布用の 1 枠 (4 バイト): 16bit 指紋 + 8bit 確率 + 8bit backoff。
///
/// 構築時の [`Slot`] と同じ位置に並べるので、線形探索の到達順も変わらない。
/// 指紋の取り違えは 1 回の探索あたり数万分の 1 程度で、未出現の並びをまれに「見たことがある」と
/// 誤るだけ (誤検出が減る方向) なので許容する。
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct QSlot {
    fp: u16,
    logp: u8,
    bow: u8,
}

#[inline]
fn fp16(h: u64) -> u16 {
    (((h >> 1) & 0xFFFF) as u16).max(1)
}

/// 量子化の範囲: log10 で [-QRANGE, 0] を 8bit (1 段 0.047) で表す。
/// 確率は構築時に 1e-12 で下限を切っているので収まる。Δ の閾値 (2〜4) に比べて誤差は十分小さい。
const QRANGE: f32 = 12.0;

#[inline]
fn quantize(v: f32) -> u8 {
    ((-v).clamp(0.0, QRANGE) / QRANGE * 255.0).round() as u8
}

#[inline]
fn dequantize(q: u8) -> f32 {
    -f32::from(q) * (QRANGE / 255.0)
}

enum Table {
    /// 構築中 (f32 のまま)
    Build(Vec<Slot>),
    /// 配布形式を mmap したもの (起動が速く、複数プロセスでページを共有できる)
    Mapped {
        map: memmap2::Mmap,
        offset: usize,
        len: usize,
    },
}

impl Table {
    fn len(&self) -> usize {
        match self {
            Table::Build(v) => v.len(),
            Table::Mapped { len, .. } => *len,
        }
    }

    #[inline]
    fn qslots(&self) -> &[QSlot] {
        match self {
            Table::Mapped { map, offset, len } => {
                // SAFETY: offset は 4 バイト境界に揃えて書き出しており、mmap の先頭はページ境界。
                // QSlot は repr(C) の POD で、ファイルは読み取り専用で開いている。
                unsafe {
                    std::slice::from_raw_parts(map.as_ptr().add(*offset).cast::<QSlot>(), *len)
                }
            }
            Table::Build(_) => &[],
        }
    }
}

/// チェッカーから見た言語モデル。確率モデル ([`Model`]) と、n-gram の有無だけを持つ
/// 軽量版 ([`crate::ngset::NgramSet`]) を差し替えられるようにする。
pub trait LanguageModel: Send + Sync {
    fn order(&self) -> usize;
    fn word_id(&self, w: &str) -> u32;
    /// log10 P(w | ctx) (軽量版では n-gram の有無から作る擬似スコア)
    fn logp(&self, ctx: &[u32], w: u32) -> f32;
    /// (ctx, w) について、実在する最長の n-gram の次数。0 は語彙外。
    fn match_order(&self, ctx: &[u32], w: u32) -> usize;
    fn vocab_len(&self) -> usize;
    /// 語彙の一覧 (語, ID)。別モデルとの ID の対応づけに使う。
    fn words(&self) -> Vec<(&str, u32)>;

    /// トークンの語 ID。語彙に無ければ品詞クラスの ID (学習時に語彙外を品詞クラスへ置き換えている)。
    fn token_id(&self, t: &crate::tokenize::Token) -> u32 {
        let id = self.word_id(t.key());
        if id == UNK {
            self.word_id(&t.class_key())
        } else {
            id
        }
    }
}

impl LanguageModel for Model {
    fn order(&self) -> usize {
        self.order
    }
    fn word_id(&self, w: &str) -> u32 {
        Model::word_id(self, w)
    }
    fn logp(&self, ctx: &[u32], w: u32) -> f32 {
        Model::logp(self, ctx, w)
    }
    fn match_order(&self, ctx: &[u32], w: u32) -> usize {
        Model::match_order(self, ctx, w)
    }
    fn vocab_len(&self) -> usize {
        Model::vocab_len(self)
    }
    fn words(&self) -> Vec<(&str, u32)> {
        self.vocab.iter().map(|(w, id)| (w.as_str(), *id)).collect()
    }
}

/// ファイル先頭の識別子を見て、確率モデルか軽量版かを開く。
pub fn load_any(path: &Path) -> Result<Box<dyn LanguageModel>> {
    let mut magic = [0u8; 8];
    std::io::Read::read_exact(&mut File::open(path)?, &mut magic)?;
    if &magic == b"CELSONS1" {
        Ok(Box::new(crate::ngset::NgramSet::load(path)?))
    } else {
        Ok(Box::new(Model::load(path)?))
    }
}

/// 完成済みモデル。
pub struct Model {
    pub order: usize,
    vocab: FxHashMap<String, u32>,
    table: Table,
    cap: usize,
}

impl Model {
    pub fn word_id(&self, w: &str) -> u32 {
        self.vocab.get(w).copied().unwrap_or(UNK)
    }

    pub fn vocab_len(&self) -> usize {
        self.vocab.len()
    }

    /// n-gram の (log10 確率, log10 backoff) を引く。
    #[inline]
    fn get(&self, key: u128) -> Option<(f32, f32)> {
        let h = key_hash(key);
        let cap = self.cap;
        let mut i = slot_index(h, cap);
        if let Table::Build(slots) = &self.table {
            loop {
                let s = &slots[i];
                if s.h == h {
                    return Some((s.logp, s.bow));
                }
                if s.h == 0 {
                    return None;
                }
                i += 1;
                if i == cap {
                    i = 0;
                }
            }
        }
        let slots = self.table.qslots();
        let fp = fp16(h);
        loop {
            let s = &slots[i];
            if s.fp == fp {
                return Some((dequantize(s.logp), dequantize(s.bow)));
            }
            if s.fp == 0 {
                return None;
            }
            i += 1;
            if i == cap {
                i = 0;
            }
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
            if let Some((lp, _)) = self.get(pack(&buf[..n])) {
                return lp + bow;
            }
            if n > 1
                && let Some((_, b)) = self.get(pack(h))
            {
                bow += b;
            }
        }
        // 語彙外
        self.get(pack(&[UNK])).map(|s| s.0).unwrap_or(-7.0) + bow
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

    /// 配布形式 (CELSOLM4: 1 枠 4 バイト) で保存する。
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        let pos = std::cell::Cell::new(0usize);
        let put = |w: &mut BufWriter<File>, b: &[u8]| -> Result<()> {
            w.write_all(b)?;
            pos.set(pos.get() + b.len());
            Ok(())
        };
        put(&mut w, b"CELSOLM4")?;
        put(&mut w, &(self.order as u32).to_le_bytes())?;
        let mut words: Vec<(&String, &u32)> = self.vocab.iter().collect();
        words.sort_by_key(|(_, id)| **id);
        put(&mut w, &(words.len() as u32).to_le_bytes())?;
        for (word, id) in words {
            put(&mut w, &id.to_le_bytes())?;
            put(&mut w, &(word.len() as u16).to_le_bytes())?;
            put(&mut w, word.as_bytes())?;
        }
        put(&mut w, &(self.table.len() as u64).to_le_bytes())?;
        // 表本体を 4 バイト境界に揃える (mmap してそのまま &[QSlot] として読むため)
        let pad = (4 - pos.get() % 4) % 4;
        put(&mut w, &[0u8; 4][..pad])?;
        let q: Vec<QSlot>;
        let slots: &[QSlot] = match &self.table {
            Table::Build(v) => {
                q = v
                    .iter()
                    .map(|s| {
                        if s.h == 0 {
                            QSlot::default()
                        } else {
                            QSlot {
                                fp: fp16(s.h),
                                logp: quantize(s.logp),
                                bow: quantize(s.bow),
                            }
                        }
                    })
                    .collect();
                &q
            }
            Table::Mapped { .. } => self.table.qslots(),
        };
        // SAFETY: QSlot は repr(C) の POD
        let bytes =
            unsafe { std::slice::from_raw_parts(slots.as_ptr().cast::<u8>(), size_of_val(slots)) };
        put(&mut w, bytes)?;
        Ok(())
    }

    /// 配布形式 (CELSOLM4) を mmap で開く (読み込みはほぼ一瞬)。
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("{}", path.display()))?;
        // SAFETY: 読み取り専用で開いたモデルファイルを mmap する。実行中に書き換えないこと。
        let map = unsafe { memmap2::Mmap::map(&file)? };
        let pos = std::cell::Cell::new(0usize);
        let bytes: &[u8] = &map;
        let take = |n: usize| -> Result<&[u8]> {
            let p = pos.get();
            if bytes.len() < p + n {
                bail!("truncated model file");
            }
            pos.set(p + n);
            Ok(&bytes[p..p + n])
        };
        if take(8)? != b"CELSOLM4" {
            bail!("not a celso model (CELSOLM4)");
        }
        let order = u32::from_le_bytes(take(4)?.try_into()?) as usize;
        let nwords = u32::from_le_bytes(take(4)?.try_into()?) as usize;
        let mut vocab = FxHashMap::default();
        vocab.reserve(nwords);
        for _ in 0..nwords {
            let id = u32::from_le_bytes(take(4)?.try_into()?);
            let len = u16::from_le_bytes(take(2)?.try_into()?) as usize;
            vocab.insert(std::str::from_utf8(take(len)?)?.to_string(), id);
        }
        let nslots = u64::from_le_bytes(take(8)?.try_into()?) as usize;
        let consumed = pos.get();
        let offset = consumed + (4 - consumed % 4) % 4;
        if offset + nslots * size_of::<QSlot>() > map.len() {
            bail!("truncated model file");
        }
        Ok(Self {
            order,
            vocab,
            table: Table::Mapped {
                map,
                offset,
                len: nslots,
            },
            cap: nslots,
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
    /// 語彙。コーパスのトークンが「表層形\x1f品詞クラス」のとき、語彙外の表層形は品詞クラスにする。
    pub vocab: Option<rustc_hash::FxHashSet<String>>,
}

/// コーパスの 1 トークンを、語彙に応じて表層形か品詞クラスにする。
pub(crate) fn corpus_key<'a>(w: &'a str, vocab: Option<&rustc_hash::FxHashSet<String>>) -> &'a str {
    match w.split_once('\x1f') {
        Some((surface, class)) => match vocab {
            Some(v) if !v.contains(surface) => class,
            _ => surface,
        },
        None => w,
    }
}

/// 分かち書き済みファイル (1 行 1 文, 空白区切り) からモデルを作る。
pub fn build(paths: &[impl AsRef<Path>], cfg: &BuildConfig) -> Result<Model> {
    assert!(cfg.order >= 1 && cfg.order <= MAX_ORDER);
    // --- 語彙 ---
    let mut wc: FxHashMap<String, u32> = FxHashMap::default();
    for p in paths {
        for line in BufReader::new(File::open(p.as_ref())?).lines() {
            for w in line?.split(' ').filter(|w| !w.is_empty()) {
                let w = corpus_key(w, cfg.vocab.as_ref());
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
                let w = corpus_key(w, cfg.vocab.as_ref());
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
        table: Table::Build(Vec::new()),
        cap: 0,
    };
    let total: usize = keep.iter().map(|k| k.iter().filter(|b| **b).count()).sum();
    // 充填率 80%。表は小さい (キャッシュに乗りやすい) ので、線形探索が多少伸びても速度はほぼ落ちない
    let cap = (total * 5 / 4).max(1024);
    model.table = Table::Build(vec![Slot::default(); cap]);
    model.cap = cap;
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
                    model.build_slots_mut()[i].bow = (freed / den).max(1e-12).log10() as f32;
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
    fn build_slots_mut(&mut self) -> &mut Vec<Slot> {
        match &mut self.table {
            Table::Build(v) => v,
            _ => panic!("model is not in build mode"),
        }
    }

    fn find_index(&self, key: u128) -> Option<usize> {
        let Table::Build(slots) = &self.table else {
            return None;
        };
        let h = key_hash(key);
        let mut i = slot_index(h, self.cap);
        loop {
            let s = &slots[i];
            if s.h == h {
                return Some(i);
            }
            if s.h == 0 {
                return None;
            }
            i += 1;
            if i == self.cap {
                i = 0;
            }
        }
    }

    fn insert(&mut self, key: u128, logp: f32) {
        let h = key_hash(key);
        let cap = self.cap;
        let slots = self.build_slots_mut();
        let mut i = slot_index(h, cap);
        loop {
            let s = &mut slots[i];
            if s.h == 0 || s.h == h {
                s.h = h;
                s.logp = logp;
                return;
            }
            i += 1;
            if i == cap {
                i = 0;
            }
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
                vocab: None,
            },
        )
        .unwrap();
        let id = |w| m.word_id(w);
        assert!(m.logp(&[BOS, id("宿泊")], id("施設")) > m.logp(&[BOS, id("宿泊")], id("から")));
        // 文末の確率がまともに引ける
        assert!(m.logp(&[id("が"), id("ある")], EOS) > -0.5);
        // 量子化して保存 → mmap で読み直しても確率はほぼ変わらない
        let mp = dir.join("m.bin");
        m.save(&mp).unwrap();
        let q = Model::load(&mp).unwrap();
        for (ctx, w) in [
            (vec![BOS, id("宿泊")], id("施設")),
            (vec![id("駅"), id("から")], id("施設")),
        ] {
            assert!((m.logp(&ctx, w) - q.logp(&ctx, w)).abs() < 0.1);
        }
    }
}
