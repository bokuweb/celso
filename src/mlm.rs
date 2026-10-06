//! 2 段目: マスク言語モデル (modernbert-ja) による採点し直し。
//!
//! n-gram は前後 3 語しか見ないので、「西口側[に]は … 多くある」のような離れた語との呼応や、
//! 文全体として自然かどうかを判断できない。1 段目で絞った候補だけについて、
//! 編集箇所の周りのサブワードを 1 つずつマスクした擬似対数尤度 (PLL) を比べる。
//!
//! 語彙 10 万の出力層が計算の大半を占めるので、マスクした位置の行だけを取り出して計算する。

use std::path::Path;

use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Module, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};
use candle_transformers::models::modernbert::{Config, ModernBert};
use tokenizers::Tokenizer;

const MASK_ID: u32 = 5;
const PAD_ID: u32 = 3;
/// 採点範囲の前後に残す文脈のサブワード数 (これより外は切り落とす)
const CONTEXT: usize = 16;
/// 1 回の forward に詰めるトークン数の上限 (B × L)
const BATCH_TOKENS: usize = 16384;

pub struct Mlm {
    model: ModernBert,
    head_dense: Linear,
    head_norm: LayerNorm,
    decoder: Linear,
    tok: Tokenizer,
    device: Device,
}

/// 採点要求: `text` の文字範囲 [start, end) の周り (前後 `margin` 文字) に掛かるサブワードを採点する。
pub struct Query<'a> {
    pub text: &'a str,
    pub start: usize,
    pub end: usize,
}

impl Mlm {
    /// HuggingFace 形式のディレクトリ (config.json / model.safetensors / tokenizer.json) を読む。
    pub fn load(dir: &Path) -> Result<Self> {
        // CELSO_DEVICE=metal で Apple GPU を使う (既定は CPU)
        let device = match std::env::var("CELSO_DEVICE").as_deref() {
            Ok("metal") => Device::new_metal(0)?,
            _ => Device::Cpu,
        };
        let config: Config =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dir.join("model.safetensors")],
                DType::F32,
                &device,
            )?
        };
        let model = ModernBert::load(vb.clone(), &config)?;
        let h = config.hidden_size;
        let head_dense = Linear::new(vb.get((h, h), "head.dense.weight")?, None);
        let head_norm =
            LayerNorm::new_no_bias(vb.get(h, "head.norm.weight")?, config.layer_norm_eps);
        // 出力層は埋め込みと重み共有
        let decoder = Linear::new(
            vb.get(
                (config.vocab_size, h),
                "model.embeddings.tok_embeddings.weight",
            )?,
            Some(vb.get(config.vocab_size, "decoder.bias")?),
        );
        let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("{e}"))?;
        Ok(Self {
            model,
            head_dense,
            head_norm,
            decoder,
            tok,
            device,
        })
    }

    /// 穴埋め採点: 範囲に掛かるサブワードを「まとめて」マスクした 1 本の系列で、
    /// 元のサブワードが入る対数確率の和 (自然対数) と、そのサブワード数を返す。
    ///
    /// PLL (1 つずつマスク) より粗いが、1 要求 1 系列で済み、さらにマスク後の系列が同じ要求
    /// (同じ長さの置換候補どうし) は 1 回の forward を共有するので桁違いに速い。
    pub fn fill_scores(&self, queries: &[Query], margin: usize) -> Result<Vec<(f32, usize)>> {
        use std::collections::HashMap;
        // マスク後の系列 → 系列番号
        let mut uniq: HashMap<Vec<u32>, usize> = HashMap::new();
        let mut seqs: Vec<Vec<u32>> = Vec::new();
        // 要求ごとの (系列番号, [(位置, 正解 ID)])
        type Plan = Option<(usize, Vec<(usize, u32)>)>;
        let mut plan: Vec<Plan> = Vec::with_capacity(queries.len());
        for q in queries {
            let enc = self
                .tok
                .encode_char_offsets(q.text, true)
                .map_err(|e| anyhow!("{e}"))?;
            let ids = enc.get_ids();
            let offs = enc.get_offsets();
            let special = enc.get_special_tokens_mask();
            let lo = q.start.saturating_sub(margin);
            let hi = q.end + margin;
            let win: Vec<usize> = (0..ids.len())
                .filter(|&i| {
                    special[i] == 0 && offs[i].0 < hi.max(lo + 1) && lo < offs[i].1 && hi > lo
                })
                .collect();
            if win.is_empty() {
                plan.push(None);
                continue;
            }
            let c0 = win[0].saturating_sub(CONTEXT);
            let c1 = (win[win.len() - 1] + 1 + CONTEXT).min(ids.len());
            let mut s = ids[c0..c1].to_vec();
            let mut tg = Vec::with_capacity(win.len());
            for &i in &win {
                s[i - c0] = MASK_ID;
                tg.push((i - c0, ids[i]));
            }
            let n = seqs.len();
            let si = *uniq.entry(s.clone()).or_insert_with(|| {
                seqs.push(s);
                n
            });
            plan.push(Some((si, tg)));
        }
        // 系列ごとに必要な (位置, ID) をまとめて計算
        let mut need: Vec<Vec<(usize, u32)>> = vec![Vec::new(); seqs.len()];
        for (si, tg) in plan.iter().flatten() {
            for t in tg {
                if !need[*si].contains(t) {
                    need[*si].push(*t);
                }
            }
        }
        let mut got: HashMap<(usize, usize, u32), f32> = HashMap::new();
        let mut order: Vec<usize> = (0..seqs.len()).collect();
        order.sort_by_key(|&i| seqs[i].len());
        let mut start = 0;
        while start < order.len() {
            let mut end = start + 1;
            while end < order.len() && seqs[order[end]].len() * (end - start + 1) <= BATCH_TOKENS {
                end += 1;
            }
            let chunk = &order[start..end];
            let lp = self.forward_positions(&seqs, chunk, &need)?;
            for ((si, pos, id), v) in lp {
                got.insert((si, pos, id), v);
            }
            start = end;
        }
        if std::env::var_os("CELSO_DEBUG").is_some() {
            eprintln!(
                "mlm fill: {} queries -> {} sequences",
                queries.len(),
                seqs.len()
            );
        }
        Ok(plan
            .iter()
            .map(|p| match p {
                None => (0.0, 0),
                Some((si, tg)) => (
                    tg.iter().map(|(pos, id)| got[&(*si, *pos, *id)]).sum(),
                    tg.len(),
                ),
            })
            .collect())
    }

    /// seqs[chunk] を 1 バッチで流し、need で指定した (位置, ID) の対数確率を返す。
    #[allow(clippy::type_complexity)]
    fn forward_positions(
        &self,
        seqs: &[Vec<u32>],
        chunk: &[usize],
        need: &[Vec<(usize, u32)>],
    ) -> Result<Vec<((usize, usize, u32), f32)>> {
        let max_len = chunk.iter().map(|&i| seqs[i].len()).max().unwrap();
        let b = chunk.len();
        let mut flat = Vec::with_capacity(b * max_len);
        let mut mask = Vec::with_capacity(b * max_len);
        for &i in chunk {
            let s = &seqs[i];
            flat.extend_from_slice(s);
            flat.extend(std::iter::repeat_n(PAD_ID, max_len - s.len()));
            mask.extend(std::iter::repeat_n(1u32, s.len()));
            mask.extend(std::iter::repeat_n(0u32, max_len - s.len()));
        }
        let input = Tensor::from_vec(flat, (b, max_len), &self.device)?;
        let attn = Tensor::from_vec(mask, (b, max_len), &self.device)?;
        let hidden = self.model.forward(&input, &attn)?;
        let h = hidden.dim(2)?;
        let hidden = hidden.reshape((b * max_len, h))?;
        let mut keys = Vec::new();
        let mut rows = Vec::new();
        for (r, &si) in chunk.iter().enumerate() {
            for &(pos, id) in &need[si] {
                keys.push((si, pos, id));
                rows.push((r * max_len + pos) as u32);
            }
        }
        let n = rows.len();
        let rows = Tensor::from_vec(rows, n, &self.device)?;
        let picked = hidden.index_select(&rows, 0)?;
        let x = self.head_dense.forward(&picked)?.gelu_erf()?;
        let x = self.head_norm.forward(&x)?;
        let logits = self.decoder.forward(&x)?;
        let logp = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
        let tgt: Vec<u32> = keys.iter().map(|k| k.2).collect();
        let tgt = Tensor::from_vec(tgt, (n, 1), &self.device)?;
        let v: Vec<f32> = logp.gather(&tgt, 1)?.i((.., 0))?.to_vec1()?;
        Ok(keys.into_iter().zip(v).collect())
    }

    /// 各要求について、範囲に掛かるサブワードの PLL の和 (自然対数) と、採点したサブワード数を返す。
    pub fn window_pll(&self, queries: &[Query], margin: usize) -> Result<Vec<(f32, usize)>> {
        // (系列, 要求番号, マスク位置, 正解 ID)
        let mut seqs: Vec<Vec<u32>> = Vec::new();
        let mut targets: Vec<(usize, usize, usize, u32)> = Vec::new();
        for (qi, q) in queries.iter().enumerate() {
            let enc = self
                .tok
                .encode_char_offsets(q.text, true)
                .map_err(|e| anyhow!("{e}"))?;
            let ids = enc.get_ids();
            let offs = enc.get_offsets();
            let special = enc.get_special_tokens_mask();
            let lo = q.start.saturating_sub(margin);
            let hi = q.end + margin;
            let win: Vec<usize> = (0..ids.len())
                .filter(|&i| special[i] == 0 && offs[i].0 < hi.max(lo + 1) && lo < offs[i].1)
                .collect();
            if win.is_empty() {
                continue;
            }
            // 採点範囲の前後 CONTEXT サブワードだけ残す (計算量は系列長に比例するため)
            let c0 = win[0].saturating_sub(CONTEXT);
            let c1 = (win[win.len() - 1] + 1 + CONTEXT).min(ids.len());
            for &i in &win {
                if i < c0 || i >= c1 {
                    continue;
                }
                let mut s = ids[c0..c1].to_vec();
                s[i - c0] = MASK_ID;
                targets.push((seqs.len(), qi, i - c0, ids[i]));
                seqs.push(s);
            }
        }
        let mut out = vec![(0f32, 0usize); queries.len()];
        // 長さの近い系列をまとめてパディングの無駄を減らす
        let mut order: Vec<usize> = (0..seqs.len()).collect();
        order.sort_by_key(|&i| seqs[i].len());
        let mut start = 0;
        while start < order.len() {
            let mut end = start + 1;
            while end < order.len() && seqs[order[end]].len() * (end - start + 1) <= BATCH_TOKENS {
                end += 1;
            }
            let chunk = &order[start..end];
            let t = std::time::Instant::now();
            let got = self.forward_chunk(&seqs, chunk, &targets)?;
            if std::env::var_os("CELSO_DEBUG").is_some() {
                eprintln!(
                    "mlm chunk: {} seqs x {} tokens in {:.2?}",
                    chunk.len(),
                    seqs[chunk[chunk.len() - 1]].len(),
                    t.elapsed()
                );
            }
            for (&si, lp) in chunk.iter().zip(got) {
                let qi = targets[si].1;
                out[qi].0 += lp;
                out[qi].1 += 1;
            }
            start = end;
        }
        Ok(out)
    }

    /// seqs[chunk] を 1 バッチで流し、各系列のマスク位置での正解の対数確率を返す。
    fn forward_chunk(
        &self,
        seqs: &[Vec<u32>],
        chunk: &[usize],
        targets: &[(usize, usize, usize, u32)],
    ) -> Result<Vec<f32>> {
        let max_len = chunk.iter().map(|&i| seqs[i].len()).max().unwrap();
        let b = chunk.len();
        let mut flat = Vec::with_capacity(b * max_len);
        let mut mask = Vec::with_capacity(b * max_len);
        for &i in chunk {
            let s = &seqs[i];
            flat.extend_from_slice(s);
            flat.extend(std::iter::repeat_n(PAD_ID, max_len - s.len()));
            mask.extend(std::iter::repeat_n(1u32, s.len()));
            mask.extend(std::iter::repeat_n(0u32, max_len - s.len()));
        }
        let input = Tensor::from_vec(flat, (b, max_len), &self.device)?;
        let attn = Tensor::from_vec(mask, (b, max_len), &self.device)?;
        let hidden = self.model.forward(&input, &attn)?; // [B, L, H]
        let h = hidden.dim(2)?;
        let hidden = hidden.reshape((b * max_len, h))?;
        let rows: Vec<u32> = chunk
            .iter()
            .enumerate()
            .map(|(r, &i)| (r * max_len + targets[i].2) as u32)
            .collect();
        let rows = Tensor::from_vec(rows, b, &self.device)?;
        let picked = hidden.index_select(&rows, 0)?; // [B, H]
        let x = self.head_dense.forward(&picked)?.gelu_erf()?;
        let x = self.head_norm.forward(&x)?;
        let logits = self.decoder.forward(&x)?; // [B, V]
        let logp = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
        let tgt: Vec<u32> = chunk.iter().map(|&i| targets[i].3).collect();
        let tgt = Tensor::from_vec(tgt, (b, 1), &self.device)?;
        Ok(logp.gather(&tgt, 1)?.i((.., 0))?.to_vec1()?)
    }
}
