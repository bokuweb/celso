//! 修正候補の採否を決める判定器 (ロジスティック回帰)。
//!
//! 種類ごとの閾値 1 つで n-gram の Δ を切るだけでは、「の」の削除のように正しい文でも Δ が出やすい
//! 言い換えと、本当の誤りを分けにくい。そこで、Δ に加えて助詞の種類・前後の品詞・文法モデルの Δ・
//! 共起の差・文書の種類などを特徴量にして、誤りである確率 (の対数オッズ) を出す。
//!
//! 特徴量は「名前 → 値」の疎な表現で、重みは名前ごとに持つ (未知の特徴量は 0 として無視する)。
//! 学習は [`train`] (L2 正則化つきの Adagrad)。重みは TSV で保存する (数 KB〜数百 KB)。

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use rustc_hash::FxHashMap;

use crate::checker::Domain;

/// 特徴量 (名前, 値)。
pub type Features = Vec<(String, f32)>;

/// 特徴量の名前のハッシュ。
#[inline]
#[must_use]
pub fn key(name: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    name.hash(&mut h);
    h.finish()
}

pub struct Reranker {
    weights: FxHashMap<String, f32>,
    /// 名前のハッシュ → 重み (検査時は名前の文字列を作らずに引くため)
    hashed: FxHashMap<u64, f32>,
    /// 文書の種類ごとの採用の閾値 (対数オッズ)。[法令文, 一般文, 契約書]
    pub tau: [f32; 3],
    /// この n-gram Δ に届かない候補は判定器にかけない (速度のため)
    pub floor: f32,
    /// 種類ごとの閾値の上書き (種類名 → [法令文, 一般文, 契約書])
    pub tau_kind: FxHashMap<String, [f32; 3]>,
    /// 判定器を使わず、種類ごとの閾値で決める種類 (活用の誤りなど)
    pub exempt: Vec<String>,
}

impl Reranker {
    #[must_use]
    pub fn new(weights: FxHashMap<String, f32>) -> Self {
        let hashed = weights.iter().map(|(k, v)| (key(k), *v)).collect();
        Self {
            weights,
            hashed,
            tau: [0.0; 3],
            floor: 1.0,
            tau_kind: FxHashMap::default(),
            exempt: Vec::new(),
        }
    }

    /// TSV (名前 \t 重み) を読む。先頭の `#tau` / `#floor` / `#tau_kind` / `#exempt` 行は設定として読む。
    pub fn from_tsv(text: &str) -> Result<Self> {
        let mut r = Self::new(FxHashMap::default());
        for line in text.lines() {
            let mut it = line.split('\t');
            let (Some(k), Some(v)) = (it.next(), it.next()) else {
                continue;
            };
            match k {
                "#tau" => {
                    for (i, x) in v.split(',').enumerate().take(3) {
                        r.tau[i] = x.trim().parse()?;
                    }
                }
                "#floor" => r.floor = v.trim().parse()?,
                // #tau_kind \t 種類 \t 法令文,一般文,契約書
                "#tau_kind" => {
                    let t = it.next().unwrap_or("");
                    let mut a = r.tau;
                    for (i, x) in t.split(',').enumerate().take(3) {
                        a[i] = x.trim().parse()?;
                    }
                    r.tau_kind.insert(v.trim().to_string(), a);
                }
                "#exempt" => r.exempt.extend(v.split(',').map(|x| x.trim().to_string())),
                _ => {
                    let w: f32 = v.trim().parse()?;
                    r.hashed.insert(key(k), w);
                    r.weights.insert(k.to_string(), w);
                }
            }
        }
        Ok(r)
    }

    pub fn load(path: &Path) -> Result<Self> {
        Self::from_tsv(
            &std::fs::read_to_string(path).with_context(|| format!("{}", path.display()))?,
        )
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        writeln!(w, "#tau\t{},{},{}", self.tau[0], self.tau[1], self.tau[2])?;
        writeln!(w, "#floor\t{}", self.floor)?;
        if !self.exempt.is_empty() {
            writeln!(w, "#exempt\t{}", self.exempt.join(","))?;
        }
        let mut kinds: Vec<_> = self.tau_kind.iter().collect();
        kinds.sort_by(|a, b| a.0.cmp(b.0));
        for (k, t) in kinds {
            writeln!(w, "#tau_kind\t{k}\t{},{},{}", t[0], t[1], t[2])?;
        }
        let mut v: Vec<(&String, &f32)> = self.weights.iter().collect();
        v.sort_by(|a, b| a.0.cmp(b.0));
        for (k, x) in v {
            writeln!(w, "{k}\t{x}")?;
        }
        Ok(())
    }

    /// 誤りである対数オッズ。
    #[must_use]
    pub fn score(&self, feats: &[(String, f32)]) -> f32 {
        feats.iter().map(|(k, v)| self.weight(k) * v).sum()
    }

    /// 特徴量 1 つの重み (未知の特徴量は 0)。
    #[inline]
    #[must_use]
    pub fn weight(&self, name: &str) -> f32 {
        self.hashed.get(&key(name)).copied().unwrap_or(0.0)
    }

    #[must_use]
    pub fn tau(&self, d: Domain) -> f32 {
        Self::pick(&self.tau, d)
    }

    /// 種類ごとの上書きがあればそれを、無ければ文書種類ごとの閾値を返す。
    #[must_use]
    pub fn tau_for(&self, kind: &str, d: Domain) -> f32 {
        self.tau_kind
            .get(kind)
            .map_or_else(|| self.tau(d), |t| Self::pick(t, d))
    }

    /// 判定器を使わない種類か。
    #[must_use]
    pub fn is_exempt(&self, kind: &str) -> bool {
        self.exempt.iter().any(|k| k == kind)
    }

    fn pick(t: &[f32; 3], d: Domain) -> f32 {
        match d {
            Domain::Legal => t[0],
            Domain::General => t[1],
            Domain::Contract => t[2],
        }
    }
}

/// 学習データ 1 件 (正例なら label = true)。`weight` は例ごとの重み (分野の釣り合いを取るため)。
pub struct Example {
    pub label: bool,
    pub weight: f32,
    pub feats: Features,
}

/// 学習用に特徴量を番号にした 1 件 (正解ラベル, 例の重み, (特徴量の番号, 値))。
type Row = (f32, f32, Vec<(usize, f32)>);

/// L2 正則化つきロジスティック回帰を Adagrad で学習する。
/// `min_count` 回未満しか出ない特徴量は捨てる (過学習を防ぎ、重みの表を小さくする)。
#[must_use]
pub fn train(
    data: &[Example],
    epochs: usize,
    lr: f32,
    l2: f32,
    min_count: usize,
) -> FxHashMap<String, f32> {
    let mut df: FxHashMap<&str, usize> = FxHashMap::default();
    for e in data {
        for (k, _) in &e.feats {
            *df.entry(k.as_str()).or_default() += 1;
        }
    }
    let mut index: FxHashMap<&str, usize> = FxHashMap::default();
    for (k, c) in &df {
        if *c >= min_count {
            let n = index.len();
            index.insert(k, n);
        }
    }
    let rows: Vec<Row> = data
        .iter()
        .map(|e| {
            (
                if e.label { 1.0 } else { 0.0 },
                e.weight,
                e.feats
                    .iter()
                    .filter_map(|(k, v)| index.get(k.as_str()).map(|&i| (i, *v)))
                    .collect(),
            )
        })
        .collect();
    let mut w = vec![0.0f32; index.len()];
    let mut g2 = vec![1e-8f32; index.len()];
    // 決定的に回すため、並びは固定の擬似乱数で混ぜる
    let mut order: Vec<usize> = (0..rows.len()).collect();
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..epochs {
        for i in (1..order.len()).rev() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            order.swap(i, (seed % (i as u64 + 1)) as usize);
        }
        for &ri in &order {
            let (y, wt, x) = &rows[ri];
            let z: f32 = x.iter().map(|(i, v)| w[*i] * v).sum();
            let p = 1.0 / (1.0 + (-z).exp());
            let g = (p - y) * wt;
            for (i, v) in x {
                let grad = g * v + l2 * w[*i];
                g2[*i] += grad * grad;
                w[*i] -= lr * grad / g2[*i].sqrt();
            }
        }
    }
    index
        .into_iter()
        .filter(|(_, i)| w[*i] != 0.0)
        .map(|(k, i)| (k.to_string(), w[i]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(label: bool, d: f32, tag: &str) -> Example {
        Example {
            label,
            weight: 1.0,
            feats: vec![("b".into(), 1.0), ("d".into(), d), (tag.into(), 1.0)],
        }
    }

    #[test]
    fn learns_that_a_tag_marks_false_positives() {
        // Δ が同じでも、tag=no の候補は誤り (正例) でないことを学ぶ
        let mut data = Vec::new();
        for _ in 0..200 {
            data.push(ex(true, 5.0, "t=yes"));
            data.push(ex(false, 5.0, "t=no"));
            data.push(ex(false, 0.5, "t=yes"));
        }
        let r = Reranker::new(train(&data, 20, 0.5, 1e-4, 1));
        let pos = r.score(&ex(true, 5.0, "t=yes").feats);
        let neg = r.score(&ex(false, 5.0, "t=no").feats);
        assert!(pos > 0.0 && neg < 0.0, "pos={pos} neg={neg}");
    }

    #[test]
    fn round_trips_through_tsv() {
        let mut w = FxHashMap::default();
        w.insert("k=delete".to_string(), 1.5);
        let mut r = Reranker::new(w);
        r.tau = [0.5, 1.0, 2.0];
        r.floor = 1.5;
        let path = std::env::temp_dir().join(format!("celso-rerank-{}.tsv", std::process::id()));
        r.save(&path).unwrap();
        let back = Reranker::load(&path).unwrap();
        assert_eq!(back.tau, [0.5, 1.0, 2.0]);
        assert!((back.floor - 1.5).abs() < 1e-6);
        assert!((back.score(&[("k=delete".into(), 2.0)]) - 3.0).abs() < 1e-6);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn reads_kind_overrides_and_exempt_kinds() {
        let r = Reranker::from_tsv(
            "#tau\t-1.5,-0.5,0.5\n#floor\t1\n#exempt\tinflection-aux\n#tau_kind\tdelete\t-1.5,-1.2,0.5\nb\t0.1\n",
        )
        .unwrap();
        assert!((r.tau_for("delete", Domain::General) + 1.2).abs() < 1e-6);
        // 上書きの無い種類は文書種類ごとの閾値
        assert!((r.tau_for("substitute", Domain::General) + 0.5).abs() < 1e-6);
        assert!(r.is_exempt("inflection-aux"));
        assert!(!r.is_exempt("inflection"));
        // 保存して読み直しても同じ
        let path =
            std::env::temp_dir().join(format!("celso-rerank-kind-{}.tsv", std::process::id()));
        r.save(&path).unwrap();
        let back = Reranker::load(&path).unwrap();
        assert!((back.tau_for("delete", Domain::Contract) - 0.5).abs() < 1e-6);
        assert!(back.is_exempt("inflection-aux"));
        std::fs::remove_file(path).ok();
    }
}
