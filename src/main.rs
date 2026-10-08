use std::collections::BTreeSet;
use std::io::{BufRead, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Result, bail};
use celso::checker::{Checker, Config, Domain, EditKind, Finding, load_inflections, load_readings};
use celso::lm::{self, BuildConfig, MAX_ORDER};
use celso::norm::norm;
use celso::synth::{Rng, corrupt};
use celso::tokenize::Tokenizer;
use clap::{Parser, Subcommand};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

#[derive(Parser)]
#[command(about = "高速な日本語誤字脱字チェッカー (n-gram LM による候補リスコアリング)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// MeCab 形式の生辞書 (UTF-8 の lex.csv / matrix.def / char.def / unk.def) をバイナリ辞書へ変換する。
    BuildDict {
        raw_dir: PathBuf,
        #[arg(short, long, default_value = "data/ipadic.dic")]
        output: PathBuf,
    },
    /// 分かち書き済みコーパスの語の出現数を数え、上位 n 語を語彙ファイルとして出力する。
    Vocab {
        #[arg(long, default_value_t = 50_000)]
        size: usize,
        #[arg(short, long)]
        output: PathBuf,
        inputs: Vec<PathBuf>,
    },
    /// 1 行 1 文のコーパス (stdin) を分かち書きして stdout へ。活用表も集める。
    Tokenize {
        /// 語彙ファイル。ここに無い語は品詞クラス (<名詞-固有名詞> など) に置き換える
        #[arg(long)]
        vocab: Option<PathBuf>,
        /// 各トークンを「表層形\x1f品詞クラス」で出す (build-lm --vocab で後から語彙を選べる)
        #[arg(long)]
        with_class: bool,
        #[arg(long)]
        inflections: Option<PathBuf>,
        /// 同音異字表 (読み \t 表層形 \t 出現数) の出力先
        #[arg(long)]
        readings: Option<PathBuf>,
    },
    /// 分かち書き済みコーパスから言語モデルを作る。
    BuildLm {
        #[arg(long, default_value_t = 4)]
        order: usize,
        /// 語彙ファイル (--with-class で分かち書きしたコーパス用)。語彙外の語は品詞クラスにする
        #[arg(long)]
        vocab: Option<PathBuf>,
        /// この語を含む n-gram は --keep-min-count で足切りする (同音異字の語など)
        #[arg(long)]
        keep_words: Option<PathBuf>,
        #[arg(long, default_value = "1,2,3,3,3")]
        keep_min_count: String,
        #[arg(long, default_value_t = 2)]
        min_word_count: u32,
        /// 次数ごとの足切り (カンマ区切り, 1-gram から)
        #[arg(long, default_value = "1,1,2,2,2")]
        min_count: String,
        #[arg(short, long)]
        output: PathBuf,
        inputs: Vec<PathBuf>,
    },
    /// キャッシュの効果を測る: 初回 → 2 回目 (全文キャッシュ命中) → 1 文だけ編集して再検査。
    BenchCache {
        #[command(flatten)]
        m: ModelArgs,
        file: PathBuf,
    },
    /// 軽量版 (n-gram の有無だけ) のモデルを作る。
    BuildSet {
        #[arg(long, default_value_t = 3)]
        order: usize,
        #[arg(long)]
        vocab: Option<PathBuf>,
        #[arg(long, default_value_t = 2)]
        min_word_count: u32,
        #[arg(long, default_value = "1,2,2,2,2")]
        min_count: String,
        #[arg(short, long)]
        output: PathBuf,
        inputs: Vec<PathBuf>,
    },
    /// 同音異字の判定に使う文内共起モデルを作る。
    BuildCooc {
        #[arg(long, default_value = "data/model.bin")]
        model: PathBuf,
        #[arg(long)]
        vocab: Option<PathBuf>,
        /// 同音異字の組になる語の一覧
        #[arg(long)]
        homophones: PathBuf,
        #[arg(long, default_value_t = 64)]
        top_k: usize,
        #[arg(long, default_value_t = 5)]
        min_pair: u32,
        #[arg(short, long)]
        output: PathBuf,
        /// 共起の集計の保存先。あれば読み込んでコーパスを数え直さない
        #[arg(long)]
        stats: Option<PathBuf>,
        /// 手がかりにする語の語彙 (1 行 1 語、漢字を含む語だけ使う)。省略時は言語モデルの語彙
        #[arg(long)]
        ctx_vocab: Option<PathBuf>,
        /// 共起語を「共起回数 × PMI」の大きい順に選ぶ (既定は PMI の大きい順)
        #[arg(long)]
        weight_by_count: bool,
        inputs: Vec<PathBuf>,
    },
    /// 誤字パターンの候補 (誤り \t 正しい \t ...) の、コーパスでの出現数を数えて列に足す。
    CountPatterns {
        /// scripts/patterns/extract.py の出力
        candidates: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        /// 数えるコーパス (1 行 1 文のテキスト)
        inputs: Vec<PathBuf>,
    },
    /// 判定器の学習データを書き出す (候補ごとの特徴量と正解ラベル)。
    ///
    /// --synth は「ファイル:文書種類:種類ごとの件数」(人工誤り + 誤りのない文)、--jwtd は実際の誤字。
    DumpRerank {
        #[command(flatten)]
        m: ModelArgs,
        #[arg(long)]
        synth: Vec<String>,
        #[arg(long)]
        jwtd: Option<PathBuf>,
        #[arg(long, default_value_t = 20_000)]
        jwtd_limit: usize,
        /// この n-gram Δ 以上の候補を書き出す
        #[arg(long, default_value_t = 1.0)]
        floor: f32,
        #[arg(long, default_value_t = 7)]
        seed: u64,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// 判定器を学習する (dump-rerank の出力から)。
    TrainRerank {
        inputs: Vec<PathBuf>,
        #[arg(long, default_value_t = 10)]
        epochs: usize,
        #[arg(long, default_value_t = 0.2)]
        lr: f32,
        #[arg(long, default_value_t = 1e-5)]
        l2: f32,
        #[arg(long, default_value_t = 3)]
        min_count: usize,
        #[arg(long, default_value_t = 1.0)]
        floor: f32,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// 活用表・同音異字表を、モデルの語彙にある語だけへ絞って書き出す (配布用)。
    PruneTables {
        #[command(flatten)]
        m: ModelArgs,
        #[arg(short, long)]
        out_dir: PathBuf,
    },
    /// テキストを検査する (ファイル省略時は stdin)。
    Check {
        #[command(flatten)]
        m: ModelArgs,
        file: Option<PathBuf>,
        /// 結果を出さず件数と時間だけ表示
        #[arg(long)]
        quiet: bool,
    },
    /// きれいな文に人工誤りを入れて精度を測る。
    Eval {
        #[command(flatten)]
        m: ModelArgs,
        file: PathBuf,
        #[arg(long, default_value_t = 500)]
        n: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// 閾値を振ったときの検出率と誤検出率の表を出す (--thresholds は 0 にして実行する)
        #[arg(long)]
        sweep: bool,
    },
    /// 京大 日本語Wikipedia入力誤りデータセット (JWTD v2) の test.jsonl で測る。
    EvalJwtd {
        #[command(flatten)]
        m: ModelArgs,
        file: PathBuf,
        /// 誤検出の例を表示する件数
        #[arg(long, default_value_t = 0)]
        show: usize,
        /// 一般文の閾値を「法令文の閾値 + δ」として δ を振り、検出率と誤検出率の表を出す
        #[arg(long)]
        sweep: bool,
    },
}

#[derive(clap::Args)]
struct ModelArgs {
    #[arg(long, default_value = "data/model.bin")]
    model: PathBuf,
    #[arg(long, default_value = "data/inflections.tsv")]
    inflections: PathBuf,
    #[arg(long, default_value = "data/readings.tsv")]
    readings: PathBuf,
    /// 一般文向けの閾値 (同じ並び)。法令文らしくない文書に使う
    #[arg(long, default_value = "4.5,5.5,3,4,4.5,inf")]
    general_thresholds: String,
    /// 契約書向けの閾値 (同じ並び)
    #[arg(long, default_value = "5,6,3.5,4.5,5,inf")]
    contract_thresholds: String,
    /// 文書の種類を固定する (legal / general / contract)。省略時は文書ごとに自動判定
    #[arg(long)]
    domain: Option<String>,
    /// 法令文向けの閾値 (log10): delete,substitute,inflection,insert,homophone,char (inf で無効)
    #[arg(long, default_value = "4,4.5,1.5,3.5,4,inf")]
    thresholds: String,
    /// 助詞の脱落 (挿入候補) を無効にする
    #[arg(long)]
    no_insert: bool,
    /// 未出現判定に使う次数 (0 で無効)
    #[arg(long, default_value_t = 3)]
    novelty: usize,
    /// 文書内でこの回数以上繰り返される並びは指摘しない (0 で無効)
    #[arg(long, default_value_t = 2)]
    doc_repeat: usize,
    /// 2 段目のマスク言語モデル (HuggingFace 形式のディレクトリ)。"none" で無効
    #[arg(long, default_value = "none")]
    mlm: String,
    /// 同音異字の判定に使う文内共起モデル (無ければ使わない)
    #[arg(long, default_value = "data/cooc.bin")]
    cooc: PathBuf,
    #[arg(long, default_value_t = 1.0)]
    cooc_weight: f32,
    /// 文法モデル (機能語 + 品詞クラスの 5-gram)。判定器の特徴量に使う。無ければ使わない
    #[arg(long, default_value = "data/func.bin")]
    aux_model: PathBuf,
    /// 助詞の削除・置換・補いの Δ に足す文法モデルの重み
    #[arg(long, default_value_t = 0.0)]
    aux_weight: f32,
    /// 実際の誤字から集めた書き換えパターン (scripts/patterns/)。無ければ使わない
    #[arg(long, default_value = "data/patterns.tsv")]
    patterns: PathBuf,
    /// カタカナ語の出現数の表 (scripts/katakana_lexicon.py)。カタカナ語の打ち間違いの検出に使う。無ければ使わない
    #[arg(long, default_value = "data/katakana.tsv")]
    katakana: PathBuf,
    /// 採否の判定器 (train-rerank の出力)。無ければ種類ごとの閾値で決める
    #[arg(long, default_value = "data/rerank.tsv")]
    rerank: PathBuf,
    /// 判定器の閾値 (法令文,一般文,契約書 の対数オッズ)。省略時はファイルの値
    #[arg(long)]
    rerank_tau: Option<String>,
    /// 種類ごとの判定器の閾値「delete=-1.5,-1.4,0.5;…」(省略時はファイルの値)
    #[arg(long)]
    rerank_tau_kind: Option<String>,
    /// 判定器を使わない種類 (カンマ区切り。省略時はファイルの値)
    #[arg(long)]
    rerank_exempt: Option<String>,
    /// 最終スコア = n-gram Δ + mlm_weight × MLM Δ
    #[arg(long, default_value_t = 1.0)]
    mlm_weight: f32,
    /// MLM 併用時に 1 段目の閾値を緩める幅
    #[arg(long, default_value_t = 1.5)]
    stage1_slack: f32,
    /// MLM の長さ補正 (サブワード 1 つあたりの nats)
    #[arg(long, default_value_t = 2.0)]
    length_penalty: f32,
    /// n-gram の最良スコアが閾値+band 以上で 2 位との差が gap 以上なら MLM を使わない
    #[arg(long, default_value_t = 2.0)]
    mlm_band: f32,
    #[arg(long, default_value_t = 1.0)]
    mlm_gap: f32,
    /// MLM を PLL (1 サブワードずつマスク) で使う。高精度だが遅い
    #[arg(long)]
    pll: bool,
    /// MLM を採否の判定にも使う (既定は修正案選びだけ)。遅くなる
    #[arg(long)]
    mlm_accept: bool,
    /// MLM で採点する範囲 (編集箇所の前後の文字数)
    #[arg(long, default_value_t = 1)]
    mlm_margin: usize,
}

impl ModelArgs {
    fn config(&self) -> Config {
        let v: Vec<f32> = self
            .thresholds
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        let mut cfg = Config::default();
        for (k, t) in ALL_KINDS.into_iter().zip(v) {
            cfg.thresholds.insert(k, t);
        }
        let g: Vec<f32> = self
            .general_thresholds
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        for (k, t) in ALL_KINDS.into_iter().zip(g) {
            cfg.general_thresholds.insert(k, t);
        }
        let ct: Vec<f32> = self
            .contract_thresholds
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        for (k, t) in ALL_KINDS.into_iter().zip(ct) {
            cfg.contract_thresholds.insert(k, t);
        }
        cfg.domain = match self.domain.as_deref() {
            Some("legal") => Some(celso::checker::Domain::Legal),
            Some("general") => Some(celso::checker::Domain::General),
            Some("contract") => Some(celso::checker::Domain::Contract),
            _ => None,
        };
        cfg.enable_insert = !self.no_insert;
        cfg.novelty_order = self.novelty;
        cfg.doc_repeat_limit = if self.doc_repeat == 0 {
            usize::MAX
        } else {
            self.doc_repeat
        };
        cfg.mlm_weight = self.mlm_weight;
        cfg.cooc_weight = self.cooc_weight;
        cfg.aux_weight = self.aux_weight;
        cfg.stage1_slack = self.stage1_slack;
        cfg.mlm_length_penalty = self.length_penalty;
        cfg.mlm_band = self.mlm_band;
        cfg.mlm_gap = self.mlm_gap;
        cfg.mlm_pll = self.pll;
        cfg.mlm_choice_only = !self.mlm_accept;
        cfg.mlm_margin = self.mlm_margin;
        cfg
    }

    fn load(&self) -> Result<Checker> {
        let t = Instant::now();
        let lm = lm::load_any(&self.model)?;
        let keep = |w: &str| lm.word_id(w) != celso::lm::UNK;
        let infl = load_inflections(&self.inflections, &keep)?;
        let readings = if self.readings.exists() {
            load_readings(&self.readings, &keep)?
        } else {
            Default::default()
        };
        eprintln!(
            "model loaded in {:.2?} (order {}, vocab {})",
            t.elapsed(),
            lm.order(),
            lm.vocab_len()
        );
        let checker = Checker::new(Tokenizer::new()?, lm, self.config(), infl, readings);
        let checker = if self.cooc.exists() {
            checker.with_cooc(celso::cooc::Cooc::load(&self.cooc)?)
        } else {
            checker
        };
        let checker = if self.aux_model.exists() {
            checker.with_aux(lm::load_any(&self.aux_model)?)
        } else {
            checker
        };
        let checker = if self.patterns.exists() {
            checker.with_patterns(celso::patterns::Patterns::load(&self.patterns)?)
        } else {
            checker
        };
        let checker = if self.katakana.exists() {
            checker.with_katakana(celso::katakana::Katakana::load(&self.katakana)?)
        } else {
            checker
        };
        let checker = if self.rerank.exists() {
            let mut r = celso::rerank::Reranker::load(&self.rerank)?;
            if let Some(t) = &self.rerank_tau {
                for (i, x) in t.split(',').enumerate().take(3) {
                    r.tau[i] = x.trim().parse()?;
                }
            }
            // 種類ごとの上書き「種類=法令文,一般文,契約書;…」
            if let Some(t) = &self.rerank_tau_kind {
                for spec in t.split(';').filter(|x| !x.is_empty()) {
                    let (k, v) = spec
                        .split_once('=')
                        .ok_or_else(|| anyhow::anyhow!("--rerank-tau-kind は 種類=a,b,c"))?;
                    let mut a = r.tau;
                    for (i, x) in v.split(',').enumerate().take(3) {
                        a[i] = x.trim().parse()?;
                    }
                    r.tau_kind.insert(k.to_string(), a);
                }
            }
            if let Some(e) = &self.rerank_exempt {
                r.exempt = e
                    .split(',')
                    .filter(|x| !x.is_empty())
                    .map(String::from)
                    .collect();
            }
            checker.with_rerank(r)
        } else {
            checker
        };
        let mlm_dir = PathBuf::from(&self.mlm);
        if self.mlm != "none" && mlm_dir.join("model.safetensors").exists() {
            eprintln!("mlm: {mlm_dir:?}");
            return Ok(checker.with_mlm(celso::mlm::Mlm::load(&mlm_dir)?));
        }
        Ok(checker)
    }
}

const ALL_KINDS: [EditKind; 6] = [
    EditKind::Delete,
    EditKind::Substitute,
    EditKind::Inflection,
    EditKind::Insert,
    EditKind::Homophone,
    EditKind::Char,
];

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::BuildCooc {
            model,
            vocab,
            homophones,
            top_k,
            min_pair,
            output,
            stats,
            ctx_vocab,
            weight_by_count,
            inputs,
        } => {
            let lm = lm::Model::load(&model)?;
            let vocab: Option<rustc_hash::FxHashSet<String>> = match vocab {
                Some(p) => Some(
                    std::fs::read_to_string(p)?
                        .lines()
                        .map(String::from)
                        .collect(),
                ),
                None => None,
            };
            let h: rustc_hash::FxHashSet<String> = std::fs::read_to_string(homophones)?
                .lines()
                .map(String::from)
                .collect();
            let t = Instant::now();
            let ctx_words: Vec<String> = match &ctx_vocab {
                Some(p) => std::fs::read_to_string(p)?
                    .lines()
                    .map(String::from)
                    .collect(),
                None => Vec::new(),
            };
            let st = match &stats {
                Some(p) if p.exists() => celso::cooc::Stats::load(p)?,
                _ => {
                    // 選び直せるように少し低めの回数まで残して数える
                    let st = celso::cooc::count(
                        &inputs,
                        &lm,
                        vocab.as_ref(),
                        &h,
                        &ctx_words,
                        min_pair.min(3),
                    )?;
                    if let Some(p) = &stats {
                        st.save(p)?;
                    }
                    st
                }
            };
            eprintln!(
                "stats ready in {:.1?} ({} pairs, {} context words)",
                t.elapsed(),
                st.pairs.len(),
                st.ctx_words.len()
            );
            let c = st.select_pmi(top_k, min_pair, weight_by_count)?;
            c.save(&output)?;
            eprintln!("built in {:.1?}", t.elapsed());
            Ok(())
        }
        Cmd::CountPatterns {
            candidates,
            output,
            inputs,
        } => {
            use std::io::{BufRead, BufWriter, Write};
            let rows: Vec<Vec<String>> = std::fs::read_to_string(&candidates)?
                .lines()
                .map(|l| l.split('\t').map(String::from).collect())
                .filter(|r: &Vec<String>| r.len() >= 2)
                .collect();
            // 誤り側・正しい側の文字列をまとめて 1 つのオートマトンにする (重複は 1 つに)
            let mut index: rustc_hash::FxHashMap<&str, usize> = Default::default();
            let mut pats: Vec<&str> = Vec::new();
            for r in &rows {
                for s in [&r[0], &r[1]] {
                    index.entry(s.as_str()).or_insert_with(|| {
                        pats.push(s.as_str());
                        pats.len() - 1
                    });
                }
            }
            let ac = aho_corasick::AhoCorasick::new(&pats)?;
            let mut counts = vec![0u64; pats.len()];
            let t = Instant::now();
            for p in &inputs {
                for line in std::io::BufReader::new(std::fs::File::open(p)?).lines() {
                    let line = celso::norm::norm(&line?);
                    for m in ac.find_overlapping_iter(&line) {
                        counts[m.pattern().as_usize()] += 1;
                    }
                }
                eprintln!("{}: done ({:.1?})", p.display(), t.elapsed());
            }
            let mut w = BufWriter::new(std::fs::File::create(&output)?);
            for r in &rows {
                writeln!(
                    w,
                    "{}\t{}\t{}",
                    r.join("\t"),
                    counts[index[r[0].as_str()]],
                    counts[index[r[1].as_str()]]
                )?;
            }
            Ok(())
        }
        Cmd::DumpRerank {
            m,
            synth,
            jwtd,
            jwtd_limit,
            floor,
            seed,
            output,
        } => dump_rerank(m, &synth, jwtd.as_deref(), jwtd_limit, floor, seed, &output),
        Cmd::TrainRerank {
            inputs,
            epochs,
            lr,
            l2,
            min_count,
            floor,
            output,
        } => {
            let mut data = Vec::new();
            for p in &inputs {
                for line in std::fs::read_to_string(p)?.lines() {
                    let mut it = line.split('\t');
                    let (Some(label), Some(weight), Some(_group)) =
                        (it.next(), it.next(), it.next())
                    else {
                        continue;
                    };
                    let feats = it
                        .filter_map(|x| x.split_once('\x1f'))
                        .filter_map(|(k, v)| v.parse().ok().map(|v| (k.to_string(), v)))
                        .collect();
                    data.push(celso::rerank::Example {
                        label: label == "1",
                        weight: weight.parse().unwrap_or(1.0),
                        feats,
                    });
                }
            }
            let pos = data.iter().filter(|e| e.label).count();
            eprintln!("{} examples ({} positive)", data.len(), pos);
            let w = celso::rerank::train(&data, epochs, lr, l2, min_count);
            eprintln!("{} weights", w.len());
            let mut r = celso::rerank::Reranker::new(w);
            r.floor = floor;
            r.save(&output)?;
            Ok(())
        }
        Cmd::PruneTables { m, out_dir } => {
            let lm = lm::Model::load(&m.model)?;
            std::fs::create_dir_all(&out_dir)?;
            let keep = |w: &str| lm.word_id(w) != lm::UNK;
            for (src, name, col) in [
                (&m.inflections, "inflections.tsv", 2usize),
                (&m.readings, "readings.tsv", 1),
            ] {
                let mut w = BufWriter::new(std::fs::File::create(out_dir.join(name))?);
                let mut n = 0;
                for line in std::io::BufReader::new(std::fs::File::open(src)?).lines() {
                    let line = line?;
                    let cols: Vec<&str> = line.split('\t').collect();
                    let ok = cols.get(col).is_some_and(|w| keep(w))
                        && (name != "readings.tsv"
                            || cols.get(2).and_then(|c| c.parse::<u32>().ok()).unwrap_or(0) >= 20);
                    if ok {
                        writeln!(w, "{line}")?;
                        n += 1;
                    }
                }
                eprintln!("{name}: {n} lines");
            }
            Ok(())
        }
        Cmd::BenchCache { m, file } => {
            let checker = m.load()?.with_cache();
            let text = std::fs::read_to_string(file)?;
            let t = Instant::now();
            let a = checker.check_document(&text);
            eprintln!(
                "1st (cold): {} findings in {:.2?} ({} sentences cached)",
                a.len(),
                t.elapsed(),
                checker.cached_sentences()
            );
            let t = Instant::now();
            let b = checker.check_document(&text);
            eprintln!("2nd (warm): {} findings in {:.2?}", b.len(), t.elapsed());
            // 真ん中あたりの長い行に 1 文字足す (= 1 文だけ変わる編集)
            let mut lines: Vec<String> = text.lines().map(String::from).collect();
            let mid = lines.len() / 2;
            let target = (mid..lines.len())
                .find(|&i| lines[i].chars().count() > 30)
                .unwrap_or(mid);
            let at = lines[target].char_indices().nth(10).map_or(0, |(i, _)| i);
            lines[target].insert(at, 'の');
            let edited = lines.join("\n");
            let t = Instant::now();
            let c = checker.check_document(&edited);
            eprintln!(
                "3rd (1 sentence edited): {} findings in {:.2?}",
                c.len(),
                t.elapsed()
            );
            Ok(())
        }
        Cmd::BuildDict { raw_dir, output } => {
            celso::tokenize::build_dict(&raw_dir, &output)?;
            eprintln!("wrote {output:?}");
            Ok(())
        }
        Cmd::BuildSet {
            order,
            vocab,
            min_word_count,
            min_count,
            output,
            inputs,
        } => {
            let mut mc = [1u32; MAX_ORDER];
            for (i, v) in min_count.split(',').enumerate().take(MAX_ORDER) {
                mc[i] = v.parse()?;
            }
            let vocab: Option<rustc_hash::FxHashSet<String>> = match vocab {
                Some(p) => Some(
                    std::fs::read_to_string(p)?
                        .lines()
                        .map(String::from)
                        .collect(),
                ),
                None => None,
            };
            let t = Instant::now();
            let s = celso::ngset::build(&inputs, order, min_word_count, mc, vocab.as_ref())?;
            s.save(&output)?;
            eprintln!("built in {:.1?}", t.elapsed());
            Ok(())
        }
        Cmd::Vocab {
            size,
            output,
            inputs,
        } => {
            let mut wc: FxHashMap<String, u64> = FxHashMap::default();
            for p in inputs {
                for line in std::io::BufReader::new(std::fs::File::open(p)?).lines() {
                    for w in line?
                        .split(' ')
                        .filter(|w| !w.is_empty() && !w.starts_with('<'))
                    {
                        let w = w.split('\x1f').next().unwrap_or(w);
                        *wc.entry(w.to_string()).or_default() += 1;
                    }
                }
            }
            let mut v: Vec<(String, u64)> = wc.into_iter().collect();
            v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let mut w = BufWriter::new(std::fs::File::create(&output)?);
            for (word, _) in v.into_iter().take(size) {
                writeln!(w, "{word}")?;
            }
            Ok(())
        }
        Cmd::Tokenize {
            vocab,
            with_class,
            inflections,
            readings,
        } => tokenize_cmd(vocab, with_class, inflections, readings),
        Cmd::BuildLm {
            order,
            vocab,
            keep_words,
            keep_min_count,
            min_word_count,
            min_count,
            output,
            inputs,
        } => {
            let mut mc = [1u32; MAX_ORDER];
            for (i, v) in min_count.split(',').enumerate().take(MAX_ORDER) {
                mc[i] = v.parse()?;
            }
            let t = Instant::now();
            let vocab = match vocab {
                Some(p) => Some(
                    std::fs::read_to_string(p)?
                        .lines()
                        .map(String::from)
                        .collect(),
                ),
                None => None,
            };
            let mut kmc = [1u32; MAX_ORDER];
            for (i, v) in keep_min_count.split(',').enumerate().take(MAX_ORDER) {
                kmc[i] = v.parse()?;
            }
            let keep_words = match keep_words {
                Some(p) => Some(
                    std::fs::read_to_string(p)?
                        .lines()
                        .map(String::from)
                        .collect(),
                ),
                None => None,
            };
            let m = lm::build(
                &inputs,
                &BuildConfig {
                    order,
                    min_word_count,
                    min_count: mc,
                    vocab,
                    keep_words,
                    keep_min_count: kmc,
                },
            )?;
            m.save(&output)?;
            eprintln!("built in {:.1?}", t.elapsed());
            Ok(())
        }
        Cmd::Check { m, file, quiet } => {
            let checker = m.load()?;
            let text = match file {
                Some(p) => std::fs::read_to_string(p)?,
                None => {
                    let mut s = String::new();
                    std::io::stdin().read_to_string(&mut s)?;
                    s
                }
            };
            let t = Instant::now();
            let findings = checker.check_document(&text);
            let el = t.elapsed();
            if !quiet {
                for f in &findings {
                    let alts: Vec<String> = f
                        .alternatives
                        .iter()
                        .map(|a| {
                            format!("「{}」→「{}」({:.2})", a.original, a.replacement, a.score)
                        })
                        .collect();
                    println!(
                        "{}..{}\t{}\t「{}」→「{}」\tΔ={:.2}\t{}{}",
                        f.start,
                        f.end,
                        f.kind.label(),
                        f.original,
                        f.replacement,
                        f.delta,
                        context(&text, f),
                        if alts.is_empty() {
                            String::new()
                        } else {
                            format!("\t別案: {}", alts.join(" "))
                        }
                    );
                }
            }
            eprintln!(
                "{} findings, {} chars in {:.2?}",
                findings.len(),
                text.chars().count(),
                el
            );
            Ok(())
        }
        Cmd::Eval {
            m,
            file,
            n,
            seed,
            sweep,
        } => eval_cmd(m, file, n, seed, sweep),
        Cmd::EvalJwtd {
            m,
            file,
            show,
            sweep,
        } => eval_jwtd(m, file, show, sweep),
    }
}

fn context(text: &str, f: &Finding) -> String {
    let chars: Vec<char> = text.chars().collect();
    let a = f.start.saturating_sub(12);
    let b = (f.end + 12).min(chars.len());
    let s: String = chars[a..f.start].iter().collect();
    let m: String = chars[f.start..f.end].iter().collect();
    let e: String = chars[f.end..b].iter().collect();
    format!("{s}[{m}]{e}").replace('\n', " ")
}

fn tokenize_cmd(
    vocab: Option<PathBuf>,
    with_class: bool,
    inflections: Option<PathBuf>,
    readings: Option<PathBuf>,
) -> Result<()> {
    let vocab: Option<rustc_hash::FxHashSet<String>> = match vocab {
        Some(p) => Some(
            std::fs::read_to_string(p)?
                .lines()
                .map(String::from)
                .collect(),
        ),
        None => None,
    };
    let mut reading_counts: FxHashMap<(String, String), u32> = FxHashMap::default();
    let tok = Tokenizer::new()?;
    let stdin = std::io::stdin();
    let mut out = BufWriter::new(std::io::stdout().lock());
    let mut infl: BTreeSet<(String, String, String)> = BTreeSet::new();
    let mut lines = stdin.lock().lines();
    loop {
        let chunk: Vec<String> = lines
            .by_ref()
            .take(20_000)
            .collect::<std::io::Result<_>>()?;
        if chunk.is_empty() {
            break;
        }
        #[allow(clippy::type_complexity)]
        let res: Vec<(String, Vec<(String, String, String)>, Vec<(String, String)>)> = chunk
            .par_iter()
            .map_init(
                || tok,
                |tok, line| {
                    let toks = tok.tokenize(&norm(line));
                    let mut inf = Vec::new();
                    let mut rd = Vec::new();
                    let mut s = String::with_capacity(line.len() * 2);
                    for t in &toks {
                        if !s.is_empty() {
                            s.push(' ');
                        }
                        match &vocab {
                            Some(v) if !t.is_num() && !v.contains(t.surface.as_str()) => {
                                s.push_str(&t.class_key());
                            }
                            _ => s.push_str(t.key()),
                        }
                        if with_class && !t.is_num() {
                            s.push('\x1f');
                            s.push_str(&t.class_key());
                        }
                        if matches!(t.pos, "動詞" | "形容詞" | "助動詞") && !t.conj_type.is_empty()
                        {
                            inf.push((
                                t.base.to_string(),
                                t.conj_type.to_string(),
                                t.surface.clone(),
                            ));
                        }
                        if matches!(t.pos, "名詞" | "動詞" | "形容詞" | "副詞")
                            && !t.reading.is_empty()
                            && t.surface.chars().any(is_kanji)
                        {
                            rd.push((t.reading.to_string(), t.surface.clone()));
                        }
                    }
                    (s, inf, rd)
                },
            )
            .collect();
        for (s, inf, rd) in res {
            if !s.is_empty() {
                writeln!(out, "{s}")?;
            }
            infl.extend(inf);
            for k in rd {
                *reading_counts.entry(k).or_default() += 1;
            }
        }
    }
    if let Some(p) = inflections {
        let mut w = BufWriter::new(std::fs::File::create(p)?);
        for (b, t, s) in infl {
            writeln!(w, "{b}\t{t}\t{s}")?;
        }
    }
    if let Some(p) = readings {
        let mut w = BufWriter::new(std::fs::File::create(p)?);
        let mut v: Vec<_> = reading_counts
            .into_iter()
            .filter(|(_, c)| *c >= 5)
            .collect();
        v.sort();
        for ((r, s), c) in v {
            writeln!(w, "{r}\t{s}\t{c}")?;
        }
    }
    Ok(())
}

fn is_kanji(c: char) -> bool {
    ('\u{4E00}'..='\u{9FFF}').contains(&c) || c == '々'
}

fn eval_cmd(m: ModelArgs, file: PathBuf, n: usize, seed: u64, sweep: bool) -> Result<()> {
    let mut checker = m.load()?;
    let text = std::fs::read_to_string(&file)?;
    // 誤り文は 1 文ずつ検査するので、文書の種類は元の文書全体で判定したものに固定する
    if checker.cfg.domain.is_none() {
        let d = celso::checker::detect_domain(&norm(&text));
        eprintln!("domain: {d:?}");
        checker.cfg.domain = Some(d);
    }
    // 平仮名を含む程度の長さの行を「文」として使う
    let mut sents: Vec<String> = Vec::new();
    for line in text.lines() {
        for s in norm(line).split_inclusive('。') {
            let s = s.trim();
            if s.chars().count() >= 12 && s.chars().any(|c| ('ぁ'..='ん').contains(&c)) {
                sents.push(s.to_string());
            }
        }
    }
    let infl = load_inflections(&m.inflections, &|_| true)?;
    let mut rng = Rng::new(seed);
    let readings = {
        let lm = checker.lm.as_ref();
        load_readings(&m.readings, &|w: &str| lm.word_id(w) != celso::lm::UNK)?
    };
    let kinds = [
        EditKind::Delete,
        EditKind::Substitute,
        EditKind::Inflection,
        EditKind::Insert,
        EditKind::Homophone,
    ];
    let mut examples = Vec::new();
    for k in kinds {
        let mut made = 0;
        let mut tries = 0;
        while made < n && tries < n * 20 {
            tries += 1;
            let s = &sents[rng.below(sents.len())];
            let e = if k == EditKind::Homophone {
                celso::synth::corrupt_homophone(&checker.tok, s, &readings, &mut rng)
            } else {
                corrupt(&checker.tok, s, k, &infl, &mut rng)
            };
            if let Some(e) = e {
                examples.push(e);
                made += 1;
            }
        }
    }
    eprintln!(
        "{} clean sentences, {} corrupted examples",
        sents.len(),
        examples.len()
    );

    // 誤り文
    let t = Instant::now();
    // 誤り文は「文書中のその文だけが誤っている」状況を想定し、文書内繰り返しの判定は
    // 元文書での出現回数 + 1 (誤り文自身) で行う
    let doc_norm = norm(&text);
    let doc_chars: Vec<char> = doc_norm.chars().collect();
    let limit = checker.cfg.doc_repeat_limit;
    let texts: Vec<&str> = examples.iter().map(|e| e.text.as_str()).collect();
    let raw = checker.check_many(&texts);
    let results: Vec<Vec<Finding>> = examples
        .iter()
        .zip(raw)
        .map(|(e, fs)| {
            let chars: Vec<char> = e.text.chars().collect();
            fs.into_iter()
                .filter(|f| {
                    let a = f.start.saturating_sub(2);
                    let b = (f.end + 2).min(chars.len());
                    let ctx: String = chars[a..b].iter().collect();
                    ctx.chars().count() < 3 || doc_norm.matches(ctx.as_str()).count() + 1 < limit
                })
                .collect()
        })
        .collect();
    let _ = &doc_chars;
    let el_err = t.elapsed();
    // きれいな文 (誤検出の測定): 文書全体として検査し、文ごとに振り分ける
    let t = Instant::now();
    let joined = sents.join("\n");
    let all = checker.check_document(&joined);
    let el_clean = t.elapsed();
    let mut clean_findings: Vec<Vec<Finding>> = vec![Vec::new(); sents.len()];
    {
        let mut starts = Vec::with_capacity(sents.len());
        let mut off = 0;
        for s in &sents {
            starts.push(off);
            off += s.chars().count() + 1;
        }
        for mut f in all {
            let i = starts.partition_point(|&st| st <= f.start) - 1;
            f.start -= starts[i];
            f.end -= starts[i];
            clean_findings[i].push(f);
        }
    }

    if sweep {
        let clean_chars: usize = sents.iter().map(|s| s.chars().count()).sum();
        println!(
            "τ     | {}",
            kinds
                .map(|k| format!("{:<10} rec/cor/fp10k", k.label()))
                .join(" | ")
        );
        for ti in 0..=30 {
            let tau = ti as f32 * 0.5;
            let mut row = format!("{tau:<5.1} |");
            for k in kinds {
                let (mut total, mut det, mut cor) = (0, 0, 0);
                for (e, fs) in examples.iter().zip(&results) {
                    if e.kind != k {
                        continue;
                    }
                    total += 1;
                    let fs: Vec<&Finding> = fs
                        .iter()
                        .filter(|f| f.kind == k && f.delta >= tau)
                        .collect();
                    if fs
                        .iter()
                        .any(|f| f.start <= e.gold_end && e.gold_start <= f.end)
                    {
                        det += 1;
                    }
                    if fs.iter().any(|f| apply(&e.text, f) == e.clean) {
                        cor += 1;
                    }
                }
                let fp = clean_findings
                    .iter()
                    .flatten()
                    .filter(|f| f.kind == k && f.delta >= tau)
                    .count();
                row.push_str(&format!(
                    " {:>5.1}% {:>5.1}% {:>6.2}      |",
                    100.0 * det as f64 / total.max(1) as f64,
                    100.0 * cor as f64 / total.max(1) as f64,
                    fp as f64 * 1e4 / clean_chars as f64
                ));
            }
            println!("{row}");
        }
    }

    // 取りこぼしの例 (CELSO_SHOW_MISS=insert などで種類を指定)
    if let Ok(kind) = std::env::var("CELSO_SHOW_MISS") {
        let mut shown = 0;
        for (e, fs) in examples.iter().zip(&results) {
            if e.kind.label() != kind || shown >= 30 {
                continue;
            }
            let hit = fs
                .iter()
                .any(|f| f.start <= e.gold_end + 1 && e.gold_start <= f.end + 1);
            if !hit {
                let chars: Vec<char> = e.text.chars().collect();
                let a = e.gold_start.saturating_sub(10);
                let b = (e.gold_end + 10).min(chars.len());
                let ctx: String = chars[a..e.gold_start].iter().collect::<String>()
                    + "["
                    + &chars[e.gold_start..e.gold_end].iter().collect::<String>()
                    + "→"
                    + &e.gold_repl
                    + "]"
                    + &chars[e.gold_end..b].iter().collect::<String>();
                println!("  MISS {kind}: {ctx}");
                shown += 1;
            }
        }
    }
    println!("kind        n    detect  correct  other-fp");
    for k in kinds {
        let (mut total, mut det, mut cor, mut other) = (0, 0, 0, 0);
        for (e, fs) in examples.iter().zip(&results) {
            if e.kind != k {
                continue;
            }
            total += 1;
            let hit = |f: &Finding| {
                if e.gold_start == e.gold_end {
                    f.start <= e.gold_start + 1 && e.gold_start <= f.end + 1
                } else {
                    f.start < e.gold_end && e.gold_start < f.end
                }
            };
            if fs.iter().any(hit) {
                det += 1;
            }
            if fs.iter().any(|f| apply(&e.text, f) == e.clean) {
                cor += 1;
            }
            other += fs.iter().filter(|f| !hit(f)).count();
        }
        if total > 0 {
            println!(
                "{:<10} {:>4}  {:>6.1}%  {:>6.1}%  {:>6}",
                k.label(),
                total,
                100.0 * det as f64 / total as f64,
                100.0 * cor as f64 / total as f64,
                other
            );
        }
    }
    let clean_fp: usize = clean_findings.iter().map(|f| f.len()).sum();
    let clean_chars: usize = sents.iter().map(|s| s.chars().count()).sum();
    println!(
        "clean: {} sentences / {} chars → {} findings ({:.2} per 10k chars)",
        sents.len(),
        clean_chars,
        clean_fp,
        clean_fp as f64 * 10000.0 / clean_chars as f64
    );
    let mut by_kind: FxHashMap<&str, usize> = FxHashMap::default();
    for f in clean_findings.iter().flatten() {
        *by_kind.entry(f.kind.label()).or_default() += 1;
    }
    println!("clean findings by kind: {by_kind:?}");
    println!("time: corrupted {:.2?}, clean {:.2?}", el_err, el_clean);
    // 誤検出のサンプル
    let mut shown = 0;
    for (s, fs) in sents.iter().zip(&clean_findings) {
        for f in fs {
            if shown < 25 {
                println!(
                    "  FP {}\t「{}」→「{}」 Δ={:.2}\t{}",
                    f.kind.label(),
                    f.original,
                    f.replacement,
                    f.delta,
                    context(s, f)
                );
                shown += 1;
            }
        }
    }
    Ok(())
}

/// 判定器の学習データを書き出す。1 行 1 候補で「ラベル \t 重み \t グループ \t 特徴量…」
/// (特徴量は「名前 \x1f 値」)。グループは同じ文の候補をまとめる番号。
#[allow(clippy::too_many_arguments)]
fn dump_rerank(
    m: ModelArgs,
    synth: &[String],
    jwtd: Option<&Path>,
    jwtd_limit: usize,
    floor: f32,
    seed: u64,
    output: &Path,
) -> Result<()> {
    use std::io::Write;
    let checker = m.load()?;
    let mut w = std::io::BufWriter::new(std::fs::File::create(output)?);
    let mut group = 0usize;
    let mut write = |w: &mut std::io::BufWriter<std::fs::File>,
                     text: &str,
                     gold: Option<&str>,
                     d: Domain|
     -> Result<(usize, usize)> {
        group += 1;
        let (mut np, mut nn) = (0, 0);
        for (f, x) in checker.candidate_features(text, d, floor) {
            let label = gold.is_some_and(|g| apply(text, &f) == g);
            if label {
                np += 1;
            } else {
                nn += 1;
            }
            let feats: Vec<String> = x.iter().map(|(k, v)| format!("{k}\x1f{v}")).collect();
            writeln!(w, "{}\t1\t{group}\t{}", u8::from(label), feats.join("\t"))?;
        }
        Ok((np, nn))
    };
    let readings = {
        let lm = checker.lm.as_ref();
        load_readings(&m.readings, &|x: &str| lm.word_id(x) != celso::lm::UNK)?
    };
    let infl = load_inflections(&m.inflections, &|_| true)?;
    for spec in synth {
        let mut parts = spec.rsplitn(3, ':');
        let (Some(n), Some(dom), Some(path)) = (parts.next(), parts.next(), parts.next()) else {
            bail!("--synth はファイル:文書種類:件数");
        };
        let n: usize = n.parse()?;
        let d = match dom {
            "legal" => Domain::Legal,
            "contract" => Domain::Contract,
            _ => Domain::General,
        };
        let text = std::fs::read_to_string(path)?;
        let mut sents: Vec<String> = Vec::new();
        for line in text.lines() {
            for s in norm(line).split_inclusive('。') {
                let s = s.trim();
                if s.chars().count() >= 12 && s.chars().any(|c| ('ぁ'..='ん').contains(&c)) {
                    sents.push(s.to_string());
                }
            }
        }
        let mut rng = Rng::new(seed);
        let (mut np, mut nn) = (0, 0);
        for k in [
            EditKind::Delete,
            EditKind::Substitute,
            EditKind::Inflection,
            EditKind::Insert,
            EditKind::Homophone,
        ] {
            let mut made = 0;
            let mut tries = 0;
            while made < n && tries < n * 20 {
                tries += 1;
                let s = &sents[rng.below(sents.len())];
                let e = if k == EditKind::Homophone {
                    celso::synth::corrupt_homophone(&checker.tok, s, &readings, &mut rng)
                } else {
                    corrupt(&checker.tok, s, k, &infl, &mut rng)
                };
                if let Some(e) = e {
                    let (p, q) = write(&mut w, &e.text, Some(&e.clean), d)?;
                    np += p;
                    nn += q;
                    made += 1;
                }
            }
        }
        // 誤りのない文 (誤検出を学ぶため)。文書全体の文を使う
        for s in &sents {
            let (_, q) = write(&mut w, s, None, d)?;
            nn += q;
        }
        eprintln!("{path}: {np} positive / {nn} negative candidates");
    }
    if let Some(path) = jwtd {
        let (mut np, mut nn, mut used) = (0, 0, 0);
        for line in std::fs::read_to_string(path)?.lines() {
            if used >= jwtd_limit {
                break;
            }
            let v: serde_json::Value = serde_json::from_str(line)?;
            if v["diffs"].as_array().is_none_or(|d| d.len() != 1) {
                continue;
            }
            let pre = norm(v["pre_text"].as_str().unwrap_or(""));
            let post = norm(v["post_text"].as_str().unwrap_or(""));
            if pre == post {
                continue;
            }
            used += 1;
            let (p, q) = write(&mut w, &pre, Some(&post), Domain::General)?;
            let (_, q2) = write(&mut w, &post, None, Domain::General)?;
            np += p;
            nn += q + q2;
        }
        eprintln!(
            "{}: {used} pairs, {np} positive / {nn} negative candidates",
            path.display()
        );
    }
    Ok(())
}

fn apply(text: &str, f: &Finding) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut s: String = chars[..f.start].iter().collect();
    s.push_str(&f.replacement);
    s.extend(chars[f.end..].iter());
    s
}

/// 誤り文と正解文の差分範囲 (誤り文側の文字オフセット)。共通接頭辞・接尾辞を除いた残り。
fn diff_span(pre: &[char], post: &[char]) -> (usize, usize) {
    let mut p = 0;
    while p < pre.len() && p < post.len() && pre[p] == post[p] {
        p += 1;
    }
    let mut s = 0;
    while s < pre.len() - p
        && s < post.len() - p
        && pre[pre.len() - 1 - s] == post[post.len() - 1 - s]
    {
        s += 1;
    }
    (p, pre.len() - s)
}

fn eval_jwtd(m: ModelArgs, file: PathBuf, show: usize, sweep: bool) -> Result<()> {
    let mut checker = m.load()?;
    let legal = checker.cfg.thresholds.clone();
    if sweep {
        // 一般文の閾値を下限まで下げて 1 回だけ検査し、後から δ ごとに足切りする
        for (k, v) in checker.cfg.general_thresholds.iter_mut() {
            if v.is_finite() {
                *v = legal[k];
            }
        }
    }
    struct Rec {
        pre: String,
        post: String,
        cat: String,
    }
    let mut recs = Vec::new();
    for line in std::fs::read_to_string(&file)?.lines() {
        let v: serde_json::Value = serde_json::from_str(line)?;
        // 誤りが 1 箇所の文だけを使う (検出位置の正誤を判定しやすくするため)
        // gold.jsonl には diffs が無いので、カテゴリ "gold" として全件使う
        let cat = match v["diffs"].as_array() {
            Some(diffs) if diffs.len() == 1 => {
                diffs[0]["category"].as_str().unwrap_or("").to_string()
            }
            Some(_) => continue,
            None => "gold".to_string(),
        };
        let pre = norm(v["pre_text"].as_str().unwrap_or(""));
        let post = norm(v["post_text"].as_str().unwrap_or(""));
        if pre != post {
            recs.push(Rec { pre, post, cat });
        }
    }
    let t = Instant::now();
    let mut texts: Vec<&str> = recs.iter().map(|r| r.pre.as_str()).collect();
    texts.extend(recs.iter().map(|r| r.post.as_str()));
    let mut all = checker.check_many(&texts);
    let posts = all.split_off(recs.len());
    let res: Vec<(Vec<Finding>, Vec<Finding>)> = all.into_iter().zip(posts).collect();
    eprintln!("checked {} pairs in {:.2?}", recs.len(), t.elapsed());
    if sweep {
        println!("δ     detect  correct  fp/文");
        for di in 0..=10 {
            let delta = di as f32 * 0.5;
            let keep = |f: &&Finding| f.delta >= legal[&f.kind] + delta;
            let (mut det, mut cor, mut fp) = (0, 0, 0);
            for (r, (fe, fc)) in recs.iter().zip(&res) {
                let pre: Vec<char> = r.pre.chars().collect();
                let post: Vec<char> = r.post.chars().collect();
                let (a, b) = diff_span(&pre, &post);
                let fe: Vec<&Finding> = fe.iter().filter(keep).collect();
                if fe.iter().any(|f| f.start <= b && a <= f.end) {
                    det += 1;
                }
                if fe.iter().any(|f| apply(&r.pre, f) == r.post) {
                    cor += 1;
                }
                fp += fc.iter().filter(keep).count();
            }
            let n = recs.len() as f64;
            println!(
                "{delta:<5.1} {:>6.1}% {:>7.1}% {:>6.1}%",
                100.0 * det as f64 / n,
                100.0 * cor as f64 / n,
                100.0 * fp as f64 / n
            );
        }
        return Ok(());
    }
    let mut cats: std::collections::BTreeMap<String, [usize; 4]> = Default::default();
    let mut shown = 0;
    for (r, (fe, fc)) in recs.iter().zip(&res) {
        let pre: Vec<char> = r.pre.chars().collect();
        let post: Vec<char> = r.post.chars().collect();
        let (a, b) = diff_span(&pre, &post);
        let e = cats.entry(r.cat.clone()).or_default();
        e[0] += 1;
        if fe.iter().any(|f| f.start <= b && a <= f.end) {
            e[1] += 1;
        }
        if fe.iter().any(|f| apply(&r.pre, f) == r.post) {
            e[2] += 1;
        }
        e[3] += fc.len();
        for f in fc {
            if shown < show {
                println!(
                    "  FP {}\t「{}」→「{}」 Δ={:.2}\t{}",
                    f.kind.label(),
                    f.original,
                    f.replacement,
                    f.delta,
                    context(&r.post, f)
                );
                shown += 1;
            }
        }
    }
    println!(
        "{:<20} {:>5} {:>8} {:>8} {:>10}",
        "category", "n", "detect", "correct", "fp(clean)"
    );
    let mut tot = [0usize; 4];
    for (c, v) in &cats {
        println!(
            "{:<20} {:>5} {:>7.1}% {:>7.1}% {:>10}",
            c,
            v[0],
            100.0 * v[1] as f64 / v[0] as f64,
            100.0 * v[2] as f64 / v[0] as f64,
            v[3]
        );
        for i in 0..4 {
            tot[i] += v[i];
        }
    }
    println!(
        "{:<20} {:>5} {:>7.1}% {:>7.1}% {:>10}  (clean 文 {} 件中の誤検出: {:.1}%/文)",
        "TOTAL",
        tot[0],
        100.0 * tot[1] as f64 / tot[0] as f64,
        100.0 * tot[2] as f64 / tot[0] as f64,
        tot[3],
        tot[0],
        100.0 * tot[3] as f64 / tot[0] as f64
    );
    Ok(())
}
