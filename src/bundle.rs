//! 配布物のディレクトリ (`scripts/build_all.sh` の `data/dist/`) から検査器を組み立てる。
//!
//! `celso check` の既定 (data/ 以下の個別ファイル) と同じ構成を、1 つのディレクトリから読む。
//! 回帰テスト (tests/regression.rs) と組み込み先 (elsa-server) が同じ組み立て方を使うためのもの。
//!
//! | ファイル | 必須 | 役割 |
//! |---|---|---|
//! | model.bin | ○ | 単語 3-gram の言語モデル |
//! | inflections.tsv | ○ | 活用表 |
//! | readings.tsv | | 同音異字表 |
//! | cooc.bin | | 同音異字の共起モデル |
//! | func.bin | | 文法モデル (判定器の特徴量) |
//! | patterns.tsv | | 実際の誤字から集めた書き換えパターン |
//! | rerank.tsv | | 採否の判定器 |
//! | katakana.tsv | | カタカナ語の出現数 (カタカナ語の打ち間違いの検出) |
//! | charlm.bin, kanji_homo.tsv | | 文字単位の言語モデルと同じ読みの漢字の表 (一般文の語の中の 1 字の誤りの検出。両方そろったときだけ使う) |
//! | charrank.tsv | | 文字単位の直しの採否の判定器 |

use std::path::Path;

use anyhow::{Context, Result};

use crate::checker::{Checker, Config, load_inflections, load_readings};
use crate::tokenize::Tokenizer;

/// `dir` の配布物から検査器を作る。任意のファイルは無ければ使わない。
pub fn load_dir(dir: &Path, tok: Tokenizer, cfg: Config) -> Result<Checker> {
    let lm = crate::lm::load_any(&dir.join("model.bin"))
        .with_context(|| format!("{}/model.bin", dir.display()))?;
    let keep = |w: &str| lm.word_id(w) != crate::lm::UNK;
    let infl = load_inflections(&dir.join("inflections.tsv"), &keep)?;
    let readings_path = dir.join("readings.tsv");
    let readings = if readings_path.exists() {
        load_readings(&readings_path, &keep)?
    } else {
        rustc_hash::FxHashMap::default()
    };
    let mut checker = Checker::new(tok, lm, cfg, infl, readings);
    let path = dir.join("cooc.bin");
    if path.exists() {
        checker = checker.with_cooc(crate::cooc::Cooc::load(&path)?);
    }
    let path = dir.join("func.bin");
    if path.exists() {
        checker = checker.with_aux(crate::lm::load_any(&path)?);
    }
    let path = dir.join("patterns.tsv");
    if path.exists() {
        checker = checker.with_patterns(crate::patterns::Patterns::load(&path)?);
    }
    let path = dir.join("katakana.tsv");
    if path.exists() {
        checker = checker.with_katakana(crate::katakana::Katakana::load(&path)?);
    }
    let (charlm, homo) = (dir.join("charlm.bin"), dir.join("kanji_homo.tsv"));
    if charlm.exists() && homo.exists() {
        let mut c = crate::charcheck::CharChecker::load(&charlm, &homo)?;
        let path = dir.join("charrank.tsv");
        if path.exists() {
            c = c.with_rank(crate::charcheck::CharRanker::load(&path)?);
        }
        checker = checker.with_charcheck(c);
    }
    let path = dir.join("rerank.tsv");
    if path.exists() {
        checker = checker.with_rerank(crate::rerank::Reranker::load(&path)?);
    }
    Ok(checker)
}
