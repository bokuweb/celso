//! celso をブラウザで動かす playground。
//!
//! モデル (CELSOLM4・共起モデル・文法モデル・判定器・誤字パターン・活用表・同音異字表) と IPADIC の生ファイルを
//! バイト列で受け取り、
//! 純 Rust の分かち書き (delarocha) で検査する。ブラウザでは mmap できないので、モデルはメモリへ読む。
//! 結果は JSON 文字列で返す (オフセットは Unicode スカラー値 = JS の `Array.from(text)` の添字)。

use celso::checker::{
    Checker, Config, Domain, Finding, detect_domain, load_inflections_from_reader,
    load_readings_from_reader,
};
use celso::cooc::Cooc;
use celso::lm::{Model, UNK};
use celso::norm::norm;
use celso::patterns::Patterns;
use celso::rerank::Reranker;
use celso::tokenize::Tokenizer;
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// 検査器本体 (wasm-bindgen に依存しない部分。ネイティブのテストからも使う)。
pub struct Engine {
    checker: Checker,
}

/// 読み込むファイル一式。
pub struct Assets<'a> {
    pub model: &'a [u8],
    pub cooc: Vec<u8>,
    pub inflections: &'a [u8],
    pub readings: &'a [u8],
    pub lex_csv: &'a [u8],
    pub matrix_def: &'a [u8],
    pub char_def: &'a [u8],
    pub unk_def: &'a [u8],
    /// 文法モデル (判定器の特徴量)。空なら使わない
    pub func: &'a [u8],
    /// 採否の判定器 (TSV)。空なら種類ごとの閾値で決める
    pub rerank: &'a [u8],
    /// 実際の誤字から集めた書き換えパターン (TSV)。空なら使わない
    pub patterns: &'a [u8],
}

#[derive(Serialize)]
struct Suggestion<'a> {
    start: usize,
    end: usize,
    original: &'a str,
    replacement: &'a str,
    kind: &'static str,
    score: f32,
}

#[derive(Serialize)]
struct Output<'a> {
    domain: &'static str,
    findings: Vec<FindingOut<'a>>,
}

#[derive(Serialize)]
struct FindingOut<'a> {
    #[serde(flatten)]
    best: Suggestion<'a>,
    alternatives: Vec<Suggestion<'a>>,
}

fn domain_label(d: Domain) -> &'static str {
    match d {
        Domain::Legal => "legal",
        Domain::General => "general",
        Domain::Contract => "contract",
    }
}

impl Engine {
    pub fn load(a: Assets<'_>) -> anyhow::Result<Self> {
        let tok = Tokenizer::from_raw(a.lex_csv, a.matrix_def, a.char_def, a.unk_def)?;
        let lm = Model::from_bytes(a.model)?;
        // 活用表・同音異字表は、言語モデルの語彙にある語だけを読む (メモリを抑える)
        let keep = |w: &str| lm.word_id(w) != UNK;
        let inflections = load_inflections_from_reader(a.inflections, &keep)?;
        let readings = load_readings_from_reader(a.readings, &keep)?;
        let mut checker =
            Checker::new(tok, Box::new(lm), Config::default(), inflections, readings).with_cache();
        if !a.cooc.is_empty() {
            checker = checker.with_cooc(Cooc::from_bytes(a.cooc)?);
        }
        if !a.func.is_empty() {
            checker = checker.with_aux(Box::new(Model::from_bytes(a.func)?));
        }
        if !a.patterns.is_empty() {
            checker = checker.with_patterns(Patterns::from_tsv(std::str::from_utf8(a.patterns)?)?);
        }
        if !a.rerank.is_empty() {
            checker = checker.with_rerank(Reranker::from_tsv(std::str::from_utf8(a.rerank)?)?);
        }
        Ok(Self { checker })
    }

    /// 文書を検査して JSON にする。
    pub fn check_json(&self, text: &str) -> String {
        let findings: Vec<Finding> = self.checker.check_document(text);
        let out = Output {
            domain: domain_label(detect_domain(&norm(text))),
            findings: findings
                .iter()
                .map(|f| FindingOut {
                    best: Suggestion {
                        start: f.start,
                        end: f.end,
                        original: &f.original,
                        replacement: &f.replacement,
                        kind: f.kind.label(),
                        score: f.delta,
                    },
                    alternatives: f
                        .alternatives
                        .iter()
                        .map(|a| Suggestion {
                            start: a.start,
                            end: a.end,
                            original: &a.original,
                            replacement: &a.replacement,
                            kind: a.kind.label(),
                            score: a.score,
                        })
                        .collect(),
                })
                .collect(),
        };
        serde_json::to_string(&out)
            .unwrap_or_else(|_| "{\"domain\":\"general\",\"findings\":[]}".to_string())
    }
}

/// JS から使う検査器。
#[wasm_bindgen]
pub struct Playground(Engine);

#[wasm_bindgen]
impl Playground {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: &[u8],
        cooc: Vec<u8>,
        inflections: &[u8],
        readings: &[u8],
        lex_csv: &[u8],
        matrix_def: &[u8],
        char_def: &[u8],
        unk_def: &[u8],
        func: &[u8],
        rerank: &[u8],
        patterns: &[u8],
    ) -> Result<Playground, JsValue> {
        Engine::load(Assets {
            model,
            cooc,
            inflections,
            readings,
            lex_csv,
            matrix_def,
            char_def,
            unk_def,
            func,
            rerank,
            patterns,
        })
        .map(Playground)
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// 検査結果の JSON (`{ domain, findings: [{ start, end, original, replacement, kind, score, alternatives }] }`)。
    pub fn check(&self, text: &str) -> String {
        self.0.check_json(text)
    }
}
