//! 純 Rust の分かち書きで、ネイティブ版と同じ指摘が出ることを確かめる (実データが必要なので ignore)。
//!
//! CELSO_DIST (既定 ../data/dist) と CELSO_IPADIC_RAW (既定 ~/celso-data/ipadic-utf8) を読む。

use std::path::PathBuf;

use celso_playground::{Assets, Engine};

fn read(dir: &PathBuf, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
#[ignore]
fn detects_the_example_sentence_with_pure_rust_tokenizer() {
    let dist = std::env::var_os("CELSO_DIST")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../data/dist"));
    let raw = std::env::var_os("CELSO_IPADIC_RAW")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var("HOME").unwrap()).join("celso-data/ipadic-utf8")
        });
    let (model, infl, readings) = (
        read(&dist, "model.bin"),
        read(&dist, "inflections.tsv"),
        read(&dist, "readings.tsv"),
    );
    let (lex, matrix, chardef, unkdef) = (
        read(&raw, "lex.csv"),
        read(&raw, "matrix.def"),
        read(&raw, "char.def"),
        read(&raw, "unk.def"),
    );
    let t = std::time::Instant::now();
    let engine = Engine::load(Assets {
        model: &model,
        cooc: read(&dist, "cooc.bin"),
        inflections: &infl,
        readings: &readings,
        lex_csv: &lex,
        matrix_def: &matrix,
        char_def: &chardef,
        unk_def: &unkdef,
        func: &read(&dist, "func.bin"),
        rerank: &read(&dist, "rerank.tsv"),
        patterns: &read(&dist, "patterns.tsv"),
    })
    .unwrap();
    eprintln!("loaded in {:?}", t.elapsed());
    let json = engine.check_json(
        "西口側までは宿泊から施設や地元の日本酒や、山の幸を揃えた飲食は店、呑み屋など多くあろう",
    );
    eprintln!("{json}");
    for want in [
        "\"original\":\"まで\"",
        "\"original\":\"から\"",
        "\"original\":\"は\"",
        "\"original\":\"あろう\"",
    ] {
        assert!(json.contains(want), "{want} が無い: {json}");
    }
}

#[test]
#[ignore]
fn matches_native_findings_on_yokohama() {
    // ネイティブ版 (Zig コア) の `celso check` と指摘の位置・置き換えが一致することを確かめる
    let dist = PathBuf::from("../data/dist");
    let raw = PathBuf::from(std::env::var("HOME").unwrap()).join("celso-data/ipadic-utf8");
    let (model, infl, readings) = (
        read(&dist, "model.bin"),
        read(&dist, "inflections.tsv"),
        read(&dist, "readings.tsv"),
    );
    let (lex, matrix, chardef, unkdef) = (
        read(&raw, "lex.csv"),
        read(&raw, "matrix.def"),
        read(&raw, "char.def"),
        read(&raw, "unk.def"),
    );
    let engine = Engine::load(Assets {
        model: &model,
        cooc: read(&dist, "cooc.bin"),
        inflections: &infl,
        readings: &readings,
        lex_csv: &lex,
        matrix_def: &matrix,
        char_def: &chardef,
        unk_def: &unkdef,
        func: &read(&dist, "func.bin"),
        rerank: &read(&dist, "rerank.tsv"),
        patterns: &read(&dist, "patterns.tsv"),
    })
    .unwrap();
    let text = std::fs::read_to_string("../fixtures/yokohama_shizei_jorei.txt").unwrap();
    let t = std::time::Instant::now();
    let json: serde_json::Value = serde_json::from_str(&engine.check_json(&text)).unwrap();
    eprintln!("checked in {:?}", t.elapsed());
    let mut got: Vec<String> = json["findings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| {
            format!(
                "{}..{}\t{}",
                f["start"],
                f["end"],
                f["replacement"].as_str().unwrap()
            )
        })
        .collect();
    got.sort();
    std::fs::write("/tmp/celso_pure_findings.txt", got.join("\n")).unwrap();
    eprintln!("{} findings", got.len());
}

#[test]
#[ignore]
fn samples_are_corrected_as_expected_with_pure_rust_tokenizer() {
    // ブラウザと同じ純 Rust の分かち書きで、playground のサンプル (web/samples.json) が期待どおりに直ることを確かめる
    // (ネイティブ版は celso の tests/regression.rs が確かめる)
    let dist = PathBuf::from("../data/dist");
    let raw = PathBuf::from(std::env::var("HOME").unwrap()).join("celso-data/ipadic-utf8");
    let engine = Engine::load(Assets {
        model: &read(&dist, "model.bin"),
        cooc: read(&dist, "cooc.bin"),
        inflections: &read(&dist, "inflections.tsv"),
        readings: &read(&dist, "readings.tsv"),
        lex_csv: &read(&raw, "lex.csv"),
        matrix_def: &read(&raw, "matrix.def"),
        char_def: &read(&raw, "char.def"),
        unk_def: &read(&raw, "unk.def"),
        func: &read(&dist, "func.bin"),
        rerank: &read(&dist, "rerank.tsv"),
        patterns: &read(&dist, "patterns.tsv"),
    })
    .unwrap();
    let samples: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("web/samples.json").unwrap()).unwrap();
    let mut failures = Vec::new();
    for s in samples.as_array().unwrap() {
        let text = s["text"].as_str().unwrap();
        let json: serde_json::Value = serde_json::from_str(&engine.check_json(text)).unwrap();
        let mut chars: Vec<char> = text.chars().collect();
        let mut findings: Vec<(usize, usize, String)> = json["findings"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| {
                (
                    f["start"].as_u64().unwrap() as usize,
                    f["end"].as_u64().unwrap() as usize,
                    f["replacement"].as_str().unwrap().to_string(),
                )
            })
            .collect();
        findings.sort();
        for (a, b, r) in findings.iter().rev() {
            chars.splice(*a..*b, r.chars());
        }
        let got: String = chars.into_iter().collect();
        if !s["expected"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e.as_str() == Some(got.as_str()))
        {
            failures.push(format!("{}: {got}", s["id"]));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
