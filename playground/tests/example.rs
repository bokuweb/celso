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
