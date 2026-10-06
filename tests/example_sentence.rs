//! ユーザー提示の例文の回帰テスト。モデル (data/model_mix.bin) がある環境でだけ走る。
//! cargo test --release -- --ignored

use std::path::Path;

use celso::checker::{Checker, Config, load_inflections, load_readings};
use celso::lm::Model;
use celso::tokenize::Tokenizer;

#[test]
#[ignore = "data/model_mix.bin が必要 (scripts/build_all.sh で作る)"]
fn detects_all_errors_in_example_sentence() {
    let model = Path::new("data/model_mix.bin");
    let checker = Checker::new(
        Tokenizer::new().unwrap(),
        Model::load(model).unwrap(),
        Config::default(),
        load_inflections(Path::new("data/inflections.tsv")).unwrap(),
        load_readings(Path::new("data/readings.tsv")).unwrap(),
    );
    let text =
        "西口側までは宿泊から施設や地元の日本酒や、山の幸を揃えた飲食は店、呑み屋など多くあろう";
    let found: Vec<(String, String)> = checker
        .check(text)
        .into_iter()
        .map(|f| (f.original, f.replacement))
        .collect();
    assert_eq!(
        found,
        vec![
            ("まで".into(), "".into()),
            ("から".into(), "".into()),
            ("は".into(), "".into()),
            ("あろう".into(), "ある".into()),
        ]
    );
}
