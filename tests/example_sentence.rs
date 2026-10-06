//! ユーザー提示の例文の回帰テスト。モデル (data/model_mix.bin) がある環境でだけ走る。
//! cargo test --release -- --ignored

use std::path::Path;

use celso::checker::{Checker, Config, load_inflections, load_readings};
use celso::lm::Model;
use celso::mlm::Mlm;
use celso::tokenize::Tokenizer;

#[test]
#[ignore = "data/model_mix.bin と data/mlm が必要 (scripts/build_all.sh で作る)"]
fn detects_all_errors_in_example_sentence() {
    let model = Path::new("data/model_mix.bin");
    let cfg = Config {
        mlm_weight: 1.0,
        ..Config::default()
    };
    let checker = Checker::new(
        Tokenizer::new().unwrap(),
        Model::load(model).unwrap(),
        cfg,
        load_inflections(Path::new("data/inflections.tsv")).unwrap(),
        load_readings(Path::new("data/readings.tsv")).unwrap(),
    )
    .with_mlm(Mlm::load(Path::new("data/mlm")).unwrap());
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
            // MLM が「…多くある」との呼応から「西口側には」を選ぶ
            ("まで".into(), "に".into()),
            ("から".into(), "".into()),
            ("は".into(), "".into()),
            ("あろう".into(), "ある".into()),
        ]
    );
}
