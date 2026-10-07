//! 誤字脱字の検出結果の回帰テスト (実データが必要)。
//!
//! `tests/regression/cases.tsv` の 1 行が 1 ケースで、「文書の種類 \t 入力 \t 期待する修正後の文」。
//! 文書の種類は legal / general / contract で固定するか、auto で自動判定させる。
//! 入力に出た指摘 (最良案) をすべて当てた結果が、期待する文 (` || ` 区切りでどれか) と一致することを確かめる。
//! 期待が `=` の行は「指摘が 1 件も出ない」ことを確かめる。改行を含む文書は `\n` と書く。
//!
//! 修正後の文が `=` でない行は、その修正後の文も「指摘が出ない」ことを確かめる
//! (直した文に別の指摘が出ると、利用者は直しても指摘が消えない)。
//!
//! 検出漏れが分かっているが今は直せないケースは `tests/regression/known_gaps.tsv` に置く
//! (同じ形式。失敗にはせず、直ったら知らせる)。
//!
//! 配布物 (scripts/build_all.sh の data/dist/) が無い環境では何もしない。場所は CELSO_DIST で変えられる。
//! cargo test --release --test regression -- --nocapture

use std::path::PathBuf;

use celso::checker::{Checker, Config, Domain};
use celso::tokenize::Tokenizer;

struct Case {
    line: usize,
    /// None なら文書ごとに自動判定 (playground・組み込み先と同じ)
    domain: Option<Domain>,
    input: String,
    expected: Vec<String>,
}

fn dist() -> Option<PathBuf> {
    let dir =
        std::env::var_os("CELSO_DIST").map_or_else(|| PathBuf::from("data/dist"), PathBuf::from);
    if dir.join("model.bin").exists() {
        Some(dir)
    } else {
        eprintln!(
            "{} が無いので回帰テストを飛ばす (scripts/build_all.sh で作る)",
            dir.display()
        );
        None
    }
}

fn parse(path: &str) -> Vec<Case> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut cases = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(
            cols.len(),
            3,
            "{path}:{}: 列は 3 つ (種類 \\t 入力 \\t 期待)",
            i + 1
        );
        let domain = match cols[0] {
            "legal" => Some(Domain::Legal),
            "general" => Some(Domain::General),
            "contract" => Some(Domain::Contract),
            "auto" => None,
            d => panic!("{path}:{}: 未知の種類 {d}", i + 1),
        };
        // 複数行の文書 (playground のサンプル) は改行を `\n` と書く
        let unescape = |s: &str| s.replace("\\n", "\n");
        let input = unescape(cols[1]);
        let expected = if cols[2] == "=" {
            vec![input.clone()]
        } else {
            cols[2].split(" || ").map(unescape).collect()
        };
        cases.push(Case {
            line: i + 1,
            domain,
            input,
            expected,
        });
    }
    cases
}

/// 指摘 (最良案) をすべて当てた文と、指摘の一覧 (表示用)。
fn corrected(checker: &mut Checker, domain: Option<Domain>, text: &str) -> (String, Vec<String>) {
    checker.cfg.domain = domain;
    let mut findings = checker.check_document(text);
    findings.sort_by_key(|f| f.start);
    let mut chars: Vec<char> = text.chars().collect();
    let shown = findings
        .iter()
        .map(|f| {
            format!(
                "{}..{} {}「{}」→「{}」{:+.2}",
                f.start,
                f.end,
                f.kind.label(),
                f.original,
                f.replacement,
                f.delta
            )
        })
        .collect();
    for f in findings.iter().rev() {
        chars.splice(f.start..f.end, f.replacement.chars());
    }
    (chars.into_iter().collect(), shown)
}

fn load() -> Option<Checker> {
    let dir = dist()?;
    Some(celso::bundle::load_dir(&dir, Tokenizer::new().unwrap(), Config::default()).unwrap())
}

/// ケースを走らせ、失敗の説明を返す。
fn run(checker: &mut Checker, cases: &[Case], path: &str) -> Vec<String> {
    let mut failures = Vec::new();
    for c in cases {
        let (got, shown) = corrected(checker, c.domain, &c.input);
        if !c.expected.contains(&got) {
            failures.push(format!(
                "{path}:{}\n  入力: {}\n  期待: {}\n  結果: {got}\n  指摘: {shown:?}",
                c.line,
                c.input,
                c.expected.join(" || ")
            ));
        }
        // 直した文 (期待が入力と違うときの 1 つ目) には指摘が出ない
        if c.expected[0] != c.input {
            let (again, shown) = corrected(checker, c.domain, &c.expected[0]);
            if again != c.expected[0] {
                failures.push(format!(
                    "{path}:{} (修正後の文)\n  入力: {}\n  結果: {again}\n  指摘: {shown:?}",
                    c.line, c.expected[0]
                ));
            }
        }
    }
    failures
}

#[test]
fn regression_cases() {
    let Some(mut checker) = load() else {
        return;
    };
    let path = "tests/regression/cases.tsv";
    let cases = parse(path);
    let failures = run(&mut checker, &cases, path);
    assert!(
        failures.is_empty(),
        "{} / {} 件が期待と違う:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
    eprintln!("{} 件すべて期待どおり", cases.len());
}

#[test]
fn known_gaps_are_reported() {
    let Some(mut checker) = load() else {
        return;
    };
    let path = "tests/regression/known_gaps.tsv";
    let cases = parse(path);
    let failures = run(&mut checker, &cases, path);
    // 既知の漏れは失敗にしない。直ったものがあれば cases.tsv へ移すよう知らせる
    let still = failures
        .iter()
        .filter(|f| !f.contains("(修正後の文)"))
        .count();
    eprintln!(
        "既知の漏れ {} 件中 {} 件が直った (直ったものは cases.tsv へ移す)",
        cases.len(),
        cases.len() - still
    );
    for f in &failures {
        eprintln!("{f}");
    }
}

/// CELSO_REGRESSION_PROBE に指定したファイルのケースを走らせて結果を表示する (ケースを足すときの下調べ用)。
#[test]
fn probe() {
    let Some(path) = std::env::var_os("CELSO_REGRESSION_PROBE") else {
        return;
    };
    let Some(mut checker) = load() else {
        return;
    };
    let path = path.to_string_lossy().into_owned();
    let cases = parse(&path);
    let failures = run(&mut checker, &cases, &path);
    for f in &failures {
        println!("{f}");
    }
    println!("{} / {} 件が期待と違う", failures.len(), cases.len());
}

/// playground のサンプル (playground/web/samples.json) が、画面で見せたい指摘のとおりに直ること。
/// サンプルは文書の種類を自動判定させる (playground と同じ)。
#[test]
fn playground_samples() {
    let Some(mut checker) = load() else {
        return;
    };
    let path = "playground/web/samples.json";
    let json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}")),
    )
    .unwrap();
    let cases: Vec<Case> = json
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, s)| Case {
            line: i + 1,
            domain: None,
            input: s["text"].as_str().unwrap().to_string(),
            expected: s["expected"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_str().unwrap().to_string())
                .collect(),
        })
        .collect();
    let failures = run(&mut checker, &cases, path);
    assert!(
        failures.is_empty(),
        "{} 件が期待と違う:\n{}",
        failures.len(),
        failures.join("\n")
    );
    eprintln!("サンプル {} 件すべて期待どおり", cases.len());
}
