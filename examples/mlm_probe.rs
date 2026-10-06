//! MLM の採点を手で確かめる: cargo run --release --example mlm_probe -- <model_dir>
use std::time::Instant;

use celso::mlm::{Mlm, Query};

fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "{}/celso-data/hf/modernbert-ja-30m",
            std::env::var("HOME").unwrap()
        )
    });
    let t = Instant::now();
    let m = Mlm::load(std::path::Path::new(&dir))?;
    println!("loaded in {:.2?}", t.elapsed());
    // 「西口側」の直後 (文字 3..5) を差し替えた候補を比べる
    let cands = [
        (
            "西口側までは宿泊施設や地元の日本酒や、山の幸を揃えた飲食店、呑み屋など多くある",
            3,
            5,
        ),
        (
            "西口側は宿泊施設や地元の日本酒や、山の幸を揃えた飲食店、呑み屋など多くある",
            3,
            3,
        ),
        (
            "西口側には宿泊施設や地元の日本酒や、山の幸を揃えた飲食店、呑み屋など多くある",
            3,
            4,
        ),
        (
            "西口側では宿泊施設や地元の日本酒や、山の幸を揃えた飲食店、呑み屋など多くある",
            3,
            4,
        ),
    ];
    let qs: Vec<Query> = cands
        .iter()
        .map(|(t, a, b)| Query {
            text: t,
            start: *a,
            end: *b,
        })
        .collect();
    for _ in 0..3 {
        let t = Instant::now();
        let s = m.window_pll(&qs, 2)?;
        println!("{:.2?}", t.elapsed());
        for ((c, _, _), v) in cands.iter().zip(&s) {
            println!("  {:8.3} ({} subwords)  {c}", v.0, v.1);
        }
    }
    Ok(())
}
