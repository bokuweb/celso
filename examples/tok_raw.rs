//! delarocha 本体の速度 (トークン数を数えるだけ / feature を読むだけ) と、Token 組み立て込みを比べる。
use std::time::Instant;
fn main() -> anyhow::Result<()> {
    let text = std::fs::read_to_string("fixtures/yokohama_shizei_jorei.txt")?;
    let n = celso::norm::norm(&text);
    let sents: Vec<&str> = n
        .split(|c: char| c == '\n' || c == '。' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .collect();
    let dict = delarocha::ffi::ZigTokenizer::from_binary_path("data/ipadic.dic")?;
    let mut w = dict.create_worker()?;
    let tok = celso::tokenize::Tokenizer::new()?;
    for _ in 0..3 {
        let t = Instant::now();
        let mut c = 0;
        for s in &sents {
            c += w.tokenize_count(s)?;
        }
        let t1 = t.elapsed();
        let t = Instant::now();
        let mut fl = 0;
        for s in &sents {
            for v in w.tokenize_borrowed_views(s)?.iter() {
                fl += v.feature().len() + v.start_char;
            }
        }
        let t2 = t.elapsed();
        let t = Instant::now();
        let mut k = 0;
        for s in &sents {
            k += tok.tokenize(s).len();
        }
        let t3 = t.elapsed();
        println!("count {t1:.2?} ({c}), views+feature {t2:.2?} ({fl}), celso Token {t3:.2?} ({k})");
    }
    Ok(())
}
