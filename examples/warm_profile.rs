//! キャッシュ命中時の処理の内訳を測る。
use std::time::Instant;
fn main() {
    let text = std::fs::read_to_string("fixtures/yokohama_shizei_jorei.txt").unwrap();
    for _ in 0..3 {
        let t = Instant::now();
        let n = celso::norm::norm(&text);
        let t1 = t.elapsed();
        let d = celso::checker::detect_domain(&n);
        let t2 = t.elapsed();
        let chars: Vec<char> = n.chars().collect();
        let t3 = t.elapsed();
        let mut sents = 0;
        let mut h = 0u64;
        let mut s = 0;
        for i in 0..=chars.len() {
            if i == chars.len() || chars[i] == '\n' || chars[i] == '。' || chars[i].is_whitespace()
            {
                if i > s {
                    let st: String = chars[s..i].iter().collect();
                    use std::hash::{Hash, Hasher};
                    let mut hh = rustc_hash::FxHasher::default();
                    st.hash(&mut hh);
                    h ^= hh.finish();
                    sents += 1;
                }
                s = i + 1;
            }
        }
        let t4 = t.elapsed();
        println!(
            "norm {t1:.2?} detect {:.2?} chars {:.2?} split+hash {:.2?} ({sents} sents, {d:?}, {h})",
            t2 - t1,
            t3 - t2,
            t4 - t3
        );
    }
}
