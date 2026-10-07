//! 1 スレッドでの処理時間の内訳 (分かち書きだけ / 検査全体)。
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let text = std::fs::read_to_string("fixtures/yokohama_shizei_jorei.txt")?;
    let n = celso::norm::norm(&text);
    let sents: Vec<String> = n
        .split(|c: char| c == '\n' || c == '。' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    let tok = celso::tokenize::Tokenizer::new()?;
    let lm = celso::lm::Model::load(std::path::Path::new("data/model.bin"))?;
    for _ in 0..2 {
        let t = Instant::now();
        let mut ntok = 0;
        for s in &sents {
            ntok += tok.tokenize(s).len();
        }
        let t_tok = t.elapsed();
        // 全トークンの logp を 1 回ずつ引く (n-gram 参照の速さの目安)
        let t = Instant::now();
        let mut acc = 0f32;
        let mut nq = 0;
        for s in &sents {
            let toks = tok.tokenize(s);
            let ids: Vec<u32> = toks
                .iter()
                .map(|t| celso::lm::LanguageModel::token_id(&lm, t))
                .collect();
            for j in 1..ids.len() {
                acc += lm.logp(&ids[j.saturating_sub(2)..j], ids[j]);
                nq += 1;
            }
        }
        let t_lp = t.elapsed();
        println!(
            "{} sents, {ntok} tokens: tokenize {t_tok:.2?}, tokenize+logp {t_lp:.2?} ({nq} queries) {acc}",
            sents.len()
        );
    }
    Ok(())
}
