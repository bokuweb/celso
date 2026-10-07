fn main() -> anyhow::Result<()> {
    use celso::lm::LanguageModel;
    let lm = celso::lm::Model::load(std::path::Path::new("data/model.bin"))?;
    let tok = celso::tokenize::Tokenizer::new()?;
    for s in [
        "駅の西口には宿泊施設が多くある。",
        "駅の西口には宿泊から施設が多くある。",
        "納税者は納付書により市税を納付しなければならない。",
    ] {
        let toks = tok.tokenize(s);
        let mut ids = vec![celso::lm::BOS];
        ids.extend(toks.iter().map(|t| lm.token_id(t)));
        let mut line = String::new();
        for j in 1..ids.len() {
            line.push_str(&format!(
                "{}:{:.2}({}) ",
                toks[j - 1].surface,
                Model_logp(&lm, &ids[j.saturating_sub(2)..j], ids[j]),
                lm.match_order(&ids[j.saturating_sub(2)..j], ids[j])
            ));
        }
        println!("{line}");
    }
    Ok(())
}
#[allow(non_snake_case)]
fn Model_logp(lm: &celso::lm::Model, c: &[u32], w: u32) -> f32 {
    lm.logp(c, w)
}
