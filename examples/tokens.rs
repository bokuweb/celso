//! 分かち書きの確認用: cargo run --release --example tokens -- "文"
fn main() -> anyhow::Result<()> {
    let tok = celso::tokenize::Tokenizer::new()?;
    for arg in std::env::args().skip(1) {
        for t in tok.tokenize(&celso::norm::norm(&arg)) {
            println!(
                "{}\t{}\t{}\t{}\t{}\t{}",
                t.surface, t.pos, t.pos1, t.conj_type, t.conj_form, t.base
            );
        }
    }
    Ok(())
}
