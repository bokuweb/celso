//! 文字数を変えない正規化。
//!
//! NFKC の結果が 1 文字になる場合だけ置き換える (全角英数・括弧・半角カナなど)。
//! コーパス抽出 (`scripts/textnorm.py`) と同じ規則なので、学習時と検査時で表記が揃い、
//! しかも検出位置を元テキストの文字オフセットへそのまま戻せる。

use unicode_normalization::UnicodeNormalization;

pub fn norm_char(c: char) -> char {
    if c.is_ascii() {
        return c;
    }
    let mut it = std::iter::once(c).nfkc();
    match (it.next(), it.next()) {
        (Some(n), None) => n,
        _ => c,
    }
}

pub fn norm(text: &str) -> String {
    text.chars().map(norm_char).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_char_count() {
        let src = "（課税の根拠）第１条　ＡＢＣ ｶﾞ ㈱";
        let n = norm(src);
        assert_eq!(src.chars().count(), n.chars().count());
        assert!(n.starts_with("(課税の根拠)第1条 ABC"));
    }
}
