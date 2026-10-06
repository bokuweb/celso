//! lindera (IPADIC) による分かち書きと、言語モデル用のキー化。

use std::borrow::Cow;

use anyhow::Result;
use lindera::dictionary::load_dictionary;
use lindera::mode::Mode;
use lindera::segmenter::Segmenter;

/// 数字列は 1 トークンにまとめて、このキーで言語モデルに渡す (「第34号」と「第5号」を同一視する)。
pub const NUM_KEY: &str = "<num>";

#[derive(Debug, Clone)]
pub struct Token {
    pub surface: String,
    /// 正規化済みテキスト内の文字オフセット (半開区間)。
    pub start: usize,
    pub end: usize,
    pub pos: String,
    pub pos1: String,
    pub conj_type: String,
    pub conj_form: String,
    pub base: String,
    /// 読み (カタカナ)。同音異字の候補引きに使う。未知語は空。
    pub reading: String,
}

impl Token {
    pub fn is_num(&self) -> bool {
        self.pos1 == "数"
            && self
                .surface
                .chars()
                .all(|c| c.is_ascii_digit() || c == ',' || c == '.')
    }

    pub fn key(&self) -> &str {
        if self.is_num() {
            NUM_KEY
        } else {
            &self.surface
        }
    }
}

#[derive(Clone)]
pub struct Tokenizer {
    seg: Segmenter,
}

impl Tokenizer {
    pub fn new() -> Result<Self> {
        let dict = load_dictionary("embedded://ipadic")?;
        Ok(Self {
            seg: Segmenter::new(Mode::Normal, dict, None),
        })
    }

    /// `text` は [`crate::norm::norm`] 済みであること。空白トークンは落とす。
    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        let Ok(mut toks) = self.seg.segment(Cow::Borrowed(text)) else {
            return Vec::new();
        };
        // byte → char オフセット変換表
        let mut b2c = vec![0usize; text.len() + 1];
        let mut ci = 0;
        for (bi, ch) in text.char_indices() {
            for k in 0..ch.len_utf8() {
                b2c[bi + k] = ci;
            }
            ci += 1;
        }
        b2c[text.len()] = ci;

        let mut out: Vec<Token> = Vec::with_capacity(toks.len());
        for t in toks.iter_mut() {
            let surface = t.surface.to_string();
            if surface.trim().is_empty() {
                continue;
            }
            let (bs, be) = (t.byte_start, t.byte_end);
            let d = t.details();
            let get = |i: usize| d.get(i).map(|s| s.to_string()).unwrap_or_default();
            let tok = Token {
                start: b2c[bs],
                end: b2c[be],
                pos: get(0),
                pos1: get(1),
                conj_type: get(4),
                conj_form: get(5),
                base: get(6),
                reading: get(7),
                surface,
            };
            // 「1」「,」「500」のように割れた数字列を 1 トークンへまとめる
            if let Some(prev) = out.last_mut()
                && prev.is_num()
                && prev.end == tok.start
                && tok.is_num()
            {
                prev.surface.push_str(&tok.surface);
                prev.end = tok.end;
                continue;
            }
            out.push(tok);
        }
        out
    }
}
