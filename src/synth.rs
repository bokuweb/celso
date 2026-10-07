//! 評価用の人工誤り生成。
//!
//! きれいな文に「チェッカーが直したい誤り」を 1 つだけ入れ、正解位置と正解の直し方を持たせる。
//! 学習コーパスに含まれない文 (例: 横浜市市税条例) に対して使う。

use rustc_hash::FxHashMap;

use crate::checker::EditKind;
use crate::tokenize::{Token, Tokenizer};

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

#[derive(Debug, Clone)]
pub struct Example {
    pub clean: String,
    pub text: String,
    /// 誤りのある文での正解範囲 (文字オフセット)。脱落は start == end。
    pub gold_start: usize,
    pub gold_end: usize,
    /// 正解の直し方 (この範囲をこの文字列にすると clean に戻る)
    pub gold_repl: String,
    /// チェッカーが使うべき編集の種類
    pub kind: EditKind,
}

const NOISE_PARTICLES: &[&str] = &[
    "が", "の", "を", "に", "と", "で", "から", "まで", "は", "も", "や",
];

fn splice(chars: &[char], a: usize, b: usize, ins: &str) -> String {
    let mut s: String = chars[..a].iter().collect();
    s.push_str(ins);
    s.extend(chars[b..].iter());
    s
}

/// `kind` の誤りを 1 つ入れる。入れられる場所がなければ None。
pub fn corrupt(
    tok: &Tokenizer,
    clean: &str,
    kind: EditKind,
    infl: &FxHashMap<(String, String), Vec<String>>,
    rng: &mut Rng,
) -> Option<Example> {
    let toks: Vec<Token> = tok.tokenize(clean);
    let chars: Vec<char> = clean.chars().collect();
    let pick = |rng: &mut Rng, idx: &[usize]| -> Option<usize> {
        if idx.is_empty() {
            None
        } else {
            Some(idx[rng.below(idx.len())])
        }
    };
    match kind {
        // 余計な助詞を入れる → 直しは削除
        EditKind::Delete => {
            let idx: Vec<usize> = (0..toks.len().saturating_sub(1))
                .filter(|&i| {
                    toks[i].pos == "名詞" && matches!(toks[i + 1].pos, "名詞" | "動詞" | "形容詞")
                })
                .collect();
            let i = pick(rng, &idx)?;
            let p = NOISE_PARTICLES[rng.below(NOISE_PARTICLES.len())];
            let at = toks[i].end;
            Some(Example {
                clean: clean.into(),
                text: splice(&chars, at, at, p),
                gold_start: at,
                gold_end: at + p.chars().count(),
                gold_repl: String::new(),
                kind,
            })
        }
        // 助詞を落とす → 直しは挿入
        EditKind::Insert => {
            let idx: Vec<usize> = (1..toks.len())
                .filter(|&i| toks[i].pos == "助詞" && matches!(toks[i].pos1, "格助詞" | "係助詞"))
                .collect();
            let i = pick(rng, &idx)?;
            Some(Example {
                clean: clean.into(),
                text: splice(&chars, toks[i].start, toks[i].end, ""),
                gold_start: toks[i].start,
                gold_end: toks[i].start,
                gold_repl: toks[i].surface.clone(),
                kind,
            })
        }
        EditKind::Substitute => {
            let idx: Vec<usize> = (0..toks.len())
                .filter(|&i| toks[i].pos == "助詞" && matches!(toks[i].pos1, "格助詞" | "係助詞"))
                .collect();
            let i = pick(rng, &idx)?;
            let p = loop {
                let p = NOISE_PARTICLES[rng.below(NOISE_PARTICLES.len())];
                if p != toks[i].surface {
                    break p;
                }
            };
            Some(Example {
                clean: clean.into(),
                text: splice(&chars, toks[i].start, toks[i].end, p),
                gold_start: toks[i].start,
                gold_end: toks[i].start + p.chars().count(),
                gold_repl: toks[i].surface.clone(),
                kind,
            })
        }
        EditKind::Inflection => {
            let idx: Vec<usize> = (0..toks.len())
                .filter(|&i| {
                    matches!(toks[i].pos, "動詞" | "形容詞")
                        && infl
                            .get(&(toks[i].base.to_string(), toks[i].conj_type.to_string()))
                            .is_some_and(|f| f.len() > 1)
                })
                .collect();
            let i = pick(rng, &idx)?;
            let forms = &infl[&(toks[i].base.to_string(), toks[i].conj_type.to_string())];
            let f = loop {
                let f = &forms[rng.below(forms.len())];
                if *f != toks[i].surface {
                    break f;
                }
            };
            Some(Example {
                clean: clean.into(),
                text: splice(&chars, toks[i].start, toks[i].end, f),
                gold_start: toks[i].start,
                gold_end: toks[i].start + f.chars().count(),
                gold_repl: toks[i].surface.clone(),
                kind,
            })
        }
        // 同音異字・文字単位の人工誤りは未実装 (JWTD の実データで評価する)
        EditKind::Homophone | EditKind::Char => None,
    }
}

/// 同音異字の誤変換を入れる (「対象」→「対照」)。直しは元の語へ戻す。
///
/// 検査側の修正候補と同じ条件 (2 文字以上で漢字を含む内容語、固有名詞を除く、
/// 漢字の違う同音語で出現数 20 以上・上位 12 語以内) の語だけを対象にする。
pub fn corrupt_homophone(
    tok: &Tokenizer,
    clean: &str,
    readings: &FxHashMap<String, Vec<(String, u32)>>,
    rng: &mut Rng,
) -> Option<Example> {
    let toks: Vec<Token> = tok.tokenize(clean);
    let chars: Vec<char> = clean.chars().collect();
    let kanji_of = |w: &str| {
        w.chars()
            .filter(|c| ('\u{4E00}'..='\u{9FFF}').contains(c) || *c == '々')
            .collect::<String>()
    };
    let alts_of = |t: &Token| -> Vec<String> {
        if !matches!(t.pos, "名詞" | "動詞" | "形容詞" | "副詞")
            || t.pos1 == "固有名詞"
            || t.surface.chars().count() < 2
            || kanji_of(&t.surface).is_empty()
        {
            return Vec::new();
        }
        let Some(alts) = readings.get(t.reading) else {
            return Vec::new();
        };
        alts.iter()
            .take(12)
            .filter(|(a, c)| {
                let (x, y) = (kanji_of(a), kanji_of(&t.surface));
                *a != t.surface
                    && *c >= 20
                    && !(x.chars().all(|ch| y.contains(ch)) || y.chars().all(|ch| x.contains(ch)))
            })
            .map(|(a, _)| a.clone())
            .collect()
    };
    let idx: Vec<usize> = (0..toks.len())
        .filter(|&i| !alts_of(&toks[i]).is_empty())
        .collect();
    if idx.is_empty() {
        return None;
    }
    let i = idx[rng.below(idx.len())];
    let alts = alts_of(&toks[i]);
    let a = &alts[rng.below(alts.len())];
    Some(Example {
        clean: clean.into(),
        text: splice(&chars, toks[i].start, toks[i].end, a),
        gold_start: toks[i].start,
        gold_end: toks[i].start + a.chars().count(),
        gold_repl: toks[i].surface.clone(),
        kind: EditKind::Homophone,
    })
}
