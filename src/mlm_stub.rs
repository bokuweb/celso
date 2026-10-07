//! `mlm` feature 無効時の代替。[`Mlm`] は値を作れない型にしてあり、検査は n-gram だけで行う。
//! (呼び出し側のコードを feature ごとに分岐させないための置き換え)

use anyhow::Result;

enum Never {}

/// `mlm` feature 無効時は作れない。
pub struct Mlm(Never);

/// 採点要求 (`mlm` feature 有効時と同じ形)。
pub struct Query<'a> {
    pub text: &'a str,
    pub start: usize,
    pub end: usize,
}

impl Mlm {
    /// 呼ばれることはない (値を作れないため)。
    pub fn fill_scores(&self, _queries: &[Query], _margin: usize) -> Result<Vec<(f32, usize)>> {
        match self.0 {}
    }

    /// 呼ばれることはない (値を作れないため)。
    pub fn window_pll(&self, _queries: &[Query], _margin: usize) -> Result<Vec<(f32, usize)>> {
        match self.0 {}
    }
}
