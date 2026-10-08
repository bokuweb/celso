"""celso の src を jlsi/fraim-lint-rs の typo-lint crate へ取り込み、fraim-lint-rs 固有の修正を当てる。

使い方: FRAIM_LINT_RS=<fraim-lint-rs のチェックアウト> python3 scripts/sync_fraim_lint.py
取り込んだあと fraim-lint-rs で cargo fmt --all を実行する。
"""
import os
import re
import shutil

C = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(os.path.expanduser(os.environ.get('FRAIM_LINT_RS', '~/ghq/github.com/jlsi/fraim-lint-rs')), 'typo-lint')
os.makedirs(f'{F}/src', exist_ok=True)
os.makedirs(f'{F}/tests/regression', exist_ok=True)

for f in ['lm', 'checker', 'norm', 'cooc', 'patterns', 'rerank', 'bundle', 'katakana']:
    shutil.copy(f'{C}/src/{f}.rs', f'{F}/src/{f}.rs')
# MLM (candle) は持ち込まない。checker のコードを celso と揃えるため、値を作れない代替を mlm.rs として置く
shutil.copy(f'{C}/src/mlm_stub.rs', f'{F}/src/mlm.rs')
shutil.copy(f'{C}/tests/regression/cases.tsv', f'{F}/tests/regression/cases.tsv')
shutil.copy(f'{C}/tests/regression/known_gaps.tsv', f'{F}/tests/regression/known_gaps.tsv')


def rd(p):
    return open(f'{F}/{p}').read()


def wr(p, s):
    open(f'{F}/{p}', 'w').write(s)


def must_replace(s, a, b, count=1):
    assert a in s, a[:80]
    return s.replace(a, b, count)


# --- tokenize.rs: Zig コアの経路だけを残し、辞書は tokenizer-delarocha と共有する ---
t = open(f'{C}/src/tokenize.rs').read()
t = t[:t.index('// ---- 純 Rust の経路')].rstrip() + '\n'
t = t.replace('// ---- Zig コアの経路 (既定) ----\n\n', '')
t = t.replace('#[cfg(feature = "zig")]\n', '')
t = must_replace(t, '''//! delarocha (Zig コア + IPADIC) による分かち書きと、言語モデル用のキー化。
//!
//! 辞書は MeCab 形式の生ファイル (UTF-8 化した mecab-ipadic) から一度だけバイナリへ変換し
//! (`celso build-dict`)、起動時はそれを mmap する。ワーカーはスレッドごとに 1 つ持つ。
//!
//! `zig` feature を外したとき (ブラウザ向けの playground) は、delarocha の純 Rust 実装を使い、
//! IPADIC の生ファイル (lex.csv / matrix.def / char.def / unk.def) のバイト列から辞書を作る
//! ([`Tokenizer::from_raw`])。どちらの経路でもトークンの組み立ては共通 ([`assemble`])。''', '''//! delarocha (Zig コア) + 同梱 IPADIC による分かち書きと、言語モデル用のキー化。
//!
//! 辞書は tokenizer-delarocha と同じもの (`asset/dict/ipadic-mecab-2_7_0`) を使う。
//! 言語モデルの語彙は分かち書きの結果に依存するので、学習時 (bokuweb/celso) と実行時で辞書を揃える必要がある。
//! 他の lint と同じプロセスで動かすときは [`Tokenizer::from_shared`] で辞書 (~40MB) を共有する。
//! ワーカーはスレッドごとに 1 つ持つ (delarocha のワーカーは `&mut` で使うため)。''')
t = re.sub(r'/// 既定の辞書パス。.*?\n\}\n\n/// MeCab 形式の生辞書.*?\n\}\n\n', '', t, flags=re.S)
t = must_replace(t, 'use std::path::Path;\nuse std::path::PathBuf;\nuse std::sync::OnceLock;\n',
                 'use std::path::Path;\nuse std::sync::{Arc, OnceLock};\n')
t = must_replace(t, 'use anyhow::Context;\nuse anyhow::Result;\n', 'use anyhow::{Context, Result};\n')
t = must_replace(t, '''// 辞書はプロセスで 1 つだけ読み、ワーカーがそれを 'static で借りる
static DICT: OnceLock<ZigTokenizer> = OnceLock::new();''', '''// 辞書はプロセスで 1 つだけ持ち、ワーカーがそれを 'static で借りる
// (Arc を static に置くので、辞書はプロセスの終わりまで解放されない)
static DICT: OnceLock<Arc<ZigTokenizer>> = OnceLock::new();''')
t = re.sub(r'    /// 既定パスの辞書で作る。\n    pub fn new\(\) -> Result<Self> \{\n.*?\n    \}\n\n    pub fn from_path\(path: &Path\) -> Result<Self> \{\n.*?\n    \}\n', '''    /// tokenizer-delarocha がビルド時に展開した同梱辞書で作る (テスト・ベンチ向け。
    /// ビルドしたマシンのパスを指すので、配布するバイナリでは [`Self::from_shared`] を使う)。
    pub fn new() -> Result<Self> {
        Self::from_path(Path::new(tokenizer_delarocha::BUNDLED_DIC_PATH))
    }

    /// 非圧縮の辞書ファイルを mmap で開く。2 回目以降は最初の辞書を使う。
    pub fn from_path(path: &Path) -> Result<Self> {
        if DICT.get().is_none() {
            let t = ZigTokenizer::from_binary_path(path)
                .with_context(|| format!("辞書 {} を開けない", path.display()))?;
            let _ = DICT.set(Arc::new(t));
        }
        Self::current()
    }

    /// 他の lint (tokenizer-delarocha の `MecabTokenizer::dictionary`) と辞書を共有する。
    /// 2 回目以降は最初の辞書を使う。
    pub fn from_shared(dict: Arc<ZigTokenizer>) -> Result<Self> {
        let _ = DICT.set(dict);
        Self::current()
    }

    fn current() -> Result<Self> {
        Ok(Self {
            dict: DICT.get().context("辞書の初期化に失敗した")?,
        })
    }
''', t, flags=re.S)
t += '''
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_with_bundled_dictionary() {
        let tok = Tokenizer::new().unwrap();
        let toks = tok.tokenize("宿泊から施設を第34号");
        let surfaces: Vec<&str> = toks.iter().map(|t| t.surface.as_str()).collect();
        assert_eq!(surfaces, ["宿泊", "から", "施設", "を", "第", "34", "号"]);
        assert_eq!(toks[5].key(), NUM_KEY);
        assert_eq!((toks[1].start, toks[1].end), (2, 4));
        assert_eq!((toks[1].pos, toks[1].pos1), ("助詞", "格助詞"));
    }
}
'''
wr('src/tokenize.rs', t)

# --- lm.rs: 存在フィルタ版 (ngset) は持ち込まない ---
s = rd('src/lm.rs')
s = must_replace(s, '''        #[cfg(feature = "ngset")]
        return Ok(Box::new(crate::ngset::NgramSet::load(path)?));
        #[cfg(not(feature = "ngset"))]
        anyhow::bail!("存在フィルタ版のモデル (CELSONS1) は ngset feature が必要");''', '''        bail!("存在フィルタ版のモデル (CELSONS1) には対応していない");''')
s = must_replace(s, '''/// チェッカーから見た言語モデル。確率モデル ([`Model`]) と、n-gram の有無だけを持つ
/// 軽量版 ([`crate::ngset::NgramSet`]) を差し替えられるようにする。''', '''/// チェッカーから見た言語モデル (テストで差し替えられるよう trait にしている)。''')
s = s.replace('"celso-lm-test"', '"typo-lint-lm-test"')
wr('src/lm.rs', s)

# --- checker.rs: 調査用の時間計測を外す (CELSO_TRACE は TYPO_LINT_TRACE に) ---
s = rd('src/checker.rs')
s = must_replace(s, '''        let dbg = std::env::var_os("CELSO_DEBUG").is_some();
        // 時計はデバッグ表示のときだけ読む (wasm32-unknown-unknown では Instant::now が panic するため)
        let t0 = dbg.then(std::time::Instant::now);
''', '')
s = re.sub(r'\n\s*if dbg \{\n\s*eprintln!\((?:[^;])*\);\n\s*\}', '', s)
s = s.replace('CELSO_TRACE', 'TYPO_LINT_TRACE')
s = s.replace('celso-checker-test', 'typo-lint-checker-test')
wr('src/checker.rs', s)

s = rd('src/rerank.rs')
s = s.replace('celso-rerank-', 'typo-lint-rerank-')
wr('src/rerank.rs', s)

s = rd('src/bundle.rs')
s = must_replace(s, '''//! `celso check` の既定 (data/ 以下の個別ファイル) と同じ構成を、1 つのディレクトリから読む。
//! 回帰テスト (tests/regression.rs) と組み込み先 (elsa-server) が同じ組み立て方を使うためのもの。''', '''//! モデルは bokuweb/celso の `scripts/build_all.sh` で作り、配布物 (`data/dist/`) をそのまま置く。
//! 回帰テスト (tests/regression.rs) と組み込み先 (elsa-server) が同じ組み立て方を使うためのもの。''')
wr('src/bundle.rs', s)

s = rd('src/cooc.rs')
s = s.replace('bail!("not a celso co-occurrence model (CELSOCO2/3)")', 'bail!("共起モデル (CELSOCO2/3) ではない")')
wr('src/cooc.rs', s)
s = rd('src/lm.rs')
s = s.replace('bail!("not a celso model (CELSOLM4)");', 'bail!("言語モデル (CELSOLM4) ではない");')
wr('src/lm.rs', s)

# --- tests/regression.rs ---
s = open(f'{C}/tests/regression.rs').read()
s = s.replace('use celso::', 'use typo_lint::').replace('celso::bundle', 'typo_lint::bundle')
s = must_replace(s, '''//! 配布物 (scripts/build_all.sh の data/dist/) が無い環境では何もしない。場所は CELSO_DIST で変えられる。
//! cargo test --release --test regression -- --nocapture''', '''//! モデル (bokuweb/celso の scripts/build_all.sh で作る data/dist/) は大きいのでリポジトリに置かない。
//! TYPO_LINT_MODEL_DIR にその場所を渡したときだけ走り、無ければ何もしない。
//! TYPO_LINT_MODEL_DIR=~/celso/data/dist cargo test --release -p typo-lint --test regression -- --nocapture''')
s = re.sub(r'fn dist\(\) -> Option<PathBuf> \{\n.*?\n\}\n', '''fn dist() -> Option<PathBuf> {
    let Some(dir) = std::env::var_os("TYPO_LINT_MODEL_DIR").map(PathBuf::from) else {
        eprintln!("TYPO_LINT_MODEL_DIR が無いので回帰テストを飛ばす");
        return None;
    };
    assert!(dir.join("model.bin").exists(), "{}/model.bin が無い", dir.display());
    Some(dir)
}
''', s, count=1, flags=re.S)
assert 'TYPO_LINT_MODEL_DIR").map' in s
s = s.replace('CELSO_REGRESSION_PROBE', 'TYPO_LINT_REGRESSION_PROBE')
# playground のサンプルは celso 側だけで確かめる
i = s.index('/// playground のサンプル')
s = s[:i].rstrip() + '\n'
s = s.replace('playground のサンプル（[playground/web/samples.json]） も同じテストで確かめる。', '')
s = re.sub(r'\n\s*// 複数行の文書 \(playground のサンプル\) は改行を `\\\\n` と書く', '\n        // 複数行の文書は改行を `\\\\n` と書く', s)
wr('tests/regression.rs', s)
print('synced')

s = rd('tests/regression.rs')
s = s.replace('/// None なら文書ごとに自動判定 (playground・組み込み先と同じ)', '/// None なら文書ごとに自動判定 (組み込み先と同じ)')
s = s.replace('// 複数行の文書 (playground のサンプル) は改行を `\\n` と書く', '// 複数行の文書は改行を `\\n` と書く')
wr('tests/regression.rs', s)
s = rd('src/cooc.rs')
s = s.replace('"celso-cooc-test-', '"typo-lint-cooc-test-')
wr('src/cooc.rs', s)
print('post-fix done')

# 採点の内訳 (本文を含む) を stderr へ出す調査用の出力は、`trace` feature を有効にしたビルドだけで使えるようにする
# (組み込み先のログへ文書の内容が出ないように、既定のビルドでは環境変数を読まない)
s = rd('src/checker.rs')
s = must_replace(s, 'trace: std::env::var_os("TYPO_LINT_TRACE").is_some(),',
                 'trace: cfg!(feature = "trace") && std::env::var_os("TYPO_LINT_TRACE").is_some(),')
s = must_replace(s, '''    /// TYPO_LINT_TRACE が設定されていれば、閾値に届かなかった候補も含めて採点の内訳を stderr へ出す
    /// (ケースの調査用。組み込み先では設定しない)''', '''    /// `trace` feature 付きのビルドで TYPO_LINT_TRACE が設定されていれば、閾値に届かなかった候補も含めて
    /// 採点の内訳を stderr へ出す (ケースの調査用。文書の内容を出すので、組み込み先では feature を有効にしない)''')
wr('src/checker.rs', s)
print('trace gated')
