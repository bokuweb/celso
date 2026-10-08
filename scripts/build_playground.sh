#!/usr/bin/env bash
# playground (GitHub Pages) を playground/web に組み立てる。
#
# - WebAssembly: playground/ crate を wasm32 でビルドし、wasm-bindgen (0.2.129) と wasm-opt を通す
# - データ: 配布モデル (data/dist) と IPADIC の生ファイル (UTF-8) を gzip して playground/web/assets へ置く
#   (ブラウザ側は DecompressionStream で展開する)。生成物は git に入れず、gh-pages ブランチへだけ置く
#
# 環境変数:
#   CELSO_DIST        配布モデルのフォルダ (既定 data/dist)
#   CELSO_IPADIC_RAW  UTF-8 化した IPADIC (lex.csv / matrix.def / char.def / unk.def) のフォルダ
#   CELSO_IPADIC_SRC  mecab-ipadic の元フォルダ (COPYING を同梱するため)
set -euo pipefail
cd "$(dirname "$0")/.."
DIST=${CELSO_DIST:-data/dist}
RAW=${CELSO_IPADIC_RAW:-$HOME/celso-data/ipadic-utf8}
SRC=${CELSO_IPADIC_SRC:-$HOME/celso-data/mecab-ipadic-2.7.0-20070801}
CARGO=${CARGO:-cargo}
OUT=playground/web

(cd playground && CARGO_TARGET_DIR=$PWD/target $CARGO build --release --target wasm32-unknown-unknown)
wasm-bindgen --target web --out-dir $OUT/pkg --no-typescript \
  playground/target/wasm32-unknown-unknown/release/celso_playground.wasm
wasm-opt -O3 --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext --enable-mutable-globals \
  $OUT/pkg/celso_playground_bg.wasm -o $OUT/pkg/celso_playground_bg.wasm

mkdir -p $OUT/assets
for f in model.bin cooc.bin inflections.tsv readings.tsv func.bin rerank.tsv patterns.tsv katakana.tsv; do
  gzip -9c "$DIST/$f" > "$OUT/assets/$f.gz"
done
for f in lex.csv matrix.def char.def unk.def; do
  gzip -9c "$RAW/$f" > "$OUT/assets/$f.gz"
done
cp "$SRC/COPYING" $OUT/IPADIC-COPYING.txt
du -sh $OUT/pkg $OUT/assets
