#!/usr/bin/env bash
# 文字単位の直しの候補 ([char] 行) を書き出す (調査用の出力は判定の前に全候補を出す)。
# 使い方: scripts/charrank/dump.sh 配布物のディレクトリ 出現数だけの判定器 入力 (1 行 1 文) 出力
# 出現数だけの判定器 (#prior 行) を渡すのは、補う・置き換えるひらがなを配布時と同じに絞るため (重みは使わない)
set -euo pipefail
D=$1
CELSO_TRACE=1 RAYON_NUM_THREADS=1 ./target/release/celso check \
  --model "$D/model.bin" --rerank "$D/rerank.tsv" --cooc "$D/cooc.bin" --aux-model "$D/func.bin" \
  --patterns "$D/patterns.tsv" --katakana "$D/katakana.tsv" \
  --charlm "$D/charlm.bin" --char-homo "$D/kanji_homo.tsv" --char-rank "$2" \
  --char-tau=1 --char-top 3 --char-susp=-3 --char-beam 6 --domain general "$3" 2>&1 | grep '\[char\]' > "$4" || true
