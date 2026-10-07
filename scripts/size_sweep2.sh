#!/usr/bin/env bash
# パターン A をさらに小さくする (3-gram, 足切り強め)
set -u
cd "$(dirname "$0")/.."
BIN=./target-local/release/celso
OUT=data/size_sweep2.txt
: > "$OUT"
for spec in "v10k_o3_c 10k 3 1,3,5" "v10k_o3_d 10k 3 1,5,10" "v5k_o3 5k 3 1,3,5"; do
  set -- $spec
  name=$1; vocab=$2; order=$3; mc=$4
  m=data/model_$name.bin
  [ -f "data/vocab$vocab.txt" ] || head -${vocab%k}000 data/vocab50k.txt > "data/vocab$vocab.txt"
  $BIN build-lm --order "$order" --min-count "$mc" --vocab "data/vocab$vocab.txt" -o "$m" \
    data/corpus/egov.wc data/corpus/wiki1.wc data/corpus/reiki.wc 2>&1 | grep -E "kept|built" >> "$OUT"
  echo "=== $name size=$(ls -la "$m" | awk '{print $5}')" >> "$OUT"
  $BIN eval --model "$m" --mlm none --n 300 --seed 11 --thresholds 0,0,0,0,inf,inf --sweep fixtures/yokohama_shizei_jorei.txt 2>&1 \
    | grep -E "^(τ|[0-9])" >> "$OUT"
done
