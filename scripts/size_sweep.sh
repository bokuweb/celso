#!/usr/bin/env bash
# 語彙・次数・足切りを変えたモデルを作り、サイズと精度を比べる。
set -u
cd "$(dirname "$0")/.."
BIN=./target-local/release/celso
OUT=data/size_sweep.txt
: > "$OUT"
run() {
  name=$1; vocab=$2; order=$3; mc=$4
  m=data/model_$name.bin
  $BIN build-lm --order "$order" --min-count "$mc" --vocab "data/vocab$vocab.txt" -o "$m" \
    data/corpus/egov.wc data/corpus/wiki1.wc data/corpus/reiki.wc 2>&1 | grep -E "kept|built" >> "$OUT"
  size=$(ls -la "$m" | awk '{print $5}')
  echo "=== $name vocab=$vocab order=$order min_count=$mc size=$size" >> "$OUT"
  $BIN eval --model "$m" --mlm none --n 300 fixtures/yokohama_shizei_jorei.txt 2>&1 | grep -E "^(delete|substitute|inflection|insert|clean)" >> "$OUT"
  $BIN eval-jwtd --model "$m" --mlm none --sweep data/jwtd/gold.jsonl 2>&1 | grep -E "^1\.0 " | sed 's/^/gold δ=/' >> "$OUT"
  echo "西口側までは宿泊から施設や地元の日本酒や、山の幸を揃えた飲食は店、呑み屋など多くあろう" \
    | $BIN check --model "$m" --mlm none 2>&1 | grep -E "^[0-9]" | cut -f2,3 | tr '\n' ' ' | sed 's/^/example: /' >> "$OUT"
  echo >> "$OUT"
}
run v50k_o4_a 50k 4 1,1,1,2
run v50k_o4_b 50k 4 1,1,2,3
run v20k_o4 20k 4 1,1,2,3
run v20k_o3 20k 3 1,1,2
run v10k_o3 10k 3 1,2,3
