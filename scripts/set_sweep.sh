#!/usr/bin/env bash
# パターン B (n-gram の有無だけ) のモデルを作り、サイズと精度を比べる。
set -u
cd "$(dirname "$0")/.."
BIN=./target-local/release/celso
OUT=data/set_sweep.txt
: > "$OUT"
run() {
  name=$1; vocab=$2; order=$3; mc=$4
  m=data/set_$name.bin
  $BIN build-set --order "$order" --min-count "$mc" --vocab "data/vocab$vocab.txt" -o "$m" \
    data/corpus/egov.wc data/corpus/wiki1.wc data/corpus/reiki.wc 2>&1 | grep -E "kept|built" >> "$OUT"
  size=$(ls -la "$m" | awk '{print $5}')
  echo "=== $name vocab=$vocab order=$order min_count=$mc size=$size" >> "$OUT"
  # 閾値を振った表 (法令文)
  $BIN eval --model "$m" --mlm none --n 300 --thresholds 0,0,0,0,inf,inf --sweep fixtures/yokohama_shizei_jorei.txt 2>&1 \
    | grep -E "^(τ|[0-9])" >> "$OUT"
  echo "西口側までは宿泊から施設や地元の日本酒や、山の幸を揃えた飲食は店、呑み屋など多くあろう" \
    | $BIN check --model "$m" --mlm none --domain legal 2>&1 | grep -E "^[0-9]" | cut -f2,3,4 | tr '\n' ' ' | sed 's/^/example: /' >> "$OUT"
  echo >> "$OUT"
}
run v50k_o3 50k 3 1,2,2
run v50k_o3_all 50k 3 1,1,1
run v20k_o3 20k 3 1,2,2
run v50k_o4 50k 4 1,2,2,2
