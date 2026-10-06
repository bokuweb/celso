#!/usr/bin/env bash
# コーパス取得 → 抽出 → 分かち書き → 言語モデル構築までを一通り行う。
# 必要ディスク: 約 8GB / メモリ: 約 10GB / 所要: 1 時間前後 (ダウンロード速度次第)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/corpus
cargo build --release
BIN=./target/release/celso

# 1. e-Gov 法令 XML 一括ダウンロード (約 300MB)
if [ ! -d data/egov ]; then
  curl -sL "https://laws.e-gov.go.jp/bulkdownload/?file_section=1&only_xml_flag=true" -o data/egov_all_xml.zip
  mkdir -p data/egov && (cd data/egov && unzip -q ../egov_all_xml.zip)
fi
# 2. Wikipedia 日本語版 (分割ダンプの 1 本目, 約 400MB)
[ -f data/jawiki1.xml.bz2 ] || curl -sL -o data/jawiki1.xml.bz2 \
  https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles1.xml-p1p114794.bz2
# 3. 評価用: 京大 日本語Wikipedia入力誤りデータセット v2 (学習コーパスからは除く)
if [ ! -d data/jwtd ]; then
  curl -sL https://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JWTD/jwtd_v2.0.tar.gz | tar xz -C data
  mv data/jwtd_v2.0 data/jwtd
fi

python3 scripts/extract_egov.py data/egov > data/corpus/egov.txt
python3 scripts/extract_wiki.py data/jawiki1.xml.bz2 \
  | python3 scripts/exclude_eval.py data/jwtd/test.jsonl > data/corpus/wiki1.f.txt

$BIN tokenize < data/corpus/egov.txt > data/corpus/egov.wakati
$BIN tokenize < data/corpus/wiki1.f.txt > data/corpus/wiki1.wakati
cat data/corpus/egov.txt data/corpus/wiki1.f.txt \
  | $BIN tokenize --inflections data/inflections.tsv --readings data/readings.tsv > /dev/null

$BIN build-lm --order 4 --min-count 1,1,1,2 -o data/model_mix.bin data/corpus/egov.wakati data/corpus/wiki1.wakati
