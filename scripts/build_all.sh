#!/usr/bin/env bash
# 辞書・コーパス取得 → 抽出 → 分かち書き → 言語モデル構築までを一通り行う。
# 必要ディスク: 約 12GB / メモリ: 約 10GB / 所要: 2 時間前後 (例規集の取得が大半)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/corpus
cargo build --release
BIN=./target/release/celso

# 0. 形態素解析辞書 (mecab-ipadic を UTF-8 化して delarocha のバイナリ辞書へ)
if [ ! -f data/ipadic.dic ]; then
  curl -sL -o data/mecab-ipadic.tar.gz https://lindera.dev/mecab-ipadic-2.7.0-20070801.tar.gz
  tar xzf data/mecab-ipadic.tar.gz -C data
  mkdir -p data/ipadic-utf8
  cat data/mecab-ipadic-2.7.0-20070801/*.csv | iconv -f EUC-JP -t UTF-8 > data/ipadic-utf8/lex.csv
  for f in matrix.def char.def unk.def; do
    iconv -f EUC-JP -t UTF-8 "data/mecab-ipadic-2.7.0-20070801/$f" > "data/ipadic-utf8/$f"
  done
  $BIN build-dict data/ipadic-utf8 -o data/ipadic.dic
fi
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
# 4. 2 段目の MLM (修正案選び用)
if [ ! -d data/mlm ]; then
  mkdir -p data/mlm
  for f in config.json model.safetensors tokenizer.json; do
    curl -sL -o "data/mlm/$f" "https://huggingface.co/sbintuitions/modernbert-ja-30m/resolve/main/$f"
  done
fi

python3 scripts/extract_egov.py data/egov > data/corpus/egov.txt
python3 scripts/extract_wiki.py data/jawiki1.xml.bz2 \
  | python3 scripts/exclude_eval.py data/jwtd/test.jsonl data/jwtd/gold.jsonl > data/corpus/wiki1.f.txt
# 5. 自治体の例規集 (30 自治体, robots.txt 準拠・低頻度で取得。横浜市は評価用なので含めない)
#    他市にも横浜市市税条例と同じ文言の条文があるので、評価が甘くならないよう完全一致する文は除く
python3 scripts/fetch_reiki.py
python3 scripts/exclude_eval.py fixtures/yokohama_shizei_jorei.txt < data/corpus/reiki.txt > data/corpus/reiki.f.txt

# 6. 分かち書き (各トークンを「表層形\x1f品詞クラス」で出し、語彙は後から選ぶ)
for f in egov wiki1.f reiki.f; do
  $BIN tokenize --with-class < "data/corpus/$f.txt" > "data/corpus/${f%.f}.wc"
done
cat data/corpus/egov.txt data/corpus/wiki1.f.txt data/corpus/reiki.f.txt \
  | $BIN tokenize --inflections data/inflections.tsv --readings data/readings.tsv > /dev/null

# 7. 語彙 1 万語 + 品詞クラス、3-gram、強い足切りで配布用モデル (約 13MB) を作る
$BIN vocab --size 10000 -o data/vocab10k.txt data/corpus/egov.wc data/corpus/wiki1.wc data/corpus/reiki.wc
$BIN build-lm --order 3 --min-count 1,5,10 --vocab data/vocab10k.txt -o data/model.bin \
  data/corpus/egov.wc data/corpus/wiki1.wc data/corpus/reiki.wc
# 8. 活用表・同音異字表をモデルの語彙で絞る (配布物は data/dist/)
$BIN prune-tables --model data/model.bin -o data/dist
cp data/model.bin data/dist/model.bin
