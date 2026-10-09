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

# 5b. 契約書 (官公庁などのモデル契約書・標準約款・ガイドライン)。手順は scripts/contracts/README.md
#     評価に使う JEITA / IPA アジャイル開発版は含めない。IPA モデル取引・契約書は評価データと重なる行を除いて足す
uv run --with pypdf --with cryptography python scripts/contracts/eval_docs.py data/eval_contracts
# 評価・判定器の学習には、文中の改行をつないだ版を使う (scripts/contracts/reflow.py)
for n in jeita_dev jeita_test ipa_agile_test; do
  python3 scripts/contracts/reflow.py data/eval_contracts/$n.txt data/eval_contracts/$n.reflow.txt
done
(cd scripts/contracts && for b in batch1 batch2 batch3 batch4; do python3 fetch.py batch $b.tsv; done \
  && uv run --with pypdf --with cryptography python dump_raw.py \
  && python3 build_contracts.py --out ../../data/corpus/contracts.txt)
python3 - <<'PY'
import re
ev = re.sub(r"\s", "", "".join(open(f"data/eval_contracts/{n}.txt").read() for n in ["jeita_dev", "jeita_test", "ipa_agile_test"]))
out = [l.rstrip("\n") for l in open("data/contracts_raw/contracts_ipa_model.txt")
       if re.sub(r"\s", "", l) and not (len(re.sub(r"\s", "", l)) >= 20 and re.sub(r"\s", "", l)[:30] in ev)]
open("data/corpus/contracts2.txt", "w").write(open("data/corpus/contracts.txt").read() + "\n".join(out) + "\n")
PY

# 6. 分かち書き (各トークンを「表層形\x1f品詞クラス」で出し、語彙は後から選ぶ)
for f in egov wiki1.f reiki.f contracts2; do
  $BIN tokenize --with-class < "data/corpus/$f.txt" > "data/corpus/${f%.f}.wc"
done
# 契約書は小さい (約 180 万字) ので 10 倍の重みで数える (30 倍では契約書での誤検出が増えた)
INPUTS=(data/corpus/egov.wc data/corpus/wiki1.wc data/corpus/reiki.wc)
for i in $(seq 1 10); do INPUTS+=(data/corpus/contracts2.wc); done
# 判例要旨 (任意): 社内で保有するデータで、このリポジトリには含めない。1 行 1 件のテキストを
# CELSO_HANREI_TEXT で渡すと、言語モデルと判定器の学習に加える (語彙・共起モデルは変えない)。
# 判決文の要旨での誤検出が 1 万字あたり 8.1 → 3.3 件に減る (学習から除いた 3000 件)。3 倍の重みで数える
LM_INPUTS=("${INPUTS[@]}")
HANREI_DIR=data/corpus/hanrei
if [ -n "${CELSO_HANREI_TEXT:-}" ]; then
  python3 scripts/hanrei_split.py "$CELSO_HANREI_TEXT" $HANREI_DIR
  $BIN tokenize --with-class < $HANREI_DIR/train.txt > $HANREI_DIR/train.wc
  for i in 1 2 3; do LM_INPUTS+=($HANREI_DIR/train.wc); done
fi
cat data/corpus/egov.txt data/corpus/wiki1.f.txt data/corpus/reiki.f.txt \
  | $BIN tokenize --inflections data/inflections.tsv --readings data/readings.tsv > /dev/null

# 7. 語彙 1 万語 + 品詞クラス、3-gram、強い足切りで配布用モデル (約 13MB) を作る
$BIN vocab --size 10000 -o data/vocab10k.txt "${INPUTS[@]}"
#    誤変換の候補にする同音異字の語を足す (約 5 千語、+0.7MB)
python3 scripts/homophone_vocab.py data/vocab10k.txt data/readings.tsv data/homo_words.txt > data/vocab.txt
$BIN build-lm --order 3 --min-count 1,5,10 --vocab data/vocab.txt -o data/model.bin "${LM_INPUTS[@]}"
# 8. 同音異字の判定に使う文内共起モデル (約 3.7MB)。手がかりの語は頻出 5 万語から選ぶ。
#    集計 (data/cooc_stats.bin) を残しておくと、--top-k などを変えて作り直すときにコーパスを数え直さない
$BIN vocab --size 50000 -o data/vocab50k.txt "${INPUTS[@]}"
$BIN build-cooc --model data/model.bin --vocab data/vocab.txt --homophones data/homo_words.txt \
  --ctx-vocab data/vocab50k.txt --stats data/cooc_stats.bin --top-k 128 --min-pair 3 --weight-by-count \
  -o data/cooc.bin "${INPUTS[@]}"
# 9. 文法モデル (機能語だけ表層形・内容語は品詞クラスの 5-gram、約 2MB)。判定器の特徴量に使う
python3 scripts/func_vocab.py data/vocab_func.txt data/corpus/egov.wc data/corpus/wiki1.wc data/corpus/reiki.wc data/corpus/contracts2.wc
FUNC_INPUTS=(data/corpus/egov.wc data/corpus/reiki.wc); for i in $(seq 1 10); do FUNC_INPUTS+=(data/corpus/contracts2.wc); done
$BIN build-lm --order 5 --vocab data/vocab_func.txt --min-word-count 1 --min-count 1,5,20,50,100 -o data/func.bin "${FUNC_INPUTS[@]}"

# 10. 誤字パターン (JWTD train の実際の誤字から。CC BY-SA 3.0)
#     候補の抽出 → 全コーパスでの出現数 → 1 段目の選別 → 分野のコーパスでの出現数 → 2 段目 (分野・送り仮名)
#     調整に使う JWTD 先頭 5000 件 (開発用) を含めると開発用の誤りがそのままパターンになり、開発用の検出が 1.5pt 甘く出る。
#     開発用を除いた train_rest から作る (test の結果は変わらない)
python3 scripts/patterns/extract.py data/jwtd/train_rest.jsonl data/patterns_cand.tsv
$BIN count-patterns data/patterns_cand.tsv -o data/patterns_counted.tsv \
  data/corpus/egov.txt data/corpus/reiki.f.txt data/corpus/wiki12.f.txt data/corpus/contracts2.txt
python3 scripts/patterns/select.py data/patterns_counted.tsv data/patterns_sel.tsv
cut -f1-4 data/patterns_sel.tsv > data/patterns_sel4.tsv
DOMAIN=(data/corpus/egov.txt data/corpus/reiki.f.txt); for i in $(seq 1 10); do DOMAIN+=(data/corpus/contracts2.txt); done
$BIN count-patterns data/patterns_sel4.tsv -o data/patterns_domain.tsv "${DOMAIN[@]}"
python3 scripts/patterns/filter.py data/patterns_domain.tsv data/patterns.tsv

# 10b. カタカナ語の出現数 (カタカナ語の打ち間違いの検出。src/katakana.rs)
python3 scripts/katakana_lexicon.py data/katakana.tsv \
  data/corpus/wiki1.txt data/corpus/egov.txt data/corpus/reiki.f.txt data/corpus/contracts2.txt

# 11. 判定器 (ロジスティック回帰)。学習には評価に使わない文書だけを使う:
#     学習に入れていない 2 自治体 (一宮市・高槻市) の例規、JEITA モデル契約 (大、調整用)、JWTD train (先頭 5000 件は開発用に除く)
python3 scripts/fetch_reiki.py --all --sites 高槻市,加古川市,一宮市,生駒市 --max-pages 150 \
  --cache ~/celso-data/reiki_heldout_html --out data/corpus/reiki_heldout.txt
for c in 一宮市 高槻市 加古川市 生駒市; do
  python3 scripts/fetch_reiki.py --all --extract-only --sites $c --cache ~/celso-data/reiki_heldout_html --out data/corpus/heldout_$c.raw.txt
done
# 学習コーパスと完全一致する行 (各地で同じ条文) は除く: scripts/heldout_dedup.py
python3 scripts/heldout_dedup.py
tail -n +5001 data/jwtd/train.jsonl > data/jwtd/train_rest.jsonl
$BIN dump-rerank --aux-model data/func.bin --patterns none \
  --synth data/corpus/heldout_一宮市.txt:legal:600 --synth data/corpus/heldout_高槻市.txt:legal:600 \
  --synth data/eval_contracts/jeita_dev.reflow.txt:contract:600 \
  --jwtd data/jwtd/train_rest.jsonl --jwtd-limit 30000 --floor=0 -o data/rerank_train.tsv
# 一般文の人工誤り (Wikipedia の 2 本目のダンプ。言語モデルには入れていない) も足す。JWTD だけだと
# 一般文の余計な助詞 (「土産物から店」) の正例が少なく、判定器が助詞の削除を強く嫌う
python3 scripts/heldout_wiki2.py
$BIN dump-rerank --aux-model data/func.bin --patterns none \
  --synth data/corpus/heldout_wiki2_train.txt:general:3000 --floor=0 -o data/rerank_train_wiki2.tsv
# 足切り (n-gram の Δ) は 0。1 にすると、変換ミス・助詞の脱落の正解の 3 分の 1 ほどが判定器に届かなかった
RERANK_INPUTS=(data/rerank_train.tsv data/rerank_train_wiki2.tsv)
# 判例要旨を使うときは、判決文の人工誤り (学習用の先頭 4 万件から 2000 件 x 5 種類) も足す
if [ -n "${CELSO_HANREI_TEXT:-}" ]; then
  $BIN dump-rerank --aux-model data/func.bin --patterns none \
    --synth $HANREI_DIR/synth_src.txt:general:2000 --floor=0 -o data/rerank_train_hanrei.tsv
  RERANK_INPUTS+=(data/rerank_train_hanrei.tsv)
fi
$BIN train-rerank "${RERANK_INPUTS[@]}" --floor=0 -o data/rerank.tsv
# 閾値 (対数オッズ) は調整用データ (JWTD 先頭 5000 件・wiki2 test・一宮市・JEITA 大) で決めた値を書き込む:
# 法令文 -1.5、一般文 0.0、契約書 0.5 (一般文は、誤字パターン・カタカナ語の検出を足したぶん厳しくして
# 正しい文での誤検出を以前と同じ水準に保つ)
# 活用の誤り (「多くあろう → ある」) は JWTD の言い換え (〜であろう) に引きずられて判定器が強く負に振れるので、
# 判定器を使わず種類ごとの閾値で決める。一般文の削除は、正しい文の助詞を消す誤検出 (「被害[が]軽減」
# 「状態[を]関数」) が多いので 0.15 と厳しくし、判定器のスコアが低く出やすい実際の誤りだけ別の閾値に残す:
# 名詞と 1 字の名詞の間の助詞 (「和菓子[は]店」、-1.3 前後に出る) は -1.35 (delete-nsfx)、係助詞の直前の助詞
# (「西口側[まで]は」、-0.8 前後) は -0.95 (delete-pp)、名詞の間の「の」(「無料のシャトルバス」) は -0.5 (delete-gen)。
# 一般文の基本の閾値 (取り違え・補い・活用) は 0.5。どれも tests/regression で固定し、JWTD 先頭 5000 件・
# 判例要旨 dev・wiki2 test の正しい文での誤検出がいずれも下がる範囲で、検出が最も伸びる値にした
# 判例要旨を使うときは、一般文の同音異字を -0.6 に下げる (判決文に寄ったぶん Wikipedia の漢字誤変換の検出が
# 38 → 34% に落ちたのを戻す。調整は JWTD 先頭 5000 件・判例要旨の dev で行った)
HANREI=${CELSO_HANREI_TEXT:+1} python3 - <<'PY'
p = 'data/rerank.tsv'
lines = [l for l in open(p).read().split('\n') if not l.startswith(('#exempt', '#tau_kind'))]
lines = ['#tau\t-1.5,0.5,0.5' if l.startswith('#tau\t') else l for l in lines]
i = next(k for k, l in enumerate(lines) if l.startswith('#floor'))
extra = ['#exempt\tinflection-aux', '#tau_kind\tdelete\t-1.5,0.15,0.5',
         '#tau_kind\tdelete-nsfx\t-1.5,-1.35,0.5', '#tau_kind\tdelete-pp\t-1.5,-0.95,0.5',
         '#tau_kind\tdelete-gen\t-1.5,-0.5,0.5']
import os
if os.environ.get('HANREI'):
    extra.append('#tau_kind\thomophone\t-1.5,-0.6,0.5')
lines[i + 1:i + 1] = extra
open(p, 'w').write('\n'.join(lines))
PY

# 12. 活用表・同音異字表をモデルの語彙で絞る (配布物は data/dist/)
$BIN prune-tables --model data/model.bin -o data/dist
cp data/model.bin data/cooc.bin data/func.bin data/rerank.tsv data/patterns.tsv data/katakana.tsv data/dist/

# 13. 文字単位の言語モデル (一般文の語の中の 1 字の誤り。src/charcheck.rs)。文字の 5-gram を足切りして約 46MB
#     (足切り 1,5,10,20,40 の 26MB 版は JWTD の検出が 0.7pt 低い)。契約書は単語モデルと同じく重みを足す
mkdir -p data/charlm
CH_INPUTS=()
for f in wiki1.f egov reiki.f contracts2; do
  python3 scripts/charlm/to_chars.py < data/corpus/$f.txt > data/charlm/$f.ch
  CH_INPUTS+=(data/charlm/$f.ch)
done
for i in 1 2 3 4; do CH_INPUTS+=(data/charlm/contracts2.ch); done
if [ -n "${CELSO_HANREI_TEXT:-}" ]; then
  python3 scripts/charlm/to_chars.py < $HANREI_DIR/train.txt > data/charlm/hanrei.ch
  CH_INPUTS+=(data/charlm/hanrei.ch)
fi
$BIN build-lm --order 5 --min-word-count 30 --min-count 1,3,6,10,16 -o data/charlm.bin "${CH_INPUTS[@]}"
#    漢字を置き換える候補 (読みの表 + JWTD の 1 字の取り違え。CC BY-SA 3.0)
python3 scripts/charlm/kanji_homo.py data/ipadic-utf8/lex.csv data/jwtd/train_rest.jsonl data/kanji_homo.tsv \
  data/charlm/wiki1.f.ch data/charlm/egov.ch ${CELSO_HANREI_TEXT:+data/charlm/hanrei.ch}
cp data/charlm.bin data/kanji_homo.tsv data/dist/

# 14. 文字単位の直しの判定器 (scripts/charrank/)。JWTD train_rest から 9.6 万組を抜き出し、候補を書き出して学習する。
#     負例には判例要旨の正しい文 (学習用から 6000 件、重み 2) も使う。閾値は直し方の種類ごと (一般文: 削除 -0.5・
#     補い 0.5・漢字の置き換え 0.1・かなの置き換え 0・入れ替え 0) で、JWTD 先頭 5000 件・判例要旨 dev・wiki2 test と
#     tests/regression で、単語モデルの閾値と合わせて決めた (補いは判例要旨で、漢字の置き換えは「呑み屋」のような
#     正しい表記で誤検出しやすい)
mkdir -p data/charrank
python3 scripts/charrank/make_data.py data/jwtd/train_rest.jsonl data/charrank
#    候補を出すときも配布時と同じひらがなに絞るため、出現数だけの判定器 (重みなし) を使う
sed 's/^/#prior\t/' data/charrank/prior.tsv > data/charrank/prior_only.tsv
TRAIN_SETS=$(for k in $(seq 0 15); do echo tr_pre$k tr_post$k; done)
if [ -n "${CELSO_HANREI_TEXT:-}" ]; then
  shuf -n 6000 --random-source=<(yes) $HANREI_DIR/train.txt > data/charrank/han_tr.txt
  TRAIN_SETS="$TRAIN_SETS han_tr"
fi
for n in $TRAIN_SETS; do
  scripts/charrank/dump.sh data/dist data/charrank/prior_only.tsv data/charrank/$n.txt data/charrank/$n.c
done
uv run --with scikit-learn python scripts/charrank/train.py data/charrank data/charrank data/charrank/prior.tsv 2 0.75 data/charrank.tsv \
  del=-0.5,ins=0.5,subk=0.1,subn=0,swap=0
cp data/charrank.tsv data/dist/
