# 契約書コーパス

官公庁・公的機関が公開しているモデル契約書・標準約款・契約ガイドラインから、学習用の契約書コーパスを作る。

```bash
# 1. 取得 (robots.txt を守り、同一ホストへは 1.1 秒以上の間隔。取得物は data/contracts_raw/files/)
for b in batch1 batch2 batch3 batch4; do python3 fetch.py batch $b.tsv; done
# 2. 本文の取り出し (PDF / Word → data/contracts_raw/rawtext/)
uv run --with pypdf --with cryptography python dump_raw.py
# 3. 整形して data/contracts_raw/contracts.txt と contracts_ipa_model.txt を作る
python3 build_contracts.py --out ../../data/corpus/contracts.txt
```

- `SOURCES.tsv` / `STATS.tsv` は 2026-10-07 に取得したときの出典と文字数 (参考)。
- 評価に使う JEITA のソフトウェア開発モデル契約と IPA のアジャイル開発モデル契約は、取得・抽出の両方で拒否する。
- IPA「情報システム・モデル取引・契約書」は別ファイル (`contracts_ipa_model.txt`) に出し、
  評価データと先頭 30 字が一致する行を除いてから学習に足す (`scripts/build_all.sh`)。
- 経産省・特許庁・中小企業庁のサイトは WAF ですべて 403 になるため取得していない。
