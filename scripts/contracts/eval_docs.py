"""契約書の評価データ (学習には使わない) を取得して整形する。

- JEITA ソフトウェア開発モデル契約 (解説付き) 2 本: 大きい方を閾値の調整 (開発用)、小さい方をテストに使う
- IPA アジャイル開発外部委託モデル契約: テストに使う

使い方: uv run --with pypdf --with cryptography python scripts/contracts/eval_docs.py data/eval_contracts
"""
import os
import re
import sys
import urllib.request

from pypdf import PdfReader

DOCS = {
    "jeita_dev": "https://home.jeita.or.jp/upload_file/20190416115959_WkDZ82m54J.pdf",
    "jeita_test": "https://home.jeita.or.jp/upload_file/20190415171923_6nPOjkV5Av.pdf",
    "ipa_agile_test": "https://www.ipa.go.jp/digital/model/ug65p90000001ldr-att/000081484.pdf",
}
JP = r"[ぁ-んァ-ヶー一-龥々、。，．（）「」『』・]"
WS = r"[\s　    ]"

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for name, url in DOCS.items():
    pdf = os.path.join(out, name + ".pdf")
    if not os.path.exists(pdf):
        req = urllib.request.Request(url, headers={"User-Agent": "celso-contracts-research"})
        with urllib.request.urlopen(req) as r, open(pdf, "wb") as f:
            f.write(r.read())
    text = "\n".join((p.extract_text() or "") for p in PdfReader(pdf).pages)
    # 日本語の文字に挟まれた空白・改行 (PDF の行折り返しと字間の空白) をつなぐ
    text = re.sub(rf"(?<={JP}){WS}+(?={JP})", "", text)
    lines = [l.strip() for l in text.split("\n")]
    keep = [l for l in lines if len(l) >= 15 and len(re.findall(r"[ぁ-ん]", l)) >= 5]
    with open(os.path.join(out, name + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(keep) + "\n")
    print(name, len(keep), sum(len(l) for l in keep))
