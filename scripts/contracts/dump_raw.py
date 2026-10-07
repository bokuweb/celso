"""files/ の各文書から生テキストを rawtext/<name>.txt に出す (ページ区切りは \f)。"""
import os, subprocess, sys
from pypdf import PdfReader
# 取得物・中間ファイルはリポジトリの外 (既定 data/contracts_raw/、gitignore 済み) に置く
ROOT = os.environ.get("CONTRACTS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "contracts_raw")
os.makedirs(ROOT, exist_ok=True)
src, dst = os.path.join(ROOT, "files"), os.path.join(ROOT, "rawtext")
for n in sorted(os.listdir(src)):
    out = os.path.join(dst, n + ".txt")
    if os.path.exists(out) and "-f" not in sys.argv:
        continue
    p = os.path.join(src, n)
    if n.endswith(".pdf"):
        try:
            r = PdfReader(p)
            pages = [(pg.extract_text() or "") for pg in r.pages]
        except Exception as e:  # noqa: BLE001
            print("FAIL", n, e); continue
        text = "\f".join(pages)
        npages = len(pages)
    elif n.endswith((".doc", ".docx", ".rtf")):
        subprocess.run(["textutil", "-convert", "txt", "-output", out, p], check=True)
        text = open(out, encoding="utf-8").read()
        npages = 0
    else:
        continue
    open(out, "w", encoding="utf-8").write(text)
    print(f"{n}\tpages={npages}\tchars={len(text)}")
