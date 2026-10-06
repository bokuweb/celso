"""評価データの文を学習コーパスから除く (stdin → stdout)。

- JWTD (*.jsonl): Wikipedia の現行版には訂正後の文がそのまま含まれるため、除かないと評価が甘くなる。
- プレーンテキスト (横浜市市税条例など): 他市の例規に同じ文言の条文があるため、完全一致する文を除く。

使い方: python3 scripts/exclude_eval.py data/jwtd/test.jsonl [fixtures/yokohama_shizei_jorei.txt ...] < in.txt > out.txt
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from textnorm import norm  # noqa: E402

ban = set()


def add(text: str) -> None:
    for s in re.findall(r"[^。]+。?", norm(text)):
        s = s.strip()
        if len(s) >= 8:
            ban.add(s)


for path in sys.argv[1:]:
    for line in open(path, encoding="utf-8"):
        if path.endswith(".jsonl"):
            d = json.loads(line)
            add(d["pre_text"])
            add(d["post_text"])
        else:
            # 条例は「第1条 本文」のように空白でラベルと本文を分けるので、空白でも区切る
            for part in norm(line).split():
                add(part)
removed = 0
for line in sys.stdin:
    s = line.rstrip("\n")
    if s in ban or any(p in ban for p in re.findall(r"[^。]+。?", s)):
        removed += 1
        continue
    sys.stdout.write(line)
print(f"removed {removed} lines", file=sys.stderr)
