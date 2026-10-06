"""評価データ (JWTD test) の文を学習コーパスから除く (stdin → stdout)。

Wikipedia の現行版には JWTD の訂正後の文がそのまま含まれるため、除かないと評価が甘くなる。
使い方: python3 scripts/exclude_eval.py data/jwtd/test.jsonl < wiki.txt > wiki.filtered.txt
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from textnorm import norm  # noqa: E402

ban = set()
for line in open(sys.argv[1], encoding="utf-8"):
    d = json.loads(line)
    for k in ("pre_text", "post_text"):
        for s in re.findall(r"[^。]+。?", norm(d[k])):
            s = s.strip()
            if len(s) >= 8:
                ban.add(s)
removed = 0
for line in sys.stdin:
    s = line.rstrip("\n")
    if s in ban:
        removed += 1
        continue
    sys.stdout.write(line)
print(f"removed {removed} lines", file=sys.stderr)
