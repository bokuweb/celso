"""語彙 (上位 N 語) に、同音異字の組になる語を足す。

語彙を 1 万語に絞ると、誤変換の候補 (対象/対照 など) にできる語が減る。
同じ読みに出現 100 回以上の表記が 2 つ以上ある語だけを足す (約 5 千語、モデルは +0.7MB)。

使い方: python3 scripts/homophone_vocab.py data/vocab10k.txt data/readings.tsv > data/vocab.txt
"""
import collections
import sys

MIN_COUNT = 100

vocab = [w for w in open(sys.argv[1], encoding="utf-8").read().split("\n") if w]
seen = set(vocab)
by_reading = collections.defaultdict(list)
for line in open(sys.argv[2], encoding="utf-8"):
    r, s, c = line.rstrip("\n").split("\t")
    by_reading[r].append((s, int(c)))
extra = {}
for v in by_reading.values():
    v = [x for x in v if x[1] >= MIN_COUNT]
    if len(v) >= 2:
        for s, c in v:
            if s not in seen:
                extra[s] = max(extra.get(s, 0), c)
for w in vocab + sorted(extra, key=lambda s: -extra[s]):
    print(w)
