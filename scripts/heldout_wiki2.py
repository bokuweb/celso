#!/usr/bin/env python3
"""言語モデルの学習に使っていない Wikipedia の 2 本目のダンプ (data/corpus/wiki2.txt) から、
判定器の学習用 (heldout_wiki2_train.txt, 2 万文) と評価用 (heldout_wiki2_test.txt, 1 万文) を抜く。

先頭から取ると記事が偏るので、60 行おきに全体からまんべんなく抜いてから混ぜる (乱数の種は固定)。
"""
import random

random.seed(5)
lines = []
with open('data/corpus/wiki2.txt', encoding='utf-8') as f:
    for i, l in enumerate(f):
        if i % 60 == 0:
            l = l.strip()
            if 20 <= len(l) <= 120:
                lines.append(l)
random.shuffle(lines)
with open('data/corpus/heldout_wiki2_train.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines[:20000]) + '\n')
with open('data/corpus/heldout_wiki2_test.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines[20000:30000]) + '\n')
print(len(lines))
