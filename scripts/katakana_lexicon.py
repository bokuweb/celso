#!/usr/bin/env python3
"""カタカナ語の出現数の表 (data/katakana.tsv) を作る。カタカナ語の打ち間違いの検出 (src/katakana.rs) に使う。

学習コーパス (Wikipedia 1 本目・e-Gov 法令・例規集・契約書) から、カタカナ (と長音) の 2 字以上の並びを数え、
4 字以上で 3 回以上出てくるものを「語 \t 出現数」で書き出す。モデルは語と出現数だけを持ち、元の文は含まない。

使い方: python3 scripts/katakana_lexicon.py data/katakana.tsv data/corpus/wiki1.txt data/corpus/egov.txt ...
"""
import collections
import re
import sys

RUN = re.compile(r'[ァ-ヺー]{4,}')


def main(out, paths):
    counts = collections.Counter()
    for p in paths:
        with open(p, encoding='utf-8') as f:
            for line in f:
                counts.update(RUN.findall(line))
    rows = sorted((w, c) for w, c in counts.items() if c >= 3)
    with open(out, 'w', encoding='utf-8') as f:
        for w, c in rows:
            f.write(f'{w}\t{c}\n')
    print(f'{len(rows)} words', file=sys.stderr)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
