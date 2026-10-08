#!/usr/bin/env python3
"""判例要旨 (1 行 1 件のテキスト) を、評価用 (dev / test 各 3000 件) と学習用に分ける。

判例要旨は社内で保有するデータで、このリポジトリには含めない。scripts/build_all.sh に
CELSO_HANREI_TEXT でテキストを渡したときだけ使う (取り出し方はデータの持ち主の手順に従う)。
並べ替えの乱数の種を固定して、評価に使う件数を学習から必ず除く。

使い方: python3 scripts/hanrei_split.py 判例要旨.txt 出力フォルダ
出力: dev.txt / test.txt / train.txt / synth_src.txt (判定器の人工誤り用。学習用の先頭 4 万件)
"""
import os
import random
import sys


def main(src, out):
    lines = [l for l in open(src, encoding='utf-8').read().split('\n') if l]
    random.seed(7)
    random.shuffle(lines)
    os.makedirs(out, exist_ok=True)
    parts = {'dev': lines[:3000], 'test': lines[3000:6000], 'train': lines[6000:]}
    for name, rows in parts.items():
        with open(os.path.join(out, f'{name}.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(rows) + '\n')
    with open(os.path.join(out, 'synth_src.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts['train'][:40000]) + '\n')
    print({k: len(v) for k, v in parts.items()}, file=sys.stderr)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
