"""文字単位の直しの判定器の学習データを作る。

JWTD の学習用 (train_rest = 開発用の 5000 件を除いたもの) から文の組を抜き出し、誤りのある文 (pre) と
直した文 (post) を 1 行 1 文で書く。抜き出した文を除いた残りから「誤り → 正しい」の出現数 (prior.tsv) を数える
(学習に使う文の直しを数えると、その直しの出現数が必ず 1 以上になり、判定器が出現数に頼りすぎるため)。

使い方: python3 scripts/charrank/make_data.py train_rest.jsonl 出力ディレクトリ
"""
import collections
import json
import random
import sys
import unicodedata


def write_pairs(out, k, lines):
    with open(f'{out}/tr_pre{k}.txt', 'w') as a, open(f'{out}/tr_post{k}.txt', 'w') as b:
        for l in lines:
            j = json.loads(l)
            if any(d.get('category') == 'not-typo' for d in j['diffs']):
                continue
            a.write(j['pre_text'].replace('\n', ' ') + '\n')
            b.write(j['post_text'].replace('\n', ' ') + '\n')


def main(src, out):
    lines = open(src, encoding='utf-8').readlines()
    # 2 回に分けて抜き出した (最初の 24,000 組を 4 分割、残りから 72,000 組を 12 分割)。配布している判定器はこの分け方で作った
    random.seed(3)
    first = random.sample(lines, 24000)
    for k in range(4):
        write_pairs(out, k, first[k::4])
    first_set = set(first)
    rest = [l for l in lines if l not in first_set]
    random.seed(5)
    more = random.sample(rest, 72000)
    for k in range(12):
        write_pairs(out, k + 4, more[k::12])
    chosen = first_set | set(more)
    cnt = collections.Counter()
    for l in lines:
        if l in chosen:
            continue
        for d in json.loads(l)['diffs']:
            if d.get('category') == 'not-typo':
                continue
            a = unicodedata.normalize('NFKC', d['pre_str'])
            b = unicodedata.normalize('NFKC', d['post_str'])
            if len(a) <= 2 and len(b) <= 2 and (a or b):
                cnt[(a, b)] += 1
            # 同じ長さで 1 字だけ違う置き換えは、その 1 字の組としても数える
            if len(a) == len(b) > 1:
                dif = [(x, y) for x, y in zip(a, b) if x != y]
                if len(dif) == 1:
                    cnt[dif[0]] += 1
    with open(f'{out}/prior.tsv', 'w') as f:
        for (a, b), v in cnt.most_common():
            if v >= 2:
                f.write(f'{a}\t{b}\t{v}\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
