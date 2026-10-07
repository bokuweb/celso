"""文法モデル (data/func.bin) の語彙を作る: 機能語 (助詞・助動詞・記号・非自立語など) の表層形だけ。

文法モデルは、内容語を品詞クラスにまとめ、機能語だけ表層形で残した 5-gram。単語 3-gram (前後 2 語) より
長い範囲の助詞の並びを見るためのもので、判定器 (data/rerank.tsv) の特徴量に使う。
使い方: python3 scripts/func_vocab.py data/vocab_func.txt data/corpus/egov.wc data/corpus/reiki.wc ...
"""
import collections
import itertools
import sys

KEEP = ('<助詞', '<助動詞', '<記号', '<名詞-非自立', '<動詞-非自立', '<名詞-接尾', '<接続詞', '<連体詞', '<形容詞-非自立')


def main(out, *paths, limit=3_000_000, min_count=300):
    cnt = collections.Counter()
    for path in paths:
        with open(path, encoding='utf-8') as f:
            for line in itertools.islice(f, limit):
                for w in line.rstrip('\n').split(' '):  # str.split() は \x1f も空白扱いするので使わない
                    if '\x1f' not in w:
                        continue
                    s, c = w.split('\x1f', 1)
                    if c.startswith(KEEP):
                        cnt[s] += 1
    vocab = sorted(s for s, n in cnt.items() if n >= min_count)
    open(out, 'w', encoding='utf-8').write('\n'.join(vocab) + '\n')
    print(f'{len(vocab)} words', file=sys.stderr)


if __name__ == '__main__':
    main(*sys.argv[1:])
