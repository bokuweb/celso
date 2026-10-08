"""文字単位の検出で漢字を置き換える候補の表 (kanji_homo.tsv) を作る。

1. IPADIC の 1 字の漢字の読みと、2 字の熟語の読みから推した各字の読み (音読み・訓読みの一部) で、
   同じ読みを持つ漢字を候補にする (出現数の多い順に 20 字まで)
2. 京大 JWTD v2 の学習用の差分で、2 回以上あった 1 字の漢字の取り違え (「後 → 跡」「型 → 形」) を足す。
   読みの表に無い取り違え (「破れる → 敗れる」「象 → 像」) を拾うため

2 の分は JWTD (CC BY-SA 3.0) の二次的著作物なので、ファイル先頭に出典とライセンスを書く。

使い方: python3 scripts/charlm/kanji_homo.py IPADIC の lex.csv JWTD の train.jsonl 出力 文字コーパス(.ch)...
"""
import collections
import json
import sys
import unicodedata


# 暦・時刻の単位
UNITS = set('年月日週時分秒')


def kan(c):
    return '一' <= c <= '鿿' or c == '々'


def main(lex_path, jwtd_path, out_path, *corpora):
    single = collections.defaultdict(set)
    lex = [l.rstrip('\n').split(',') for l in open(lex_path, encoding='utf-8')]
    for c in lex:
        if len(c) >= 12 and len(c[0]) == 1 and kan(c[0]) and c[11] != '*':
            single[c[0]].add(c[11])
    # 2 字の熟語の読みから、片方の字の読みを引いた残りをもう片方の字の読みとみなす (3 語以上で見たものだけ)
    cnt = collections.Counter()
    for _ in range(2):
        for c in lex:
            if len(c) >= 12 and len(c[0]) == 2 and all(kan(x) for x in c[0]) and c[11] != '*':
                a, b = c[0]
                r = c[11]
                for ra in list(single.get(a, ())):
                    if r.startswith(ra) and len(r) > len(ra):
                        cnt[(b, r[len(ra):])] += 1
                for rb in list(single.get(b, ())):
                    if r.endswith(rb) and len(r) > len(rb):
                        cnt[(a, r[:-len(rb)])] += 1
    rd = collections.defaultdict(set)
    for k, v in single.items():
        rd[k] |= v
    for (k, r), n in cnt.items():
        if n >= 3:
            rd[k].add(r)
    # 字の出現数 (コーパスの 5 行に 1 行から)
    freq = collections.Counter()
    for f in corpora:
        with open(f, encoding='utf-8') as fh:
            for i, l in enumerate(fh):
                if i % 5 == 0:
                    freq.update(c for c in l if kan(c))
    by_r = collections.defaultdict(set)
    for k, rs in rd.items():
        for r in rs:
            by_r[r].add(k)
    table = {}
    for k in rd:
        if freq[k] < 20:
            continue
        alts = set()
        for r in rd[k]:
            alts |= by_r[r]
        alts.discard(k)
        alts = [a for a in sorted(alts, key=lambda a: -freq[a]) if freq[a] >= 50][:20]
        if alts:
            table[k] = alts
    # JWTD の学習用の差分にある 1 字の漢字の取り違え
    pairs = collections.Counter()
    for l in open(jwtd_path, encoding='utf-8'):
        j = json.loads(l)
        for d in j['diffs']:
            if d.get('category') == 'not-typo':
                continue
            a = unicodedata.normalize('NFKC', d['pre_str'])
            b = unicodedata.normalize('NFKC', d['post_str'])
            if len(a) == len(b) and a:
                dif = [(x, y) for x, y in zip(a, b) if x != y]
                if len(dif) == 1 and kan(dif[0][0]) and kan(dif[0][1]):
                    pairs[dif[0]] += 1
    for (a, b), v in pairs.most_common():
        # 年月日・時分秒どうしの直し (「1988月 → 1988年」) は、事実の訂正 (「来月 → 来年」) と区別できず、
        # 正しい文で誤検出になる (どちらも正しい語) ので候補にしない
        if a in UNITS and b in UNITS:
            continue
        if v >= 2 and b not in table.setdefault(a, []):
            table[a].append(b)
    with open(out_path, 'w', encoding='utf-8') as out:
        out.write('# 漢字 \\t 置き換える候補。読みの表 (IPADIC) と、京大 JWTD v2 の学習用の差分の 1 字の取り違え\n')
        out.write('# (CC BY-SA 3.0, https://nlp.ist.i.kyoto-u.ac.jp/?日本語Wikipedia入力誤りデータセット) から作った二次的著作物 (CC BY-SA 3.0)\n')
        for k in sorted(table):
            if table[k]:
                out.write(k + '\t' + ''.join(table[k]) + '\n')
    print(f'{len(table)} kanji', file=sys.stderr)


if __name__ == '__main__':
    main(*sys.argv[1:])
