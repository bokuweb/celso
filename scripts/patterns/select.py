"""誤字パターンの候補から、誤検出しにくいものを選ぶ (extract.py → celso count-patterns → これ)。

基準:
  - train での支持数が min_support 以上 (偶然の 1 件を拾わない。抽出の時点で 2 件以上に絞っている)
  - コーパス (法令・例規・契約書・Wikipedia) で、誤り側の出現が正しい側の 1/ratio 以下
支持数の少ないパターンは文脈によって当たり外れがあるので、検査時に直した文の n-gram の改善幅と
支持数で採否を決める (src/checker.rs の pattern_accepted)。そのぶんここでは広めに採る
(支持数 3・比 200 から 2・50 にして、JWTD の開発用で検出 +3pt、誤検出は採否の絞り込みで相殺)。
    (正しい文でもよく使う書き方なら、誤りとは言えない)
  - 文脈の幅が違う同じ書き換えのうち、条件を満たす一番短いものだけを残す (表を小さく、照合を速く)

出力 (TSV): 誤り \t 正しい \t 支持数 \t 誤りの出現数 \t 正しい側の出現数 \t カテゴリ
JWTD (CC BY-SA 3.0) から作るので、出力も CC BY-SA 3.0 で扱う。
"""
import sys

def main(src, out, min_support=2, ratio=50, max_wrong=50, min_right=10):
    rows = []
    for line in open(src, encoding='utf-8'):
        p = line.rstrip('\n').split('\t')
        if len(p) < 6:
            continue
        w, r, sup, cat, wc, rc = p[0], p[1], int(p[2]), p[3], int(p[4]), int(p[5])
        if sup < min_support or rc < min_right or wc > max_wrong or wc * ratio > rc:
            continue
        # 空白・ASCII だけの書き換え、数字を含むものは除く (表記の揺れや固有の数値が多い)
        if w.strip() == '' or all(ord(c) < 128 for c in w + r) or any(c.isdigit() for c in w + r):
            continue
        rows.append((w, r, sup, wc, rc, cat))
    # 短い順に採用し、採用済みの書き換えを含む長いものは捨てる
    rows.sort(key=lambda x: (len(x[0]), -x[2]))
    kept = []
    kept_pairs = []
    for w, r, sup, wc, rc, cat in rows:
        if any(kw in w and kr in r and w.replace(kw, kr, 1) == r for kw, kr in kept_pairs):
            continue
        kept.append((w, r, sup, wc, rc, cat))
        kept_pairs.append((w, r))
    with open(out, 'w', encoding='utf-8') as f:
        for row in sorted(kept, key=lambda x: -x[2]):
            f.write('\t'.join(map(str, row)) + '\n')
    print(f'{len(kept)} patterns', file=sys.stderr)

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args[0], args[1], *(int(a) for a in args[2:]))
