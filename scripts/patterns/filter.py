"""誤字パターンの第 2 段の絞り込み (select.py → celso count-patterns (分野のコーパス) → これ)。

1. 法令・例規・契約書 (契約書は 10 倍の重み) で誤り側がよく出るものを外す。
   Wikipedia では珍しくても、その分野では正しい書き方のものがある (「ユーザが仕様…」「貴社」)。
   基準: 分野での誤り側の出現が 2 回以下、または正しい側の 1/1000 以下、
         または 20 回以下かつ正しい側の 1/50 以下。
2. 送り仮名の違い (漢字の直後で、直す中身が平仮名だけのもの。「伴ない」「失なう」「話しを」) を外す。
   表記の揺れは対象外 (法令では古い送り仮名が正しいこともある)。

入力: 誤り \t 正しい \t 支持数 \t 全コーパスでの誤り側の出現数 \t 分野での誤り側 \t 分野での正しい側
出力: 誤り \t 正しい \t 支持数 (照合に使う列はこの 3 つ)
JWTD (CC BY-SA 3.0) から作るので、出力も CC BY-SA 3.0 で扱う。
"""
import sys


def is_kanji(c: str) -> bool:
    return '一' <= c <= '鿿' or c == '々'


def is_hiragana(s: str) -> bool:
    return all('ぁ' <= c <= 'ゖ' for c in s)


def okurigana(w: str, r: str) -> bool:
    p = 0
    while p < min(len(w), len(r)) and w[p] == r[p]:
        p += 1
    s = 0
    while s < len(w) - p and s < len(r) - p and w[len(w) - 1 - s] == r[len(r) - 1 - s]:
        s += 1
    a, b = w[p:len(w) - s], r[p:len(r) - s]
    return p > 0 and is_kanji(w[p - 1]) and is_hiragana(a) and is_hiragana(b)


NOTICE = (
    '# celso の誤字パターン。京都大学 日本語Wikipedia入力誤りデータセット (JWTD v2) の train から作成した。\n'
    '# 元データ: https://nlp.ist.i.kyoto-u.ac.jp/?日本語Wikipedia入力誤りデータセット (CC BY-SA 3.0)\n'
    '# このファイルも CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/deed.ja) で提供する。\n'
)


def main(src: str, out: str) -> None:
    kept = 0
    with open(out, 'w', encoding='utf-8') as f:
        # 配布するファイル自体に出典とライセンスを書く (タブを含まない行は読み込み時に読み飛ばされる)
        f.write(NOTICE)
        for line in open(src, encoding='utf-8'):
            p = line.rstrip('\n').split('\t')
            w, r, sup, dw, dr = p[0], p[1], p[2], int(p[4]), int(p[5])
            if not (dw <= 2 or dw * 1000 <= dr or (dw <= 20 and dw * 50 <= dr)):
                continue
            if okurigana(w, r):
                continue
            f.write(f'{w}\t{r}\t{sup}\n')
            kept += 1
    print(f'{kept} patterns', file=sys.stderr)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
