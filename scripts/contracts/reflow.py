"""PDF から取り出した契約書の本文を、Word の段落に近い形へ整える (評価用)。

PDF の本文は文の途中で改行されている (「そこで使用さ⏎れる画面」)。celso は改行を文の区切りとして
扱うので、そのままでは誤検出と取りこぼしの両方が水増しされる。
1. かな・漢字・数字に挟まれた空白を詰める (「第 20 条」→「第20条」)。行頭の字下げは残す
2. 文の途中の改行をつなぐ (前の行が句点等で終わっておらず、次の行が見出し・番号で始まらない場合)

使い方: python3 scripts/contracts/reflow.py 入力.txt 出力.txt
"""
import re
import sys

JA = r'[ぁ-んァ-ヺー一-龥々0-9０-９]'
HIRA = re.compile(r'[ぁ-ゖ]')
LABEL = re.compile(
    r'^\s*(第\s*[0-9０-９一二三四五六七八九十○]+\s*[条項号章節]|[（(][0-9０-９一二三四五六七八九十a-zア-ン]+[）)]'
    r'|[0-9０-９]+[．.]|[①-⑳]|・|※|【|■|●|○|)'
)


def ja(c: str) -> bool:
    return bool(re.match(r'[ぁ-ゖァ-ヺー一-龥々、0-9０-９]', c))


def squeeze(line: str) -> str:
    head = re.match(r'^[ 　]*', line).group(0)
    body = line[len(head):]
    prev = None
    while prev != body:
        prev = body
        body = re.sub(rf'(?<={JA})[ 　]+(?={JA})', '', body)
    return head + body


def reflow(text: str) -> str:
    out: list[str] = []
    for line in (squeeze(l) for l in text.split('\n')):
        s = line.strip()
        if (
            out and s and out[-1] and ja(out[-1][-1]) and not LABEL.match(line) and ja(s[0])
            and (HIRA.match(out[-1][-1]) or out[-1][-1] == '、' or HIRA.match(s[0]) or len(out[-1].strip()) >= 30)
        ):
            out[-1] += s
        else:
            out.append(line.rstrip())
    return '\n'.join(out)


if __name__ == '__main__':
    open(sys.argv[2], 'w', encoding='utf-8').write(reflow(open(sys.argv[1], encoding='utf-8').read()))
