"""Wikipedia 日本語版ダンプ (pages-articles*.xml.bz2) から本文の文を抽出する。

wikitext を厳密に解釈する必要はなく、言語モデルの学習に使える「地の文」が取れればよい。
テンプレート・表・参照・見出し・箇条書きは落とし、句点で終わり平仮名を含む文だけを残す。

使い方: python3 scripts/extract_wiki.py data/jawiki1.xml.bz2 ... > data/corpus/wiki.txt
"""
import bz2
import html
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from textnorm import norm  # noqa: E402

RE_TEXT = re.compile(r"<text[^>]*>(.*?)</text>", re.S)
RE_NS = re.compile(r"<ns>(\d+)</ns>")
RE_REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.S)
RE_TAG = re.compile(r"<[^>]+>")
RE_COMMENT = re.compile(r"<!--.*?-->", re.S)
RE_LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]")
RE_EXT = re.compile(r"\[https?://[^\s\]]+\s?([^\]]*)\]")
RE_QUOTE = re.compile(r"'{2,}")
RE_HIRA = re.compile(r"[ぁ-ん]")
RE_SENT = re.compile(r"[^。]+。")


def strip_nested(text: str, open_: str, close: str) -> str:
    out, depth, i = [], 0, 0
    lo, lc = len(open_), len(close)
    while i < len(text):
        if text.startswith(open_, i):
            depth += 1
            i += lo
        elif depth and text.startswith(close, i):
            depth -= 1
            i += lc
        else:
            if depth == 0:
                out.append(text[i])
            i += 1
    return "".join(out)


def clean(wt: str) -> list[str]:
    wt = html.unescape(wt)
    wt = RE_COMMENT.sub("", wt)
    wt = RE_REF.sub("", wt)
    wt = strip_nested(wt, "{{", "}}")
    wt = strip_nested(wt, "{|", "|}")
    # ファイル・カテゴリなど名前空間付きリンクは丸ごと落とす
    wt = re.sub(r"\[\[(?:ファイル|画像|File|Image|Category|カテゴリ):[^\]]*(?:\[\[[^\]]*\]\][^\]]*)*\]\]", "", wt)
    wt = RE_LINK.sub(r"\1", wt)
    wt = RE_EXT.sub(r"\1", wt)
    wt = RE_QUOTE.sub("", wt)
    wt = RE_TAG.sub("", wt)
    out = []
    for line in wt.split("\n"):
        line = line.strip()
        if not line or line[0] in "=*#:;|!{}[":
            continue
        for s in RE_SENT.findall(line):
            s = s.strip()
            if 8 <= len(s) <= 300 and RE_HIRA.search(s) and "http" not in s:
                out.append(norm(s))
    return out


def pages(path: str):
    buf = []
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if "<page>" in line:
                buf = [line]
            else:
                buf.append(line)
                if "</page>" in line:
                    yield "".join(buf)
                    buf = []


def main() -> None:
    w = sys.stdout.write
    for path in sys.argv[1:]:
        n = 0
        for page in pages(path):
            m = RE_NS.search(page)
            if not m or m.group(1) != "0":
                continue
            t = RE_TEXT.search(page)
            if not t or t.group(1).lstrip().upper().startswith("#REDIRECT") or "#転送" in t.group(1)[:20]:
                continue
            for s in clean(t.group(1)):
                w(s + "\n")
            n += 1
        print(f"{path}: {n} pages", file=sys.stderr)


if __name__ == "__main__":
    main()
