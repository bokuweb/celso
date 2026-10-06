"""e-Gov 法令一括ダウンロード (XML) から 1 行 1 文のテキストを抽出する。

使い方: python3 scripts/extract_egov.py data/egov > data/corpus/egov.txt
同一法令の複数版がある場合は、ディレクトリ名の日付が最も新しい版だけを使う。
"""
import os
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from textnorm import kanji_numbers_to_arabic, norm  # noqa: E402

SENTENCE_PARENTS = {"EnactStatement", "ArticleCaption", "ParagraphCaption", "Remarks"}


def latest_versions(root: str) -> list[str]:
    best: dict[str, tuple[str, str]] = {}
    for d in os.listdir(root):
        parts = d.split("_")
        if len(parts) < 2:
            continue
        law_id, date = parts[0], parts[1]
        if law_id not in best or date > best[law_id][0]:
            best[law_id] = (date, d)
    out = []
    for _, d in best.values():
        p = os.path.join(root, d, d + ".xml")
        if os.path.exists(p):
            out.append(p)
    return sorted(out)


def extract(path: str) -> list[str]:
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return []
    lines = []
    for el in tree.iter():
        tag = el.tag
        # 目次・表・別表は文になっていないので除外
        if tag in ("TOC", "TableStruct", "AppdxTable", "AppdxStyle", "AppdxFormat"):
            el.clear()
            continue
        if tag.endswith("Sentence") and tag != "Sentence" or tag in SENTENCE_PARENTS:
            text = "　".join("".join(s.itertext()).strip() for s in el.iter("Sentence")) or "".join(el.itertext())
            text = text.strip()
            if text:
                lines.append(norm(kanji_numbers_to_arabic(text)))
    return lines


def main() -> None:
    files = latest_versions(sys.argv[1])
    print(f"{len(files)} laws", file=sys.stderr)
    out = sys.stdout
    with ProcessPoolExecutor() as ex:
        for lines in ex.map(extract, files, chunksize=16):
            for line in lines:
                out.write(line.replace("\n", " ") + "\n")


if __name__ == "__main__":
    main()
