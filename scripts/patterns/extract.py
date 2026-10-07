"""JWTD (日本語 Wikipedia 入力誤りデータセット) の train から、誤字の書き換えパターンの候補を作る。

enno.jp のような「誤りのパターンを蓄積して照合する」方式を、実際の誤字から自前で作るための第 1 段。
各誤りの位置を前後の文字ごと切り出し (文脈の幅を変えた複数の候補)、(誤り → 正しい) の組を数える。
誤検出しにくいものへの絞り込みは、学習コーパスでの出現数を数えてから行う (select.py)。

出力 (TSV): 誤り \t 正しい \t 支持数 (train での件数) \t カテゴリ
JWTD のライセンスは CC BY-SA 3.0。ここから作ったパターン表も同じ条件で扱う。
"""
import collections
import json
import sys
import unicodedata

CONTEXTS = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2)]


def norm(s: str) -> str:
    # celso の norm (NFKC + 一部の記号の統一) に合わせて、全角英数を半角にする程度に揃える
    return unicodedata.normalize("NFKC", s)


def diff_span(a: str, b: str):
    """a と b の共通の前置・後置を除いた差分の範囲 (a 側の [i, j)、b 側の [i, k))。"""
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    j, k = len(a), len(b)
    while j > i and k > i and a[j - 1] == b[k - 1]:
        j -= 1
        k -= 1
    return i, j, k


def main(path: str, out: str) -> None:
    cnt: collections.Counter = collections.Counter()
    cat_of: dict = {}
    n = 0
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if len(d.get("diffs", [])) != 1:
            continue
        cat = d["diffs"][0].get("category", "")
        if cat in ("not-typo", "others"):
            continue
        pre, post = norm(d["pre_text"]), norm(d["post_text"])
        i, j, k = diff_span(pre, post)
        if j - i > 4 or k - i > 4 or (j == i and k == i):
            continue
        n += 1
        for l, r in CONTEXTS:
            if i - l < 0 or j + r > len(pre) or k + r > len(post):
                continue
            wrong = pre[i - l : j + r]
            right = post[i - l : k + r]
            if len(wrong) < 2 or wrong == right:
                continue
            key = (wrong, right)
            cnt[key] += 1
            cat_of.setdefault(key, cat)
    with open(out, "w", encoding="utf-8") as f:
        for (w, r), c in cnt.items():
            if c >= 2:
                f.write(f"{w}\t{r}\t{c}\t{cat_of[(w, r)]}\n")
    print(f"{n} edits, {sum(1 for c in cnt.values() if c >= 2)} candidate patterns (support >= 2)", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
