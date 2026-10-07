"""rawtext/*.txt (dump_raw.py の出力) から契約書コーパスを作る。

  uv run --with pypdf --with cryptography python dump_raw.py      # files/ → rawtext/
  python3 build_contracts.py [--out PATH] [--stats]                # rawtext/ → contracts.txt

処理:
  - PDF はページ (\\f 区切り) ごとに、複数ページに繰り返し出る行 (ヘッダ・フッタ) とページ番号・目次行を落とす
  - PDF の行折り返しをつなぎ直す (行が版面幅近くまで埋まっていて文末でなく、次行が条項ラベル等で始まらない場合)
  - 日本語文字どうしのあいだの空白を詰める
  - 文頭の条項ラベル (第N条 / 2 / (1) / 一 / ア など) を外し、括弧だけの見出し行は捨てる (fetch_reiki.py と同じ流儀)
  - 括弧の外の「。」で文に分け、1 行 1 文で出す
  - 文字化け・表の残骸・様式の空欄 (日本語が少ない、ひらがながない等) の行を落とす
  - celso/scripts/textnorm.py の kanji_numbers_to_arabic() と norm() で正規化する

評価データ漏洩防止のため、JEITA の文書と IPA アジャイル開発版モデル契約は入力に含めない (EXCLUDE で二重に拒否)。
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import statistics
import sys

# 取得物・中間ファイルはリポジトリの外 (既定 data/contracts_raw/、gitignore 済み) に置く
ROOT = os.environ.get("CONTRACTS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "contracts_raw")
os.makedirs(ROOT, exist_ok=True)
CELSO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(CELSO, "scripts"))
from textnorm import kanji_numbers_to_arabic, norm  # noqa: E402

RAW = os.path.join(ROOT, "rawtext")
DEFAULT_OUT = os.path.join(CELSO, "data", "corpus", "contracts.txt")

# 評価データ (JEITA ソフトウェア開発モデル契約関連, IPA アジャイル開発版) は絶対に入れない
BANNED = re.compile(r"jeita|agile|アジャイル", re.I)
# フォントの ToUnicode が壊れていて本文を復元できない PDF
MOJIBAKE_DOCS = {
    "bunka_bijutsu_kougei_guidebook.pdf",
    "bunka_freelance_keiyaku_guidebook.pdf",
    "bunka_haiyu_haikyu_keiyaku_lesson.pdf",
}
# IPA 情報システム・モデル取引・契約書 (第二版) は JEITA のモデル契約と系譜が同じで文言の重なりが大きい
# 可能性があるため、既定では contracts.txt に入れず別ファイルに出す (--with-ipa で同梱)
IPA_DOCS = {"ipa_model_torihiki_keiyakusho_v2.docx"}
# 目視で同一文書の重複取得と確認したもの (研修会テキストの巻末に同じ基本契約書が収録されている)
EXPLICIT_DUP = {"bunka_kihon_keiyakusho.docx"}

JA = r"々〆〇぀-ヿ㐀-鿿豈-﫿｡-ﾟ、。，．・「」『』【】〔〕（）［］｛｝〈〉《》ー―‐～…：；！？"
JA_RE = re.compile(f"[{JA}]")
# 日本語文字 (と括弧類) にはさまれた空白は PDF 抽出のゴミ
SPACE_BETWEEN = re.compile(rf"(?<=[{JA}0-9()\[\]%])[ \t　]+(?=[{JA}0-9()\[\]])|(?<=[{JA}])[ \t　]+(?=[A-Za-z])|(?<=[A-Za-z0-9])[ \t　]+(?=[{JA}])")
# ページ番号だけの行: "12" "- 12 -" "－12－" "12/96" "P.12" "(12)"
PAGE_NO = re.compile(r"^[\s\-－–—―ー‐・]*(?:p\.?\s*)?\(?\d{1,4}\)?(?:\s*/\s*\d{1,4})?[\s\-－–—―ー‐・]*$", re.I)
# 目次: リーダー罫で終わるかページ番号へ続く行
TOC = re.compile(r"(?:[・\.…‥･·]\s*){4,}|…{2,}\s*\d+\s*$|\.{3,}\s*\d+\s*$")
# 文字化けで出る文字 (コプト・キリル・ギリシャ拡張、私用領域、制御文字、置換文字など)
BAD_CHARS = re.compile(r"[\u0080-¦¨-¯¸-¿Ā-ͯͰ-  - ‰-‱‴-›‼-ℂ℄-ℕ℗-℠℣-↏Ⰰ-⹿-￰-￿\U00010000-\U0010ffff]")
# 次行がこれで始まるなら前行とつながない (条項ラベル・箇条書き・見出し)
_KANA_LABEL = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
_IROHA = "イロハニホヘトチリヌルヲワカヨタレソツネナラムウヰノオクヤマケフコエテアサキユメミシヱヒモセス"
BLOCK_START = re.compile(
    r"^(?:第[0-9一二三四五六七八九十百]+[条章節款項号編]|\(\d+\)|\d+[\.\s　]|[①-⑳⑴-⒇❶-❿➀-➉]|[一二三四五六七八九十]+[\s　]|"
    rf"[{_KANA_LABEL}][\s　]|\([{_KANA_LABEL}{_IROHA}a-zA-Z]\)|[・●○◆◇■□▪▶►✓✔※★☆→【《〈<＜Ｑ]|Q\d*[\.\s:：]|A\d*[\.\s:：]|\([^()]{{1,20}}\)$)"
)
SENT_END = re.compile(r"[。！？!?」』)]$")

LABEL = re.compile(
    r"^(?:(?:第\d+条(?:の\d+)*|第\d+項)(?:\([A-Z]\))?|\(\d+\)|\d+(?:\.\d+)*\.(?=[^\d\s.])|\d+(?:\.\d+)*\.?|[一二三四五六七八九十]+|[①-⑳⑴-⒇❶-❿]|"
    rf"[{_KANA_LABEL}{_IROHA}]|\([{_KANA_LABEL}{_IROHA}]\)|[a-zA-Z]\.?|\([a-zA-Z]\)|[ⅰ-ⅿi-x]+\.?|\([ⅰ-ⅿi-x]+\)|[・●○◆◇■□▪※])(?:\s+|(?<=\.)(?=[^\d\s]))"
)
LEADING_BULLET = re.compile(r"^[*・●○◆◇■□▪▶►✓✔★☆→·•‧∙◎▼▽▲△]\s*")


_CIRCLED = {c: i + 1 for i, c in enumerate("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳")}
_CIRCLED.update({c: i + 1 for i, c in enumerate("❶❷❸❹❺❻❼❽❾❿")})
_CIRCLED.update({c: i + 1 for i, c in enumerate("➀➁➂➃➄➅➆➇➈➉")})
_CIRCLED.update({c: i + 1 for i, c in enumerate("⓵⓶⓷⓸⓹⓺⓻⓼⓽⓾")})
_CIRCLED.update({c: i + 11 for i, c in enumerate("⓫⓬⓭⓮⓯⓰⓱⓲⓳⓴")})
_ZW = re.compile(r"[\u200b-\u200f\u2060\ufeff\u00ad]")


def prenorm(line: str) -> str:
    """norm() の前処理。行頭の丸数字はラベルなので「(n) 」にしておく (NFKC だと「1」になり本文に貼り付く)。"""
    line = _ZW.sub("", line)
    m = re.match(r"^\s*([" + "".join(_CIRCLED) + r"])\s*", line)
    if m:
        line = f"({_CIRCLED[m.group(1)]}) " + line[m.end():]
    return norm(line)


def page_lines(text: str) -> list[list[str]]:
    return [[prenorm(l).rstrip() for l in p.split("\n")] for p in text.split("\f")]


def drop_headers_footers(pages: list[list[str]]) -> list[list[str]]:
    """複数ページに同じ形で出る短い行 (数字は伏せて比較) をヘッダ・フッタとみなして落とす。"""
    if len(pages) < 4:
        return pages
    key = lambda l: re.sub(r"\d+", "#", re.sub(r"\s+", "", l))  # noqa: E731
    cnt = collections.Counter()
    for p in pages:
        # ヘッダ・フッタはページの先頭・末尾付近に出る
        cand = [l for l in p if l.strip()]
        for k in {key(l) for l in cand[:3] + cand[-3:]}:
            cnt[k] += 1
    thr = max(3, int(len(pages) * 0.25))
    rep = {k for k, c in cnt.items() if c >= thr and 0 < len(k) <= 60}
    return [[l for l in p if key(l) not in rep] for p in pages]


def join_pdf_lines(pages: list[list[str]]) -> list[str]:
    lines = [l.strip() for p in pages for l in p + [""]]  # ページ境界は空行扱いにはしない (後で判定)
    lens = [len(l) for l in lines if len(l) >= 8]
    if not lens:
        return []
    # 版面幅: 長めの行の代表値。これに近い長さの行は折り返しとみなす
    width = sorted(lens)[int(len(lens) * 0.8)]
    paras: list[str] = []
    cur = ""
    for l in lines:
        if not l:
            continue
        if cur:
            last = cur.rsplit("\n", 1)[-1]
            prev_full = len(last) >= width * 0.7
            # 表のセルなど幅の狭い枠の折り返し: 行末が助詞・読点など句の途中で終わっている
            mid_phrase = re.search(r"[\u3041-\u3093、，・(「『]$", last) and not re.search(r"(?:について|とは|ください|ます|です|ません|でした|ました)$", last)
            # 次行が語の途中から始まる (長音・小書き仮名・閉じ括弧・読点で始まる)
            cont_next = re.match(r"^[ーぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ、。)」』]", l)
            if (prev_full or mid_phrase or cont_next) and not SENT_END.search(cur) and not BLOCK_START.match(l):
                cur += "\n" + l
                continue
            paras.append(cur)
        cur = l
    if cur:
        paras.append(cur)
    # 段落内の改行は「つなぐ」: 日本語どうしなら詰め、英字どうしなら空白
    out = []
    for p in paras:
        p = re.sub(rf"(?<=[{JA}0-9])\n(?=[{JA}0-9])", "", p)
        p = p.replace("\n", " ")
        out.append(p)
    return out


def split_sentences(p: str) -> list[str]:
    """括弧 (丸括弧・かぎ括弧) の外にある「。」で分ける。"""
    out, buf, depth = [], [], 0
    for ch in p:
        buf.append(ch)
        if ch in "(「『【〔[":
            depth += 1
        elif ch in ")」』】〕]":
            depth = max(0, depth - 1)
        elif ch == "。" and depth == 0:
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))
    return out


def clean_sentence(s: str) -> str | None:
    s = norm(kanji_numbers_to_arabic(re.sub(r"\s+", " ", s).strip()))
    s = LEADING_BULLET.sub("", s).strip()
    s = LABEL.sub("", s, count=1).strip()
    s = LEADING_BULLET.sub("", s).strip()
    s = SPACE_BETWEEN.sub("", s)
    s = SPACE_BETWEEN.sub("", s)  # 1 文字おきの空白 (「建 設 工 事」) は 1 回では詰め切れない
    if BAD_CHARS.search(s):
        return None
    if re.search(r"https?://|www\.|@[a-z0-9.-]+\.[a-z]{2,}", s, re.I):
        return None
    if TOC.search(s) or len(re.findall(r"\d+\s*\.\s*\d+\s*\.", s)) >= 2:
        return None
    t = re.sub(r"\s", "", s)
    if len(t) < 8:
        return None
    ja = len(JA_RE.findall(t))
    hira = len(re.findall(r"[ぁ-ゟ]", t))
    if ja / len(t) < 0.6 or hira < 3 or hira / len(t) < 0.1:
        return None
    # 様式の差込欄 (「/の受注を承諾します。」「<受注者名>」だけの行など)
    if re.match(r"^[/／<＞>]", t) or re.fullmatch(r"※?<[^<>]*>", t):
        return None
    # 括弧だけの見出し・注記
    if re.fullmatch(r"\([^()]*\)|【[^【】]*】|〔[^〔〕]*〕|「[^「」]*」", t):
        return None
    # 様式の記入欄 (「年月日」「〇〇株式会社」「印」だけの行など)
    if re.fullmatch(r"[年月日印殿様〇○◯□\s:・()]+", t) or t.count("○") + t.count("〇") > len(t) * 0.3:
        return None
    return s


def extract_doc(name: str, text: str) -> list[str]:
    if name.endswith(".pdf"):
        pages = page_lines(text)
        pages = drop_headers_footers(pages)
        pages = [[l for l in p if not PAGE_NO.match(l) and not TOC.search(l) and not BAD_CHARS.search(l)] for p in pages]
        paras = join_pdf_lines(pages)
    else:
        # Word / RTF は 1 行 1 段落で出るので折り返しのつなぎ直しは不要
        paras = [prenorm(l).strip() for l in text.split("\n")]
        paras = [p for p in paras if p and not PAGE_NO.match(p) and not TOC.search(p)]
    out = []
    for p in paras:
        # 表のセル区切り (タブ) は文の区切りとして扱う
        for cell in re.split(r"\t+", p):
            for s in split_sentences(cell):
                c = clean_sentence(s)
                if c:
                    out.append(c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--ipa-out", default=os.path.join(ROOT, "contracts_ipa_model.txt"))
    ap.add_argument("--with-ipa", action="store_true")
    ap.add_argument("--stats", default=os.path.join(ROOT, "STATS.tsv"))
    args = ap.parse_args()

    docs = {}
    for fn in sorted(os.listdir(RAW)):
        name = fn[:-4]
        if BANNED.search(name):
            raise SystemExit(f"評価データらしき文書が入力にある: {name}")
        if name in MOJIBAKE_DOCS:
            continue
        docs[name] = extract_doc(name, open(os.path.join(RAW, fn), encoding="utf-8").read())

    # 同じ文書の重複取得 (版違い・型違いの取り込み) の検出: 文集合の重なりが大きい組を報告し、後ろを捨てる
    sets = {n: set(v) for n, v in docs.items()}
    dropped = set()
    names = list(docs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a in dropped or b in dropped or not sets[a] or not sets[b]:
                continue
            inter = len(sets[a] & sets[b])
            r = inter / min(len(sets[a]), len(sets[b]))
            if r >= 0.3:
                print(f"overlap {r:.2f} {a} / {b}", file=sys.stderr)
            if r >= 0.95 or {a, b} & EXPLICIT_DUP:
                small = a if len(sets[a]) < len(sets[b]) else b
                dropped.add(small)
                print(f"  → 重複取得とみなして {small} を除外", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as f, open(args.ipa_out, "w", encoding="utf-8") as fi, open(args.stats, "w", encoding="utf-8") as st:
        st.write("doc\tlines\tchars\tincluded\n")
        for n, lines in docs.items():
            inc = n not in dropped and (n not in IPA_DOCS or args.with_ipa)
            dst = f if inc else (fi if n in IPA_DOCS else None)
            if dst:
                for l in lines:
                    dst.write(l + "\n")
            st.write(f"{n}\t{len(lines)}\t{sum(len(l) for l in lines)}\t{'yes' if inc else 'no'}\n")


if __name__ == "__main__":
    main()
