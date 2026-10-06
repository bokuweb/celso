"""コーパス抽出とチェッカーで共有する文字正規化。

文字数を変えない NFKC (1 文字 → 1 文字になる場合だけ置換) にする。チェッカー側も同じ規則で
正規化するので、検出位置を元テキストのオフセットへそのまま戻せる。
"""
import re
import unicodedata

_cache: dict[str, str] = {}


def norm_char(c: str) -> str:
    r = _cache.get(c)
    if r is None:
        n = unicodedata.normalize("NFKC", c)
        r = n if len(n) == 1 else c
        _cache[c] = r
    return r


def norm(text: str) -> str:
    return "".join(norm_char(c) for c in text)


_KANJI_DIGIT = {"〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_KANJI_UNIT = {"十": 10, "百": 100, "千": 1000}
_KNUM = "〇零一二三四五六七八九十百千"


def kanji_to_int(s: str) -> int | None:
    total, cur = 0, 0
    has_digit_only = all(ch in _KANJI_DIGIT for ch in s)
    if has_digit_only:
        # 「二〇二四」のような位取りなし表記
        return int("".join(str(_KANJI_DIGIT[ch]) for ch in s))
    for ch in s:
        if ch in _KANJI_DIGIT:
            cur = _KANJI_DIGIT[ch]
        elif ch in _KANJI_UNIT:
            total += (cur or 1) * _KANJI_UNIT[ch]
            cur = 0
        else:
            return None
    return total + cur


# 法令 XML は漢数字 (第二百二十六号, 昭和二十五年, 三十日以内) だが、条例や一般文は算用数字が多い。
# 番号・日付・数量の文脈 (第〜 / 元号〜 / 助数詞の前 / 分の〜) に限って算用数字へ寄せ、
# 「一部」「一の」「十分」等の語は壊さない。
_COUNTERS = "条|項|号|編|章|節|款|目|年|月|日|円|人|回|件|倍|箇月|か月|ヶ月|時間|週間|歳|割|パーセント|分の"
_NUM_CTX = re.compile(
    rf"(?:(?<=第)|(?<=明治)|(?<=大正)|(?<=昭和)|(?<=平成)|(?<=令和))([{_KNUM}]+)"
    rf"|([{_KNUM}]+)(?=(?:{_COUNTERS}))"
    rf"|(?<=条の)([{_KNUM}]+)|(?<=分の)([{_KNUM}]+)"
)


def kanji_numbers_to_arabic(text: str) -> str:
    def repl(m: re.Match) -> str:
        s = next(g for g in m.groups() if g)
        v = kanji_to_int(s)
        return str(v) if v is not None else s

    return _NUM_CTX.sub(repl, text)
