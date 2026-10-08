"""1 行 1 文のテキストを、文字単位の言語モデル用に「文字 空白 文字 …」へ変える (空白類は落とす)。

使い方: python3 scripts/charlm/to_chars.py < コーパス > 出力.ch
"""
import sys

for line in sys.stdin:
    cs = [c for c in line.rstrip('\n') if not c.isspace()]
    if len(cs) >= 5:
        sys.stdout.write(' '.join(cs) + '\n')
