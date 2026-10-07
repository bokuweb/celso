"""学習に入れていない自治体の例規から、学習コーパス (例規・法令) と完全に一致する行を除く。

自治体の例規には各地でほぼ同じ条文があり、そのまま評価や判定器の学習に使うと、言語モデルが
見たことのある文で甘い結果になる (取得した行の約 4 割が一致した)。
"""
seen = set()
for p in ['data/corpus/reiki.f.txt', 'data/corpus/egov.txt']:
    for line in open(p, encoding='utf-8'):
        seen.add(hash(line.strip()))
for c in ['一宮市', '高槻市', '加古川市', '生駒市']:
    lines = [x.strip() for x in open(f'data/corpus/heldout_{c}.raw.txt', encoding='utf-8') if x.strip()]
    keep = [x for x in lines if hash(x) not in seen]
    open(f'data/corpus/heldout_{c}.txt', 'w', encoding='utf-8').write('\n'.join(keep) + '\n')
    print(c, len(lines), '->', len(keep))
