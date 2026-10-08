"""文字単位の直しの判定器 (charrank.tsv) を学習する。

入力は celso check の調査用の出力 (CELSO_TRACE) から [char] 行を抜き出したもの (dump.sh)。
- 正例: JWTD の誤りのある文 (tr_pre*) で、直した文が正しい文 (tr_post*) と一致した候補
- 負例: JWTD の正しい文 (tr_post*) と、判例要旨の正しい文 (han_tr) で出た候補 (判例要旨は重みを足す)

使い方: python3 scripts/charrank/train.py 文のディレクトリ 候補のディレクトリ prior.tsv 判例の重み 閾値 出力
(scikit-learn が必要)
"""
import os
import sys

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from feat import feats, load_prior, nf, parse  # noqa: E402


def main(text_dir, cand_dir, prior_path, han_weight, tau, out_path):
    load_prior(prior_path)
    X, y, w = [], [], []
    for k in range(64):
        if not os.path.exists(f'{cand_dir}/tr_post{k}.c'):
            continue
        posts = {nf(l.rstrip('\n')) for l in open(f'{text_dir}/tr_post{k}.txt', encoding='utf-8')}
        for c in parse(f'{cand_dir}/tr_pre{k}.c'):
            if nf(c['fx']) in posts:
                X.append(feats(c)); y.append(1); w.append(1.0)
        for c in parse(f'{cand_dir}/tr_post{k}.c'):
            X.append(feats(c)); y.append(0); w.append(1.0)
    for c in parse(f'{cand_dir}/han_tr.c'):
        X.append(feats(c)); y.append(0); w.append(float(han_weight))
    print(f'examples {len(y)} positives {sum(y)}', file=sys.stderr)
    dv = DictVectorizer()
    clf = LogisticRegression(C=1.0, max_iter=3000)
    clf.fit(dv.fit_transform(X), y, sample_weight=w)
    # 切片は bias の重みに足す (celso は特徴量の重みの和だけを見る)
    W = dict(zip(dv.feature_names_, clf.coef_[0]))
    W['bias'] = W.get('bias', 0.0) + clf.intercept_[0]
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('# 文字単位の直しの採否の判定器 (celso scripts/charrank)。#prior 行は京大 JWTD v2 (CC BY-SA 3.0) の\n')
        f.write('# 学習用の差分から数えた「誤り → 正しい」の出現数で、JWTD の二次的著作物として CC BY-SA 3.0 で提供する。\n')
        f.write(f'#tau\t{tau},{tau},{tau}\n')
        for k in sorted(W):
            if abs(W[k]) > 1e-6:
                f.write(f'{k}\t{W[k]:.6g}\n')
        for l in open(prior_path, encoding='utf-8'):
            f.write('#prior\t' + l)


if __name__ == '__main__':
    main(*sys.argv[1:])
