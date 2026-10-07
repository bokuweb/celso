#!/usr/bin/env bash
# playground/web を gh-pages ブランチへ置いて push する (GitHub Pages の配信元)。
# 先に scripts/build_playground.sh で組み立てておくこと。gh-pages は生成物だけを持つ孤立ブランチで、
# 毎回 1 コミットに作り直す (モデルの更新で履歴が膨らまないように)。
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=playground/web
[ -s "$SRC/pkg/celso_playground_bg.wasm" ] && [ -s "$SRC/assets/model.bin.gz" ] || {
  echo "先に scripts/build_playground.sh を実行してください" >&2
  exit 1
}
REV=$(git rev-parse --short HEAD)
ORIGIN=$(git remote get-url origin)
TMP=$(mktemp -d)
cp -R "$SRC"/. "$TMP"/
touch "$TMP/.nojekyll"
(
  cd "$TMP"
  git init -q -b gh-pages
  git add -A
  git commit -q -m "playground: celso ${REV} から生成"
  git push -q -f "$ORIGIN" gh-pages
)
rm -rf "$TMP"
