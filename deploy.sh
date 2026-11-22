#!/usr/bin/env bash
# Build the site from master and publish the output to the gh-pages branch.
#
# GitHub Pages serves gh-pages as plain static files (legacy build, no Jekyll),
# so gh-pages must contain only the built _site/. Never edit gh-pages by hand:
# change the source on master, then run ./deploy.sh.
#
#   ./deploy.sh            build, show what changed, commit and push
#   ./deploy.sh --dry-run  build and show what would change, no push
set -euo pipefail

cd "$(dirname "$0")"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

branch=$(git rev-parse --abbrev-ref HEAD)
[[ "$branch" == "master" ]] || { echo "Run from master (on $branch)." >&2; exit 1; }
git diff --quiet && git diff --cached --quiet || { echo "Commit your changes first." >&2; exit 1; }

src_sha=$(git rev-parse --short HEAD)
work=$(mktemp -d)
trap 'git worktree remove --force "$work/pages" 2>/dev/null || true; rm -rf "$work"' EXIT

git fetch -q origin gh-pages
git worktree add -q "$work/pages" origin/gh-pages

JEKYLL_ENV=production bundle exec jekyll build -q -d "$work/site"

rsync -a --delete --exclude .git "$work/site/" "$work/pages/"

cd "$work/pages"
git add -A
if git diff --cached --quiet; then
  echo "gh-pages already up to date."
  exit 0
fi
git diff --cached --stat
$DRY_RUN && { echo "(dry run, nothing pushed)"; exit 0; }

git commit -q -m "Publish built site from master@${src_sha}"
git push -q origin HEAD:gh-pages
echo "Published master@${src_sha} to gh-pages."
