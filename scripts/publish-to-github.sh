#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

remote_url="https://github.com/AI-am-walking-here/DRL_Project.git"

if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "$remote_url"
elif [[ "$(git remote get-url origin)" != "$remote_url" ]]; then
  git remote set-url origin "$remote_url"
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "GitHub CLI is not authenticated."
  echo "Run: gh auth login --web"
  exit 1
fi

git fetch origin

if git merge-base --is-ancestor origin/main main 2>/dev/null; then
  echo "Pushing main to origin..."
  git push -u origin main
elif git merge-base --is-ancestor main origin/main 2>/dev/null; then
  echo "Local main is behind origin/main. Pull first, then push."
  exit 1
else
  cat <<'EOF'
Local main and origin/main have unrelated histories.

  - GitHub (DRL_Project): flat layout (code at repo root)
  - Local workspace: monorepo layout (code under robot-routes/)

To publish this monorepo to GitHub (replaces remote layout):
  git push -u origin main --force-with-lease

To adopt the GitHub flat layout instead, clone fresh:
  git clone https://github.com/AI-am-walking-here/DRL_Project.git
EOF
  exit 1
fi

echo
echo "Published: https://github.com/AI-am-walking-here/DRL_Project"
