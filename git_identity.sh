#!/usr/bin/env bash
set -euo pipefail

# Configure global Git identity. You can override via env vars:
#   EMAIL=my@email NAME=myname ./git_identity.sh

EMAIL="${EMAIL:-averywright.21202@gmail.com}"
NAME="${NAME:-avewright}"

git config --global user.email "$EMAIL"
git config --global user.name "$NAME"

echo "Configured git user.name=$(git config --global --get user.name)"
echo "Configured git user.email=$(git config --global --get user.email)"


