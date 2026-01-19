#!/usr/bin/env bash
set -euo pipefail

echo "=== checking for homebrew ==="
if ! command -v brew >/dev/null; then
  echo "installing homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

if [ -x /opt/homebrew/bin/brew ]; then
  eval "$(/opt/homebrew/bin/brew shellenv)"
elif [ -x /usr/local/bin/brew ]; then
  eval "$(/usr/local/bin/brew shellenv)"
fi

echo "=== checking for python ==="
if ! command -v python3 >/dev/null; then
  brew install python
fi

echo "=== checking for openscad ==="
if ! command -v openscad >/dev/null; then
  brew install --cask openscad
fi

BREW_PREFIX=$(brew --prefix)
if ! echo "$PATH" | grep -q "$BREW_PREFIX/bin"; then
  echo "export PATH=\"$BREW_PREFIX/bin:\$PATH\"" >> ~/.zprofile
fi

echo "=== installing requirements ==="
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo "done! run 'python app.py' to get started with a11yshape."