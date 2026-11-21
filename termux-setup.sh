#!/data/data/com.termux/files/usr/bin/bash
# MHexeract gateway on Android (Termux).
# Lite mode uses only the Python standard library and routes every model through OpenRouter,
# so nothing native has to compile on the phone. Full LiteLLM mode is optional (see --full).
set -euo pipefail

pkg update -y
pkg install -y python git termux-api || pkg install -y python git

if [ "${1:-}" = "--full" ]; then
  # LiteLLM needs Rust-built wheels (tokenizers/tiktoken/pydantic-core); this can take a while on a phone.
  pkg install -y rust binutils
  pip install --upgrade pip
  pip install litellm
  echo "Full LiteLLM mode installed. Direct provider keys (OPENAI_API_KEY, ...) will now work."
else
  echo "export GATEWAY_LITE=1" >> ~/.bashrc
  echo "Lite mode enabled (stdlib only, OpenRouter for all models)."
fi

if ! grep -q OPENROUTER_API_KEY ~/.bashrc 2>/dev/null; then
  read -r -p "OpenRouter API key (leave blank to set later): " KEY
  [ -n "$KEY" ] && echo "export OPENROUTER_API_KEY=$KEY" >> ~/.bashrc
fi

cat <<'EOF'

Done. Next:
  source ~/.bashrc
  python gateway.py chat "Hello from my phone"
  python gateway.py gui            # then open http://localhost:8765 in Chrome
                                   # Chrome menu > Add to Home screen installs it as an app
Keep it running in the background:  termux-wake-lock && python gateway.py gui
News across the spectrum:           python gateway.py news
Windows apps on Android:            install Winlator (https://winlator.org); Termux cannot run Wine directly
EOF
