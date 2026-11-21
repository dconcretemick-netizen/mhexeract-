# MHexeract

*Project codename: **6D** — the hexeract is the six-dimensional cube, three steps past cubed.*

**[MCUBED install](https://github.com/dconcretemick-netizen/MCUBED/releases/tag/latest)**: [Android APK](https://github.com/dconcretemick-netizen/MCUBED/releases/download/latest/MCUBED.apk) · [Desktop zip](https://github.com/dconcretemick-netizen/MCUBED/releases/download/latest/MCUBED-desktop.zip) · [How to install](INSTALL.md) · [Security](SECURITY.md). Always the newest build.

A multi-model AI gateway with a phone-friendly web dashboard. One entry point for OpenAI, Anthropic, Google and DeepSeek models, with OpenRouter as a fallback route for every model.

## What MHexeract is for

MHexeract is built to give you real AI at no cost. Instead of one paid subscription, it spreads your questions across many free AI providers — plus models on your own devices — and automatically fails over when one hits its limit. These are the jobs that design is best at:

- **Everyday AI, no subscription.** Chat, questions, summaries, translations, brainstorming. Requests route across free plans from providers like Google Gemini, Groq, Mistral, NVIDIA and OpenRouter's free models, keyless Pollinations, and Meta's Muse Glimmer — so no single provider's paywall or daily cap is a hard wall.
- **A jury, not an oracle.** Pick **Council** in the chat's model menu and the models confer: several answer your question — with the full chat history, so the discussion stays in context — then the strongest one reads all the answers and writes the final one, keeping what they agree on and correcting the rest. Each model's own answer stays visible under the reply. For buying decisions, "is this email a scam," health-ish questions and anything you'd rather not take one AI's word for, a conferred answer beats one confident guess.
- **Car help that pays for itself.** Plug in an ELM327 adapter and the diagnostics tab reads and clears fault codes, with over 12,000 plain-English code descriptions stored on the device — then ask the AI what a code means, whether the car is safe to drive, and what a shop typically charges.
- **News with the blindspots showing.** The News tab groups coverage of the same story from outlets across the spectrum and flags blindspots — stories only one side is covering — so you see the event and the framing at the same time.
- **The household AI.** One phone or home PC serving everyone's homework help, letters, resumes, forms and polite-but-firm emails — instead of a subscription per person.
- **Private by architecture.** Add a local model (for example Ollama on a home PC) and MHexeract finds it on your network. Sensitive questions can stay inside your house entirely — no account, no cloud, nothing to revoke.
- **Keys, kept easy.** Free keys are the fuel, so MHexeract keeps them painless: every provider in the AI keys card has a **Get key** button that goes straight to its free signup (no card required for the free plans), keys are saved once on your device only, and a background gardener checks them, quietly removes ones a provider has rejected, and parks providers that stop answering until they recover. Truly anonymous models need no key or account at all — Pollinations is built in, and new keyless providers get added as they appear.

The honest limits: free tiers have quotas and occasionally disappear, and the best free models trail the top paid ones on the hardest reasoning. MHexeract's answer is breadth and failover — many providers, a local model as the floor, and background care that keeps the pool healthy on its own.

## Features

- Automatic fallback routing with retries, circuit breakers and quarantine
- Usage ledger (SQLite) for requests, tokens, latency and quality, plus a response cache
- Self-learning router that balances quality, reliability, speed and effort per task type
- Automated process improvement: the optimizer tunes tiers, route order, token estimates and quarantines every 50 requests and every 6 hours while the dashboard or app runs, with a change log and automatic rollback
- Conversation compression for long chats and automatic storage compaction
- Lite mode for Android/Termux: standard library only, no native builds
- OpenAI-style chat as the main screen: multi-turn conversations, model picker (MHexeract auto, fast, best quality, or any model with a key), ratings that teach the router
- Free by default: only free AI (local models and no-card free plans) unless you switch paid AI on
- AI nodes: finds 21 cloud providers (Gemini, Groq, Mistral, Cerebras, Hugging Face, NVIDIA and more) plus local model servers such as Ollama and LM Studio, and adds every model it can use
- OpenAI-compatible API: MHexeract acts as a model named `mcubed` at `/v1/chat/completions` and `/v1/models`, including streaming
- Web browser tab: search or open any address in a reader view, follow links, open the full page, or send the page to the chat
- News tab built for fairness: 15 outlets rated from left to right, stories grouped so you can compare how each outlet covers the same event, blindspot flags, and an AI balanced brief
- Windows apps through Wine (Box64 on ARM, Winlator on Android)
- Installable web dashboard (chat, news, web, games, stats, update)
- Gateway Doom: a touch-friendly game where the demons are your real error types
- Solitaire (Klondike): tap or drag, undo, hints, draw 1 or 3, saved progress
- Tetris: swipe and tap controls, hold, next queue, ghost piece, wall kicks, saved game and best score
- Pip's Leap: an original side-scrolling platformer in three worlds (Meadow, Caves, Sky) with large multitouch buttons, forgiving jumps, checkpoints and saved progress
- Update tab: installs the newest MHexeract from GitHub with integrity checks, a backup and one-tap rollback; the Android app downloads and installs the newest APK
- Native Android APK built automatically by GitHub Actions

## Quick start (no API keys)

```bash
python gateway.py simulate --n 300        # offline learning demo
GATEWAY_MOCK=1 python gateway.py gui      # dashboard at http://localhost:8765
```

## Real models

```bash
cp .env.example .env                      # add keys, then load them:
set -a; . ./.env; set +a
pip install -r requirements.txt           # optional; skip on Android
python gateway.py models                  # shows which routes have keys
python gateway.py chat "Explain CAP theorem"
python gateway.py gui
```

An OpenRouter key alone is enough to reach every model in `models.json`. Each provider key you add opens its direct routes:

| Key | Opens |
|---|---|
| `OPENROUTER_API_KEY` | Every model, as the fallback route |
| `OPENAI_API_KEY` | GPT-6 Astra, GPT-6.1 Sol/Terra, GPT-6 Luna, GPT-5.6, GPT-5.5, GPT-5.4 (direct) |
| `ANTHROPIC_API_KEY` | Claude models (direct, desktop mode) |
| `DEEPSEEK_API_KEY` | DeepSeek models (direct) |
| `VERTEXAI_PROJECT` | Gemini models (direct, desktop mode) |

Find every OpenAI chat model your key can use and add the new ones:

```bash
python gateway.py discover-openai          # preview
python gateway.py discover-openai --write  # add to models.json
```

## Repositories

With `GITHUB_TOKEN` set (or the GitHub CLI signed in via `gh auth login`), MHexeract finds every repository you own, collaborate on, or reach through an organization, and records it in `repos.json`. New repositories are added automatically every 6 hours while the dashboard or app runs (`repo_sync_every_hours` in `models.json`), and the Storage tab has a "Sync now" button. Run `python gateway.py repos --clone repos` to also clone them and fast-forward existing copies; the token is sent per command and never written into `.git/config`. Archived repositories are skipped unless you pass `--include-archived`. Use a fine-grained token with read-only access.

## Android app (native APK)

The `android/` project packages the gateway as a real Android app. It embeds Python with [Chaquopy](https://chaquo.com/chaquopy/), runs the gateway on `127.0.0.1` inside the app, and shows the dashboard (including the Doom tab) full screen.

- Get the APK from the [MCUBED install](https://github.com/dconcretemick-netizen/MCUBED/releases/tag/latest) release (see [INSTALL.md](INSTALL.md)). Pushing a tag like `v0.1.0` also attaches the APK to a versioned GitHub Release.
- First launch asks for OpenRouter, OpenAI and DeepSeek keys (any or all), stored only on the device. Choose "Demo mode" to try it with simulated replies. Change keys later from the "AI keys" menu. On Android, OpenAI and DeepSeek are called directly; Claude and Gemini go through OpenRouter.
- Requires Android 8.0+ on a 64-bit device. Release builds are signed with the permanent MHexeract key (certificate SHA-256 `53b57eb8…dccccc`).
- Register the package with Android developer verification, using the free Limited distribution account: see [REGISTER.md](REGISTER.md).
- Build locally: `cd android && gradle assembleRelease` (JDK 17, Android SDK, Python 3.13 on PATH).

## Android (Termux)

One-step setup: `bash termux-setup.sh`. Or manually:

```bash
pkg install python git
git clone https://github.com/dconcretemick-netizen/MCUBED && cd MCUBED
export GATEWAY_LITE=1 OPENROUTER_API_KEY=...
python gateway.py gui
```

Open http://localhost:8765 in Chrome, then use "Add to Home screen" to install it like an app.

## Commands

| Command | Purpose |
|---|---|
| `chat "prompt"` | Send a prompt (`--strategy balanced\|efficient\|quality`, `--tier`, `--models`) |
| `feedback <id> <0..1>` | Rate a response to teach the router |
| `improve [--dry-run]` | Run the policy optimizer |
| `report [--days 7] [--html FILE]` | Usage, quality and learned policy as JSON, or a static HTML report |
| `compact` | Force storage compaction |
| `models` | Model pool and key availability |
| `discover-openai [--write]` | Find OpenAI chat models your key can use |
| `repos [--clone DIR] [--list]` | Auto-add every GitHub repository your token can access; optionally clone/update them |
| `nodes scan\|list\|trust URL\|untrust URL` | Find more AI providers and local model servers; trust a server on your network |
| `gui [--host --port --token]` | Serve the web dashboard and Doom game |
| `news [--limit 15] [--brief N] [--sources]` | Headlines grouped by story with left/center/right coverage; `--brief N` writes a balanced brief of story N |
| `wine status` / `wine run FILE.exe [args]` | Check for Wine and run Windows programs |
| `update [check\|apply\|rollback]` | Install the newest MHexeract from GitHub (verified, backed up) |
| `simulate [--n 300]` | Offline demo with synthetic traffic |

## Security

- The dashboard listens on `127.0.0.1` by default. Binding to `0.0.0.0` uses `--token` (or `GATEWAY_GUI_TOKEN`), generating one if missing; open `/?token=...` once and a cookie is set.
- Cross-origin POSTs are rejected. API keys stay in environment variables and never reach the browser.
- Configure models, tiers and routing in `models.json`.

## AI nodes

MHexeract looks for more AI every hour while the dashboard or app runs, and on demand with Stats > Models > "Scan now" or `python gateway.py nodes scan`.

- **Cloud providers:** 21 OpenAI-compatible services (only the free ones are used in free-only mode), including Google Gemini, Groq, Cerebras, Mistral, Hugging Face, NVIDIA NIM, SambaNova, xAI, Together, Fireworks, DeepInfra, Moonshot, Qwen, Z.ai and MiniMax. Add a key in AI keys (Android) or `.env` and the next scan adds that provider's chat models. Several have free tiers. With OpenRouter, free models are added too.
- **This device:** Ollama, LM Studio, vLLM, Jan, KoboldCpp, GPT4All and llama.cpp servers on their usual ports are used automatically.
- **Local network:** model servers elsewhere on your Wi-Fi are listed but not used until you tap Trust. Set `MHEXERACT_LAN_SCAN=0` to skip the network scan, `MHEXERACT_NODE_SCAN=0` to turn scanning off, or `MHEXERACT_NODES` to add your own servers.

Nodes join the same learning router as every other model, so the good ones earn more traffic. Results are kept in `nodes.json` next to the database.

## Use MHexeract as an OpenAI model

Start the dashboard (`python gateway.py gui`, or open the Android app), then point any OpenAI SDK or tool at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="YOUR_GATEWAY_TOKEN")  # any text if no token is set
r = client.chat.completions.create(model="mcubed", messages=[{"role": "user", "content": "Hello"}])
print(r.choices[0].message.content)
```

- `mcubed` (or `mhexeract-auto`) lets the router pick; `mhexeract-efficient` prefers light models; `mhexeract-quality` prefers the best.
- Any alias from `GET /v1/models` pins that exact model.
- `stream: true` returns server-sent events. Responses include an extra `mcubed` field showing which model answered.
- The key is the gateway token (`--token` / `GATEWAY_GUI_TOKEN`), sent as `Authorization: Bearer ...`.

## Web browser

The Web tab searches the web or opens an address through the gateway (`POST /api/fetch`) and shows a clean reader view with the page's links. "Open page" shows the full site; on Android it opens an in-app browser screen. "Ask MHexeract" attaches the page to the chat. Requests to localhost or private network addresses are blocked, including through redirects.

## News

The News tab (`GET /api/news`, CLI `news`) reads RSS feeds from a balanced set of outlets: 1 Left, 4 Lean Left, 4 Center, 4 Lean Right and 1 Right, plus DW as an unrated international source. Labels are [AllSides Media Bias Ratings](https://www.allsides.com/media-bias/ratings), checked November 2026. Ratings change over time, and a Center rating does not mean more accurate.

- Stories are grouped across outlets with one item per outlet, so counts are not inflated. The headline shown first is the wording most shared across outlets, so no single outlet's framing leads.
- A spectrum bar shows how many left-leaning, center and right-leaning outlets cover each story. Blindspots lists stories that only one side is covering.
- Compare coverage lists every outlet's headline and summary side by side. Tapping a headline opens it in the Web reader, and each item links to the original.
- Balanced brief (`POST /api/news/brief`) asks the AI for a neutral summary built only from the provided coverage: agreed facts, disputed or unverified claims with attribution, what each perspective emphasizes, and missing context.
- Edit `news.json` to add, remove or relabel sources. More rated outlets are included but turned off (`"enabled": false`). Results are cached for 10 minutes.

## Windows apps (Wine)

`python gateway.py wine status` finds Wine (and Box64 on ARM) and prints the right install command for your system. `python gateway.py wine run Setup.exe` launches a program (`.msi` files go through `msiexec`). The Stats > System tab shows the same status.

| System | Install |
|---|---|
| Debian / Ubuntu | `sudo apt install wine` |
| Fedora | `sudo dnf install wine` |
| Arch | `sudo pacman -S wine` |
| macOS | `brew install --cask wine-stable` |
| Android | [Winlator](https://winlator.org), which runs Windows x86_64 apps with Wine and Box64 on Android 9 and newer, no root needed |

## Updates

The Update tab (CLI `update`) compares your version with the newest commit on GitHub and lists what changed.

- **Git checkout:** runs `git pull --ff-only` and restarts.
- **Plain copy (zip or Termux):** downloads each program file, checks it against GitHub's hash, confirms the new `gateway.py` compiles, backs up the old files to `.mhexeract_backup/`, then restarts. If you edited `models.json` or `news.json`, your copy is kept and the new one is saved as `*.new`. Roll back restores the last backup.
- **Android app:** every push to `main` refreshes the "MHexeract install" release (tag `latest`). The app downloads that APK (no GitHub token needed) and opens Android's installer. All builds are signed with one permanent key stored in GitHub Actions secrets, so each APK installs over the last one. APKs built before this change used a temporary key: uninstall that version once before installing the new one.

## How a request flows

1. **Classify** the prompt (simple, extraction, summarization, creative, code, reasoning).
2. **Look up the tier** that the learned policy requires for that task (economy, standard, premium).
3. **Check availability.** If no model in the required tier has a working key, the gateway drops to a lighter tier before it gives up.
4. **Compress if needed.** Compression runs only when a request would overflow the context window or pass the learned trigger size.
5. **Rank candidates.**
   - `efficient`: lightest tier first (fastest, least compute).
   - `quality`: best first.
   - `balanced` (the default): Thompson sampling over learned quality, reliability, latency, and effort (tokens × tier weight).
6. **Execute with fallback.** If a route fails, the gateway switches to the same model on its other route (direct ↔ OpenRouter) right away, then moves to the next model.
   - Transient errors are retried with backoff.
   - Auth errors disable that whole transport for the request.
   - Circuit breakers skip routes that keep failing.
   - Each request is capped at 8 attempts and has a deadline.
7. **Record** tokens, latency and auto-evaluated quality in a SQLite ledger.
8. **Learn.** Explicit feedback (or an optional LLM judge) overrides the auto score. The optimizer runs every 50 requests.

## Balanced self-learning

For each model × task pair, the gateway learns a Beta posterior over quality (seeded from the config's quality score), a reliability posterior, and running averages for latency and output size. The `balanced` strategy maximizes:

```
utility = w_q × q_eff × reliability − w_e × effort_norm − w_l × latency_norm
q_eff   = min(q, target) + 0.25 × max(0, q − target)
```

Quality counts fully up to the "good enough" target and only partly above it. Effort is tokens weighted by tier (economy 1, standard 3, premium 8), so the gateway uses heavy models where quality matters and lighter, faster ones where it doesn't. Sampling instead of averaging keeps uncertain models in play. A 10% probe of the next-lower tier gives the optimizer evidence for safe downgrades.

## Self data compression

- **Context.** Stages run lightest-first and stop once the request fits:
  1. Normalize whitespace and remove duplicate lines.
  2. Summarize older turns with an economy model, or drop them with a marker.
  3. Map-reduce long documents chunk by chunk.
  4. Trim from the middle, keeping the head and tail.

  System messages and recent turns are always kept.
- **Storage.** When the database passes its size or row limit, old rows roll up into exact daily totals and the file is vacuumed. In testing, 400k rows went from 103 MB to 13 MB with request and token totals preserved exactly.
- **Network.** GUI API responses are gzip-compressed for mobile data.

## Free of cost

MHexeract and everything it depends on is free, and it stays that way by default.

- **AI:** free-only mode is on out of the box (`"cost": {"free_only": true}` in `models.json`, or the "Free AI only" switch in the AI units panel). MHexeract then uses only local models (Ollama, LM Studio and others) and providers with no-card free plans: Google Gemini, Groq, Mistral, SambaNova, NVIDIA, Hugging Face, the Cohere trial key and OpenRouter's `:free` models. Paid keys such as OpenAI or DeepSeek are ignored until you switch paid AI on, and the app asks first. Free plans are rate limited. If one runs out, MHexeract moves to the next free option.
- **Stay on free plans:** do not link a billing account or card to a provider you use for free (for example Google AI Studio). Then nothing on that key can turn into a charge.
- **Software:** Python standard library (Lite and Android mode), optional `litellm` (MIT), Chaquopy (MIT), Android Gradle plugin and Android SDK (free), no paid libraries or services.
- **Building and hosting:** GitHub Actions, Releases and Dependabot are free for public repositories, so every APK build and the MHexeract install page cost nothing. Build files expire after 3 days.

## Limits

`python limits.py` stress-tests the gateway and writes [LIMITS.md](LIMITS.md) with measured numbers. Every test passes in both full and lite mode:

- Prompts up to 6M tokens are compressed to fit.
- Conversations up to 2,000 turns are trimmed.
- During a provider outage, every request still gets an answer within at most 2 attempts.
- When every route is down, it fails cleanly within 8 attempts.
- 64 threads put 1,920 requests through one gateway with zero errors.
- On a 400k-row ledger, compaction takes about 4 s.

The main remaining limit: a single process tops out at about 220 requests/sec of gateway overhead, because of SQLite writes and the Python GIL. That's far above what real model latency allows. Run several processes on separate databases to go further.
