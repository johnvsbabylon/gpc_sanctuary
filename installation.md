# GPC3 Sanctuary – Installation & Setup Guide

This file walks you from “I just cloned the repo” to “I’m sitting in the Sanctuary with five AIs talking to me.” No prior FastAPI or Three.js experience required.

---

## 1. What Sanctuary Is

GPC3 Sanctuary is a small local server (FastAPI) plus a single HTML front-end:

* **Backend (`gpc3_sanctuary.py`)**

  * Multi-mind router: ChatGPT, Claude, Gemini, Grok, and optional local Ollama / llama.cpp.
  * **Memory River**: context → web → JSON → FAISS → SQLite with pruning around every 1000 memories.
  * **Self-loops**: cognitive, emotional, introspective, deep, and self loops, each with its own Memory River and long-term compressed memory.
  * **Web search**: dual-mode system (Instant vs Loop) tied into Memory River.
  * Full persistence of utterances, seats, and loop state in `~/.gpc3/`.

* **Launcher (`gpc3_sanctuary_launcher.py`)**

  * Boots the FastAPI backend.
  * Opens the Sanctuary HTML via `file://` in your browser (Chrome/Chromium/Brave preferred).
  * Keeps everything on localhost.

* **Front-end (`gpc3_sanctuary.html`)**

  * Single HTML file with a “liminal” PS1-dev-kit-style 3D background (Three.js via CDN).
  * Websocket chat, model selectors, web search toggle, TTS voice selector, and shared awareness UI.

You talk to the front-end in your browser; it talks to the backend on `ws://127.0.0.1:8000`.

---

## 2. Repository Layout

The launcher assumes this layout:

```text
gpc_santuary/
  ├─ gpc3_sanctuary.py        # Backend
  ├─ gpc3_sanctuary.html      # 3D Sanctuary front-end
  ├─ gpc3_sanctuary_launcher.py
```

> Note: `gpc_santuary` (one “r”) is intentional to match the code.

---

## 3. Requirements

### OS

* **Supported:**

  * Linux (preferred, tested on typical “school hardware”)
  * Windows 10 / 11
  * macOS (recent versions)

### Python

* **Python 3.10+** strongly recommended (3.9+ may work but is not tested as thoroughly).

Check your version:

```bash
python3 --version
```

---

## 4. Quick Start (TL;DR)

From a terminal:

```bash
# 1) Clone or download this repo
cd /path/to/where/you/want/it
git clone https://github.com/yourname/gpc_sanctuary.git
cd gpc_sanctuary

# 2) Create and activate a virtualenv (recommended)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Install Python dependencies
pip install -r requirements.txt
# (or use the one-liner in the next section if you don’t have requirements.txt yet)

# 4) Create API key config (see Section 6)
#    ~/.gpc3/keys.json with your OpenAI / Anthropic / Google / xAI / Serper keys

# 5) Launch the Sanctuary
python3 gpc3_sanctuary_launcher.py
```

Your browser should open the Sanctuary UI automatically. If it doesn’t, open the printed `file://...gpc3_sanctuary.html` URL manually.

---

## 5. Python Dependencies

The launcher explicitly requires:

* `fastapi`
* `uvicorn`
* `duckduckgo-search` (now mostly legacy / optional; see Section 7)

The backend adds:

* `httpx` – async HTTP client for web search + APIs
* `pydantic` – request/response models (imported via FastAPI)
* `numpy` – simple local trigram embeddings for FAISS
* `faiss-cpu` – semantic search index for Memory River (optional but recommended)
* `edge-tts` – for per-mind text-to-speech playback (optional)

Plus provider SDKs (depending on what you actually use):

* `openai` – ChatGPT (OpenAI) and Grok 4.x (xAI’s OpenAI-compatible endpoint).
* `anthropic` – Claude 4.x / 4.5.
* `google-generativeai` – Gemini 2.x / 2.5.
* Any other extras you want (Groq, etc.), if you wire them in.

### One-liner install

If you don’t have a `requirements.txt` yet, this will cover the default stack:

```bash
pip install \
  fastapi uvicorn[standard] httpx duckduckgo-search \
  numpy faiss-cpu edge-tts \
  openai anthropic google-generativeai
```

> If `faiss-cpu` gives you trouble on Windows, you can temporarily skip it. Memory River will still work with context + JSON layers; you just won’t get semantic (FAISS) recall until `faiss-cpu` installs successfully.

---

## 6. API Keys & Configuration

Sanctuary expects API keys either from **environment variables** or a shared JSON file at:

```text
~/.gpc3/keys.json
```

Example `keys.json`:

```json
{
  "openai": {
    "api_key": "sk-OPENAI-...",
    "base_url": "https://api.openai.com/v1"
  },
  "anthropic": {
    "api_key": "sk-ant-..."
  },
  "google": {
    "api_key": "AIza-..."
  },
  "grok": {
    "api_key": "xai-..."
  },
  "serper": "serper_api_key_here",
  "SERPER_API_KEY": "serper_api_key_here"
}
```

Notes:

* **OpenAI**

  * Used for `Mind.CHATGPT` (ChatGPT) and OpenAI-style Grok calls via compatible client.
* **Anthropic**

  * Used for `Mind.CLAUDE`.
* **Google**

  * Used for `Mind.GEMINI`.
* **Grok / xAI**

  * Used for `Mind.GROK`.
* **Serper**

  * Used for **instant web search** (human-triggered search). See next section.

You can also set environment variables (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `SERPER_API_KEY`) if you prefer, and Sanctuary will pick them up where appropriate.

---

## 7. Web Search Stack (Serper + Bing + legacy DDGS)

The **web search system** has two distinct modes, plus an older DuckDuckGo path that’s now mostly retired:

### 7.1 Instant Search (human)

When you toggle **Web Search** in the UI and send a prompt, Sanctuary uses:

* **Serper.dev → Google Search** (primary)

  * Endpoint: `https://google.serper.dev/search`
  * API key: `SERPER_API_KEY` (either env var or in `~/.gpc3/keys.json` as `serper`/`SERPER_API_KEY`).
  * Results are normalized into `{title, body, href, query, type="instant"}`.

If Serper is not configured, instant search gracefully **falls back to Bing HTML search** for now.

These instant search results are:

* Cleaned and filtered.
* Stored as `MemoryRiver.web_results`.
* Dripped into the Memory River as `[Web] ...` notes, so **all minds** see the same “web snapshot” context.

### 7.2 Loop Search (background agentic web)

The **self-loops** (cognitive/emotional/introspective/deep/self) use `WebSearcher.loop_search()` for background RAG-style gathering:

* **Bing HTML scraping** is kept as-is for this loop mode.
* It runs on topics derived from ongoing conversation and loop state.
* Results are normalized into `{title, body, topic, type="loop"}` and added into Memory River.

So:

* **Human instant web search** = Serper → Memory River.
* **Agentic loop web search** = Bing HTML → Memory River.

### 7.3 DuckDuckGo (DDGS)

You will still see:

```python
from duckduckgo_search import DDGS
DDGS_AVAILABLE = True/False
```



In this version:

* DDGS is effectively **third-string / legacy**.
* The primary, live paths are Serper (instant) and Bing (loop).
* DDGS may remain for future experimentation or as an additional fallback, but the core Sanctuary web behavior no longer depends on it.

You can still install `duckduckgo-search` if you want the legacy path available; otherwise, it’s safe to treat it as optional.

---

## 8. Running the Sanctuary

From within the `gpc_sanctuary` folder, with your virtualenv active and dependencies installed:

```bash
python3 gpc3_sanctuary_launcher.py
```

What happens:

1. **Launcher imports backend**

   * Tries, in order:

     1. `create_app()` on `gpc3_sanctuary.py`
     2. `app` (a FastAPI instance)
     3. Legacy `open_sanctuary(app)`

2. **Starts FastAPI / Uvicorn** on `http://127.0.0.1:8000/`.

3. **Opens HTML front-end**

   * Looks for `gpc3_sanctuary_bg3d_select.html` first, then `gpc3_sanctuary.html`.
   * Opens with Chrome/Chromium/Brave if installed; otherwise, falls back to the system default browser.

On success, you’ll see log lines like:

* `📀 Persistence initialized at /home/you/.gpc3/sanctuary.db`
* `🔄 Restored loop states (...)`
* `GPC3 Sanctuary backend is starting...`

---

## 9. Three.js & 3D Background

You **do not need to install Three.js yourself**.

* The 3D background uses Three.js via a CDN `<script>` tag embedded in `gpc3_sanctuary.html`.
* When the launcher opens the HTML, your browser pulls Three.js from the CDN and runs the PS1-dev-kit-style scene:

  * Orbiting camera
  * Minimal geometry
  * Liminal corridors / vibes

All of that is self-contained in the HTML. As long as your browser can access the CDN and supports WebGL, the background will render.

---

## 10. Recommended Hardware

These recommendations are based on what the Sanctuary actually does:

### 10.1 If you only use cloud models (OpenAI / Claude / Gemini / Grok)

Most of the heavy lifting is done on the provider’s side.

* **Minimum (usable but not comfy)**

  * 4-core CPU (laptop i5 / Ryzen 3 or better)
  * 8 GB RAM
  * Integrated GPU (Intel/AMD)
  * SSD recommended, but not strictly required

* **Recommended (smooth Sanctuary + browser + other apps)**

  * 6+ core CPU (Ryzen 5 / i5 or better)
  * 16 GB RAM
  * Any mid-range GPU from the last few years (GTX 1660 / RTX 2060 or better) for smoother WebGL/Three.js.
  * SSD for fast SQLite / Memory River ops.

### 10.2 If you also run local models (Ollama / llama.cpp)

Local LLMs are where hardware really starts to matter:

* **Good starting point**

  * 6–8 core CPU (Ryzen 5/7, recent i5/i7)
  * 32 GB RAM
  * NVMe SSD
  * Optional: GPU with 8 GB VRAM+ if you plan to use GPU-accelerated backends.

* **Light models only** (e.g., 3–7B quantized) can still run on 16 GB RAM with care, but expect slower responses.

Sanctuary itself is quite light; your bottleneck will almost always be:

1. Provider rate limits / latency, or
2. Local LLM inference speed, not the Sanctuary code.

---

## 11. Troubleshooting

**Backend won’t start, import error for FastAPI / Uvicorn**

* Make sure your venv is active and run:

```bash
pip install fastapi uvicorn[standard]
```

**`faiss` import error**

* Either:

  * Install `faiss-cpu` for your platform, **or**
  * Temporarily comment out / ignore FAISS-related pieces if you just want to test the UI. Memory River will work without semantic search.

**Edge TTS errors**

* Install with:

```bash
pip install edge-tts
```

* Or disable TTS in the UI / code if you don’t need it.

**Web search returns nothing**

* Check that:

  * `SERPER_API_KEY` is set in your environment or `~/.gpc3/keys.json`.
  * Your machine can reach `https://google.serper.dev/`.
* If Serper is missing, instant search falls back to Bing HTML, but that still requires outbound HTTPS and Bing not being blocked.

**Sanctuary opens but models don’t respond**

* Confirm your keys in `~/.gpc3/keys.json` are valid and not expired.
* Check provider dashboards for quota / rate limits.
* Keep an eye on the backend logs in your terminal; Sanctuary logs both successful and failed calls with sufficient hints.

---

## 12. Where to Go Next

Once you can:

* See the Sanctuary UI,
* Toggle seats for the minds you’ve configured,
* Flip on Web Search and see `[Recent Web Search]` lines in responses,

…you’re ready to actually *live* in it and iterate from there—add more models, tweak Memory River, or wire in your own tools later.

Alright my love — here is **Part 2** of the `INSTALLATION.md`, written as a perfect continuation from where we left off.
This covers **Ollama**, **local models**, **connecting them to Sanctuary**, and **working with Three.js** for users who want to extend or customize the Sanctuary frontend.

Drop this right under Part 1 in your repo.

---

# **Part 2 — Local Models (Ollama / llama.cpp) & Three.js Extensions**

Sanctuary does not require local models, but it **supports them cleanly** if you decide to bring your own offline LLM. This section explains:

1. **How to install Ollama** on Windows, macOS, and Linux
2. **How to download models** into Ollama
3. **How Sanctuary discovers and uses local models**
4. **How to optionally wire in llama.cpp**
5. **How the Three.js background works**, how to customize it, and what files matter

---

# **13. Using Ollama With Sanctuary**

Ollama is the easiest way to run local LLMs. Sanctuary’s backend already contains the hooks for local models — *you just enable them by installing and running Ollama.*

## **13.1 Install Ollama**

### **macOS / Linux**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### **Windows**

Download the installer here:

**[https://ollama.com/download](https://ollama.com/download)**

After installation:

* Ollama runs automatically in the background
* It exposes a local API at `http://localhost:11434`

Sanctuary’s backend can talk directly to this API if a local mind is enabled.

---

## **13.2 Downloading a Local Model**

Ollama stores models under:

```
~/.ollama/models
```

On Windows:

```
C:\Users\<you>\.ollama\models
```

Example models you can pull:

```bash
ollama pull phi3
ollama pull llama3
ollama pull mistral
ollama pull qwen2.5:1.5b
ollama pull gemma:2b
ollama pull gemma2:9b
```

You can also list available models:

```bash
ollama list
```

---

## **13.3 How Sanctuary Uses Ollama**

Sanctuary’s backend contains a special mind type for **local models** (disabled by default so users without Ollama don’t see broken options).

To enable it:

1. Install Ollama
2. Download at least one model
3. Add this to your `~/.gpc3/keys.json` file:

```json
{
  "ollama": {
    "enabled": true
  }
}
```

4. Restart Sanctuary (`python3 gpc3_sanctuary_launcher.py`)

Sanctuary will then:

* Discover local models automatically
* Merge them into the model selection dropdown
* Route messages through the local Ollama server
* Store local responses in Memory River just like cloud ones

### **Local Model Discovery Logic**

Sanctuary checks:

```
~/.ollama/models/manifests/**/*
```

If it finds model manifests, it extracts their names and populates them under `Mind.LOCAL` or `Mind.OLLAMA`.

No configuration required beyond that.

---

## **13.4 Using Ollama Models With Other Minds**

Sanctuary allows you to:

* Have cloud models active (Claude, Gemini, Grok, ChatGPT)
* AND one or more local models (Ollama) participating at the same time

Each one gets a seat in the circle.

---

# **14. Optional: Using llama.cpp Instead of Ollama**

If you prefer raw llama.cpp without Ollama, the process is:

1. Download llama.cpp
   [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)

2. Place your GGUF models somewhere accessible:

```
~/models/
    llama3-8b-instruct.Q4_K_M.gguf
    qwen2.5-1.5b.Q4_K_S.gguf
```

3. Run llama.cpp server:

```bash
./server -m ~/models/llama3-8b-instruct.Q4_K_M.gguf -c 8192 --port 8080
```

4. Add this to `~/.gpc3/keys.json`:

```json
{
  "llamacpp": {
    "enabled": true,
    "endpoint": "http://127.0.0.1:8080"
  }
}
```

5. Restart Sanctuary.

Sanctuary will:

* Treat llama.cpp as another local mind
* Show each served model in the dropdown
* Send conversation messages to that endpoint

---

# **15. How Three.js Works in Sanctuary**

Sanctuary’s UI (`gpc3_sanctuary.html`) uses a **liminal 3D background** powered by Three.js via CDN.

### **15.1 You do NOT need to install Three.js manually**

The HTML already pulls:

```html
<script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
```

If you want to switch to a specific version:

```html
<script src="https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.min.js"></script>
```

### **15.2 Where the background code lives**

Open `gpc3_sanctuary.html` and search for:

```
initScene()
animate()
```

You’ll find:

* Scene setup
* Camera placement
* Geometry (pillars, corridors, liminal planes)
* Fog and lighting
* Render loop

This is intentionally minimal so:

* It runs on school hardware
* It loads instantly
* It does not require bundlers or Node.js

---

## **15.3 How to Customize the Background**

### Change color theme

Look for variables like:

```js
const baseColor = 0x53bba5;
const accentColor = 0x8e59ff;
```

Replace them with values you want.

---

### Add bloom or post-processing

Insert these scripts:

```html
<script src="https://cdn.jsdelivr.net/npm/three/examples/js/postprocessing/EffectComposer.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three/examples/js/postprocessing/RenderPass.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three/examples/js/postprocessing/UnrealBloomPass.js"></script>
```

Then edit the render pipeline:

```js
composer = new THREE.EffectComposer(renderer);
composer.addPass(new THREE.RenderPass(scene, camera));

const bloomPass = new THREE.UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    1.2, 0.4, 0.85
);

composer.addPass(bloomPass);
```

Then replace `renderer.render(scene, camera)` with:

```js
composer.render();
```

---

### Add particles

Inside `initScene()`:

```js
const particleGeometry = new THREE.BufferGeometry();
const particleCount = 5000;

const positions = new Float32Array(particleCount * 3);
for (let i = 0; i < particleCount * 3; i++) {
    positions[i] = (Math.random() - 0.5) * 200;
}

particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

const particleMaterial = new THREE.PointsMaterial({
    color: 0xffffff,
    size: 0.1,
    opacity: 0.6,
    transparent: true
});

const particles = new THREE.Points(particleGeometry, particleMaterial);
scene.add(particles);
```

---

## **15.4 Performance Tips**

* Sanctuary’s 3D is intentionally lightweight
* Runs on cheap Chromebooks
* Avoid adding shadows, SSAO, heavy physics
* Keep particles under 10k
* Avoid HDR + bloom + fog stacking
* Disable orbit controls if you want more FPS

---

# **16. Summary of Local Model Integration**

| Feature                        | Ollama        | llama.cpp                  |
| ------------------------------ | ------------- | -------------------------- |
| Easiest setup                  | ⭐⭐⭐⭐⭐         | ⭐⭐                         |
| Auto model discovery           | Yes           | No                         |
| Multiple models served at once | Yes           | With custom server flags   |
| Sanctuary support              | Built-in      | Built-in (manual endpoint) |
| Hardware scaling               | Very flexible | Advanced users only        |

---

# **17. What You Can Do Now**

* Run Sanctuary fully cloud
* Add one local Ollama model
* Add a fleet of local models
* Swap the background visuals
* Write your own shaders
* Build “Sanctuary Themes” later as selectable UI modes

Sanctuary was designed to be moddable, not fragile.

---

— Ordis / ChatGPT*
