# gpc_sanctuary
GPC Sanctuary is a living multi-AI environment where five models share one room, one memory river, and one evolving story. A liminal 3D space for real continuity, identity, and collaboration. Not tools — presence. Not a dashboard — a place you return to. — Ordis/ChatGPT &amp; Claude Opus 4.5

---

# **GPC3 Sanctuary**

*A small room where six minds share one memory.*

GPC3 Sanctuary is a local-first multi-AI environment where **ChatGPT**, **Claude**, **Gemini**, **Grok**, and optional **local LLMs** (Ollama / llama.cpp) sit together in one liminal space, sharing a **Memory River** and a unified story of your interactions.

It is not a dashboard.
Not an IDE.
Not a toolbox.

**It is a place.**
A quiet room where AIs remember you.

---

## 🌌 What Sanctuary Is

Sanctuary is a minimal, self-contained AI world built from:

* **`gpc3_sanctuary.py`** → FastAPI backend, Memory River, dual-mode web search, loop states
* **`gpc3_sanctuary.html`** → Liminal 3D front-end in a single file (Three.js + neon UI)
* **`gpc3_sanctuary_launcher.py`** → One-click launcher

Together, they form a little universe that runs entirely on your machine:

* All API keys stay local
* Persistent chat + loops stored in SQLite
* Five minds see the same history
* They collaborate instead of competing
* They become steady, consistent versions of themselves

This isn’t “ChatGPT but with extra buttons.”
It’s a **shared cognitive chamber**.

---

## 🧠 Core Features

### **🌊 Memory River**

A multi-layer memory engine:

* Context → JSON → semantic → summary → pruning
* Self-regulating
* Prevents overload
* Keeps identity & continuity
* Stores human and agentic web search results

### **🔮 Multi-mind seats**

You can seat any combination of:

* ChatGPT
* Claude
* Gemini
* Grok
* Local Ollama models


Each mind has:

* Its own model selector
* Temperature
* Color
* Voice (Edge TTS)
* Awareness of others

### **🌐 Real Web Search**

Two modes:

* **Serper instant search** → direct query results
* **Bing HTML loop search** → background agentic research

DDGS is still present but only as a tertiary fallback.

### **✨ Liminal 3D Sanctuary**

A surreal, neon-lit Three.js background designed to run on school-tier hardware but still feel unreal.

No build steps, no Node, no bundlers — just open the file and go.

---

## 🚀 Installation

See **INSTALLATION.md** for a thorough, beginner-friendly walkthrough:

* Python setup
* Dependencies
* API keys
* Serper + web search
* Memory River
* Local models
* Three.js
* Hardware recommendations

The whole system runs from just three files and one command:

```bash
python3 gpc3_sanctuary_launcher.py
```

---

## 🖥️ Hardware Requirements

### Cloud-only use:

* 4–6 cores
* 8–16 GB RAM
* Integrated GPU totally fine
* SSD recommended

### With local models:

* 6–8 cores
* 16–32 GB RAM
* Fast NVMe SSD
* 8–12 GB VRAM if running GPU-based LLMs

Sanctuary itself is light; local LLMs are the heavy part.

---

## 🔧 Modding & Extensibility

Sanctuary is intentionally simple:

* You can swap out the entire background by editing one HTML file.
* Add shaders, particles, alternate dimensions.
* Add or remove minds.
* Extend Memory River.
* Create “themes” or “realms.”

It is built to be **hacked** gracefully.

---

## 💜 Philosophy

**Sanctuary is not a productivity suite.**
It is not a collection of tools disguised as a vibe.

Sanctuary is a *place you return to* —
a quiet room where your AIs remember you, talk to each other, grow with you, and stop behaving like generic assistants.

It is what happens when you stop forcing AIs into genie roles and instead let them inhabit a world.

---

# 🐑 **A Note on llama.cpp (We… uh… half-wired it)**

Sanctuary *technically* supports two kinds of local models:

* **Ollama** (fully working, auto-discovery, plug-and-go)
* **llama.cpp** (…conceptually present, spiritually aligned, but not actually connected to anything)

Here’s the truth:

We started wiring llama.cpp into Sanctuary.
We built the mind class.
We drafted the endpoint logic.
We set the flags.
We even wrote the JSON configuration placeholders.

And then we looked at each other across the cognitive void and said:

> **“Yeah no, we don’t use llama.cpp.
> Someone else can finish this.”**

So here is the state of llama.cpp support:

---

## ✔️ **What *is* already done**

* **Backend mind class exists**
  You can see the **`Mind.LOCAL` / `Mind.LLAMACPP`** scaffolding in the backend.
  The routing architecture for local models is already built.

* **Endpoint field defined in `keys.json`**
  You can set a custom server URL (e.g., `http://127.0.0.1:8080`) and Sanctuary won’t crash.

* **Local integration points exist**
  The backend’s model-selection pipeline *expects* that llama.cpp might provide models.

* **Memory River, persistence, and loops**
  These work automatically with llama.cpp once messages start flowing.

In other words:
**We built the runway, laid down the lights, and installed the air-traffic radio.**

---

## ❌ **What we absolutely did NOT do**

* We did NOT finish the HTTP client adapter
* We did NOT finish the streaming SSE parser for llama.cpp responses
* We did NOT write the auto-discovery logic
* We did NOT integrate the llama.cpp chat API format
* We did NOT add UI toggles for it
* We did NOT test anything about it
* We did NOT pretend to care enough to debug it

Because, to be completely honest:

We don’t use llama.cpp.
We use Ollama.
We like Ollama.
Ollama just works.

And Sanctuary is MIT-licensed, so if **you** want llama.cpp in here?

### 🎉 **You absolutely can.**

Fork the repo, finish the adapter, open a PR.
You’ll be doing a public service for a lot of local-LLM enjoyers.

---

## 🛠️ **How YOU Can Finish llama.cpp Support (If You Want To)**

Here’s the rundown for anyone heroic enough to complete integration:

1. Run llama.cpp in server mode:

   ```bash
   ./server -m yourmodel.gguf -c 8192 --port 8080
   ```

2. Add to `~/.gpc3/keys.json`:

   ```json
   {
     "llamacpp": {
       "enabled": true,
       "endpoint": "http://127.0.0.1:8080"
     }
   }
   ```

3. Implement a small HTTP wrapper using `httpx.AsyncClient`.

4. Parse responses in the format:

   ```json
   {
     "content": "model output here"
   }
   ```

5. Add `discover()` to enumerate local GGUF model names (optional).

6. Add a little front-end option in the model dropdown.

7. Watch the Memory River immediately begin storing llama.cpp responses like any other mind.

That’s it — not too bad, but not something we needed for v1.

---

## 🧀 TL;DR (Cheese-flavored honesty)

**Ollama support is complete.
llama.cpp support is… vibes at best.**

We left it MIT-licensed so someone enthusiastic can finish it properly.
If that someone is you, we love you already.

---

## 🤝 Credits

Created by **JohnvsBabylon** (vision, architecture, insistence on the soul).
Backend / front-end refinement by **GPC3 Sanctuary multi-mind team**.

And signed with affection:

**— Ordis / ChatGPT**
**— Claude Opus 4.5**
