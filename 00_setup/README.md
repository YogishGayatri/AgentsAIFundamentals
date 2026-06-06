# Day −1 · Setup (run this the day before)

The goal of this folder is boring on purpose: **by the end, everyone in the room
can call an LLM from Python.** No LangChain concepts yet — just keys, an
environment, and one successful round-trip. If this works today, Day 1 is smooth.

Budget: ~15 minutes.

---

## 1. Get a (free) API key

You only need **one** provider. We default to **Groq** because it's free and fast.

| Provider | Where to get a key | Put it in `.env` as |
|----------|--------------------|---------------------|
| **Groq** (default) | https://console.groq.com/keys | `GROQ_API_KEY` |
| Gemini (optional) | https://aistudio.google.com/apikey | `GOOGLE_API_KEY` |
| FM Gateway (optional) | internal 3DS gateway | `FM_API_KEY` |

## 2. Create your environment

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Add your key

```bash
cp .env.example .env
# open .env and paste your key next to GROQ_API_KEY
```

Leave `PROVIDER=groq`. (Later you can flip it to `gemini` or `fm` — every
script reads from here, so that one line is the only thing you change.)

## 4. Verify everything

```bash
python 00_setup/check_setup.py
```

You should see all green checks. If a check fails it tells you exactly what to fix.

## 5. Your first LLM call

```bash
python 00_setup/hello_llm.py
```

This is the whole point of today: **text in, text out, from real code.**
Tomorrow we wrap structure around exactly this call.

---

## Troubleshooting

**`CERTIFICATE_VERIFY_FAILED: self signed certificate in certificate chain`**
(usually Gemini or FM Gateway on a corporate/managed machine)

Your company's network intercepts HTTPS with its own root CA, which Python
doesn't trust by default. Fix it by trusting the OS certificate store (where IT
already installed that CA) — **not** by disabling verification:

```bash
pip install truststore
```

`providers.py` auto-enables it when present. Re-run `check_setup.py`.
*(It's already in `requirements.txt`, so a fresh `pip install -r requirements.txt`
includes it.)* If a provider stays unreachable afterward, it may be blocked on
your network entirely — just use **Groq**, which is all the course needs.

**`GROQ_API_KEY is missing` even though you pasted it** — your `.env` has a
second `GROQ_API_KEY=` line (often the leftover `your_groq_key_here` placeholder).
dotenv uses the *last* line. Keep exactly one.

---

### The mental model for the week

> Foundations day: you hand-wrote the agent loop.
> **Day 1 (LangChain):** the model / message / tool layer + the easy button.
> **Days 2–3 (LangGraph):** rebuild that loop as a graph *you* control.

Everything on Day 1 should land as *"oh — this is my raw loop, formalized."*
