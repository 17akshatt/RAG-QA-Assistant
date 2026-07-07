# RAG Q&A Assistant — Build Guide

> **What this project is:** a "chat with your documents" web app. You give it documents, and it answers questions **using only those documents** — with citations, and an honest "I don't know" when the answer isn't there. This is the backbone skill of modern AI engineering (it's called **RAG** — Retrieval-Augmented Generation).

> **Who this is for:** the human building this (AJ) is a **beginner in AI**. He knows some Python. He does **not** know AI concepts yet. Explain as you go.

---

## How to use this document (AJ — read this first)

1. Save this file as **`CLAUDE.md`** in an empty project folder. Claude Code reads `CLAUDE.md` automatically every session, so it always has this plan.
2. Do the **two "Decisions" below** yourself before you start building. They're the only two things Claude Code can't do for you.
3. Open Claude Code in that folder and say: *"Read CLAUDE.md and start Phase 0. Stop at the checkpoint so I can test."*
4. Build **one phase at a time**. Test at each checkpoint before moving on. Don't let it race ahead.
5. If Claude Code ever asks you a question you don't understand, check **Part 6 (FAQ)** — it's probably pre-answered there. If not, paste the question to your mentor (the chat you got this doc from).

---

## Part 1 — Two things only YOU can do first

### Decision A — Get your free Gemini API key (5 minutes)

1. Go to **aistudio.google.com** → sign in with a Google account.
2. Click **Get API key** → **Create API key** (choose "in a new project" for a clean start).
3. **Copy the key immediately** and paste it somewhere safe for a moment.
4. No credit card needed. This is the free tier — fine for learning and this whole project.

You'll put this key into a `.env` file in Phase 0 (Claude Code will set that up). **Never paste the key directly into code, and never commit it to GitHub.**

### Decision B — Pick your documents

The app needs something to answer questions *about*. Two options:

- **Recommended for a strong demo:** the technical documentation of any tool you like (e.g. download a few pages of a library's docs as `.txt`, `.md`, or `.pdf`). This makes the finished app read as a **developer tool** — great for a resume.
- **Zero-effort fallback:** tell Claude Code *"create 2–3 sample text files in `docs/` for now"* and swap in real ones later.

Either way, the documents live in a folder called **`docs/`** in the project.

---

## Part 2 — Locked tech stack (Claude Code: use EXACTLY this — do not substitute or ask)

| Layer | Choice | Notes |
|---|---|---|
| Language | **Python 3.11+** | |
| LLM (answer generation) | **Google Gemini** via the **`google-genai`** SDK | Model: `gemini-2.5-flash` (free tier). **Do NOT use the deprecated `google-generativeai` package.** |
| Embeddings | **ChromaDB's built-in local embedding function** | No separate embedding API or key. Chroma embeds documents automatically on its own. |
| Vector database | **ChromaDB**, persistent (`./chroma`) | Local, free, saves to disk. |
| Document reading | **`pypdf`** for PDFs; plain read for `.txt`/`.md` | |
| Secrets | **`python-dotenv`** + a `.env` file | Key name: `GEMINI_API_KEY` |
| UI | **Streamlit** | |
| Deployment | **Streamlit Community Cloud** (free) | |
| Version control | **Git + GitHub** | |

**Explicitly OUT of scope (do not add unless AJ asks):** LangChain, LlamaIndex, or any RAG framework (we write it directly so AJ *sees* every step); user login/auth; a separate database; Docker; a testing framework; async. Keep the codebase small and readable.

---

## Part 3 — Rules for Claude Code (how to work with a beginner)

1. **Explain before you do.** Before each phase, say in plain English *what* you're about to build and *why*, with no unexplained jargon. Define every new AI term the first time it appears.
2. **Work one phase at a time.** At the end of each phase, **stop at the checkpoint** and tell AJ the exact command to run and exactly what he should see. Wait for him to confirm it works before continuing.
3. **Never ask an open-ended question.** If you genuinely need a decision, give **2–3 concrete options with your recommendation**, and default to the recommendation if AJ is unsure. Most decisions are already made in Part 2 — check there first.
4. **Comment the code** so a beginner can read it. Short, plain comments explaining what each block does.
5. **Security is non-negotiable:** the `.env` file must be listed in `.gitignore` from the very first commit. Never print the API key. Never commit it.
6. **Prefer clarity over cleverness.** Simple, boring, readable code beats a clever one-liner.
7. **When something errors**, explain what the error means in plain English and how you're fixing it, so AJ learns to debug.

---

## Part 4 — Target project structure

```
rag-qa-assistant/
├── CLAUDE.md            # this file
├── .env                 # your secret API key (NEVER committed)
├── .gitignore           # must ignore .env, venv/, __pycache__/, chroma/
├── requirements.txt     # dependencies
├── docs/                # your source documents (Decision B)
├── llm.py               # thin wrapper around the Gemini call
├── ingest.py            # loads docs → chunks → stores in Chroma
├── rag.py               # the core: retrieve chunks + ask the LLM
├── app.py               # the Streamlit web app
├── eval.py              # small evaluation script
├── eval_questions.json  # test questions for eval
└── README.md            # written in Phase 7
```

---

## Part 5 — The build, phase by phase

Each phase lists: **Goal**, what **Claude Code builds**, the **Checkpoint** (what AJ runs + sees), and the **Concept** AJ is learning.

### Phase 0 — Setup & your first LLM call
- **Goal:** a working environment and proof the Gemini key works.
- **Claude Code builds:** a virtual environment; `requirements.txt` (`google-genai`, `chromadb`, `pypdf`, `streamlit`, `python-dotenv`); a `.gitignore` (ignoring `.env`, `venv/`, `__pycache__/`, `chroma/`); a `.env` file with `GEMINI_API_KEY=` (AJ pastes his key); initialize a git repo; and a tiny `test_llm.py` that loads the key and sends one prompt to Gemini.
- **The correct, current SDK call pattern** (use this, not the old package):
  ```python
  from google import genai
  client = genai.Client()  # reads GEMINI_API_KEY from the environment
  resp = client.models.generate_content(
      model="gemini-2.5-flash",
      contents="Say hello in one sentence."
  )
  print(resp.text)
  ```
- **Checkpoint:** AJ runs `python test_llm.py` and sees a one-line reply from the model.
- **Concept for AJ:** An "LLM API call" is just sending text to a model over the internet and getting text back. The `.env` file keeps your secret key out of your code so you never accidentally share it.

### Phase 1 — Talking to the model properly
- **Goal:** understand how to control the model.
- **Claude Code builds:** `llm.py` with one reusable function, e.g. `ask(prompt, system_instruction=None)`, that wraps the Gemini call. A short demo showing the same question answered with different `system_instruction` values and `temperature` settings.
- **Checkpoint:** AJ runs the demo and sees how a system instruction changes the model's behavior.
- **Concept for AJ:** The **system instruction** sets the model's role/rules. **Temperature** controls randomness (low = focused, high = creative). This is "prompt engineering."

### Phase 2 — Embeddings (the key idea)
- **Goal:** *see* what an embedding is before using it.
- **Claude Code builds:** `embeddings_demo.py` that uses Chroma's embedding function to turn ~4 sentences into vectors and prints how similar each pair is (e.g. cosine similarity).
- **Checkpoint:** AJ sees that *"I love dogs"* and *"puppies are wonderful"* score as **similar**, while *"the stock market crashed"* scores as **different** — even though the words don't overlap.
- **Concept for AJ:** An **embedding** turns text into a list of numbers that represents its *meaning*. Similar meanings land close together. This is how the app finds relevant text even when the question uses different words than the document.

### Phase 3 — Build the index (ingestion)
- **Goal:** load your documents into the vector database.
- **Claude Code builds:** `ingest.py` that reads every file in `docs/` (`.txt`, `.md`, and `.pdf` via `pypdf`), **splits each into chunks** (default: ~800 characters with ~100 character overlap), and adds them to a **persistent Chroma collection**. Chroma embeds them automatically. Store the source filename with each chunk (needed for citations later).
- **Checkpoint:** AJ runs `python ingest.py` and sees something like `Indexed 42 chunks from 3 files`, and a `chroma/` folder appears.
- **Concept for AJ:** Documents are too big to hand the model whole, so we cut them into **chunks**. **Overlap** keeps sentences from being awkwardly split. Each chunk gets embedded and stored so we can search by meaning.

### Phase 4 — The RAG loop (retrieve + answer)
- **Goal:** the heart of the project.
- **Claude Code builds:** `rag.py` with `answer(question)` that: (1) searches Chroma for the top **k=4** most relevant chunks; (2) builds a prompt containing those chunks plus the question and clear instructions — *"Answer using ONLY the context below. If the answer isn't in the context, say you don't know. Cite the source filenames you used."*; (3) calls Gemini; (4) returns the answer **and** the list of source filenames. Plus a small `ask.py` so AJ can test from the command line.
- **Checkpoint:** AJ runs `python ask.py "a question answerable from the docs"` and gets a correct answer **with sources**. Then asks something **not** in the docs and gets an honest *"I don't know."*
- **Concept for AJ:** This is **RAG**. The model doesn't answer from memory — we *retrieve* the relevant text and *ground* the model in it. That's what prevents made-up answers ("hallucinations") and lets us cite sources.

### Phase 5 — The web app
- **Goal:** a clickable interface.
- **Claude Code builds:** `app.py` in Streamlit: a text box for the question, a button, the answer displayed clearly, and the sources shown in an expandable section. Read the API key in a way that works **both** locally (from `.env`) **and** on Streamlit Cloud (from `st.secrets`) — this prevents a deployment failure later.
- **Checkpoint:** AJ runs `streamlit run app.py`, a browser opens, he asks questions and sees answers + sources.
- **Concept for AJ:** Streamlit turns a Python script into a web app with almost no web code.

### Phase 6 — Evaluation
- **Goal:** prove it works, with a number.
- **Claude Code builds:** `eval_questions.json` with ~6–8 questions and their expected answers/sources; `eval.py` that runs each question through `answer()` and grades it (use an **LLM-as-judge**: ask Gemini to score whether the produced answer matches the expected one), then prints a score like `6/8 correct`.
- **Checkpoint:** AJ runs `python eval.py` and sees a score.
- **Concept for AJ:** Real AI engineers don't just *feel* that a system works — they **measure** it. "LLM-as-judge" means using a model to grade outputs at scale. This is a strong thing to talk about in interviews.

### Phase 7 — Deploy + README
- **Goal:** a public link and a professional writeup.
- **Claude Code builds:** a clean `README.md` (what it does, a diagram/screenshot spot, the stack, how it works, how to run it locally); confirms `requirements.txt` is complete; guides AJ to push to **GitHub** and deploy on **Streamlit Community Cloud**.
- **Deployment note (important):** the API key must be added in **Streamlit's Secrets settings** (in the app dashboard), **not** committed to GitHub. Claude Code should remind AJ of this exact step.
- **Checkpoint:** a live URL that works from any browser.
- **Concept for AJ:** Deploying means running your app on someone else's computer so anyone can use it. Secrets (like API keys) are configured on the platform, never in the code.

---

## Part 6 — If Claude Code asks you something, here are your answers

Most of these are already decided — this table is your safety net.

| If it asks… | Your answer |
|---|---|
| "Which LLM provider/model should I use?" | Google Gemini, model `gemini-2.5-flash`, via the `google-genai` SDK. |
| "Do you have an API key?" | Yes — it's in `.env` as `GEMINI_API_KEY` (set in Decision A). |
| "Which embedding model/provider?" | None separately — use ChromaDB's built-in default embeddings. |
| "Which vector database?" | ChromaDB, persistent, stored at `./chroma`. |
| "What documents should I index?" | The files in `./docs` (Decision B). If empty, create 2–3 sample text files. |
| "What chunk size / overlap?" | ~800 characters, ~100 overlap. Fine to tune later. |
| "How many chunks to retrieve?" | Top 4. |
| "Should I use LangChain / LlamaIndex?" | **No.** Write it directly with the plain SDKs. |
| "Should I add login / a database / Docker / async / tests?" | **No** — out of scope. Keep it minimal. |
| "The model name errors / isn't found." | Open aistudio.google.com to see currently-free model names and use one of those (any free `flash` model). |
| "Which Python version?" | 3.11 or newer. |

---

## Part 7 — Plain-English glossary

- **LLM** — the AI text model (here, Gemini). Sends text in, gets text out.
- **API key** — your password to use the model. Kept secret in `.env`.
- **Embedding** — text turned into numbers that capture meaning. Similar meanings → close numbers.
- **Vector database (Chroma)** — a database that stores embeddings and finds the closest ones fast.
- **Chunk** — a small piece of a document, sized so it fits nicely into a prompt.
- **Retrieval** — finding the chunks most relevant to a question.
- **RAG** — Retrieval-Augmented Generation: retrieve relevant chunks, then have the LLM answer using them.
- **Grounding** — forcing the model to answer from provided text instead of its own memory.
- **Hallucination** — when a model confidently makes something up. RAG + "say I don't know" reduces this.
- **Prompt** — the full text you send the model, including instructions + the retrieved chunks + the question.
- **LLM-as-judge** — using an LLM to automatically grade other LLM outputs.

---

## Part 8 — The payoff (for AJ's resume & interviews)

**Resume line:**
> Built and deployed a Retrieval-Augmented Generation (RAG) question-answering app in Python — implementing document chunking, vector embeddings, semantic search (ChromaDB), and grounded LLM answer generation with source citations and an anti-hallucination guard; added an LLM-as-judge evaluation suite and deployed it publicly on Streamlit Cloud.

**Things you'll be able to explain in an interview** (practice saying these out loud):
- Why RAG grounds the model in real documents instead of relying on its training memory.
- What an embedding is and why semantic search beats keyword search.
- The chunking trade-off (too big = imprecise retrieval; too small = lost context).
- How you *measured* quality with an evaluation set, not just vibes.
- One thing you'd improve next (e.g. re-ranking retrieved chunks, streaming responses, handling larger document sets).

---

*Stack verified against Google's current SDK and free tier as of mid-2026. If any model name stops working, check Google AI Studio for the current free models.*
