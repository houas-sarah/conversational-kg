# Deploy

Three free options. The first is recommended — no credit card required.

---

## Option 1 · Hugging Face Spaces (recommended)

**Cost:** free · **Sleeps after idle:** no · **Credit card:** not required · **WebSocket:** yes · **Always-on:** yes

### one-time setup

1. **Sign up** at <https://huggingface.co/join> (email or GitHub OAuth, no card).
2. **Create a new Space:**
   - Click your avatar → *New Space*
   - **Space name:** `the-commonplace` (or anything)
   - **License:** MIT
   - **Space SDK:** select **Docker**
   - **Docker template:** *Blank*
   - **Visibility:** Public (free) or Private
   - Click *Create Space*.
3. **Add your Groq key as a secret:**
   - In the new Space, go to *Settings* (top right) → *Variables and secrets*
   - Click *New secret*
   - Name: `GROQ_API_KEY`, Value: your key from <https://console.groq.com>
4. **Add the HF frontmatter to README.md.** Open `README.md` and add this *at the very top* of the file (before the existing content):

   ```yaml
   ---
   title: the commonplace
   emoji: 🧠
   colorFrom: indigo
   colorTo: purple
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```

   This repo already ships an `hf-deploy` branch with that header added, so it
   stays separate from your clean GitHub `main`.

### push the code

You can either **import from GitHub** (easier) or **push directly to the Space's git remote**.

**A. Import from GitHub (one-way mirror):**
- In the Space's *Settings* → *Repository*, click *Sync from GitHub repository* and paste your repo URL.

**B. Push directly to the Space:**

```bash
# Add the HF Space as a second remote
git remote add hf https://huggingface.co/spaces/<your-username>/the-commonplace

# Push (use a HF access token from https://huggingface.co/settings/tokens as the password)
git push hf main
```

The Space will build the Docker image automatically (~3–5 min the first time). Subsequent pushes only rebuild what changed.

Your app will be live at `https://huggingface.co/spaces/<your-username>/the-commonplace`.

---

## Option 2 · Render

**Cost:** free tier · **Sleeps after idle:** yes (15 min, ~30 s wake-up) · **Credit card:** not required · **WebSocket:** yes

1. Sign up at <https://render.com> with GitHub.
2. *New +* → *Web Service* → connect your GitHub repo.
3. Configure:
   - **Environment:** Docker
   - **Region:** closest to you
   - **Instance type:** Free
   - **Environment variables:** add `GROQ_API_KEY`
4. Click *Create Web Service*.

The Dockerfile already in this repo will be used as-is.

---

## Option 3 · Local + Cloudflare Tunnel (free, no signup)

Run on your laptop and expose it publicly via a temporary tunnel — useful for showing the running app to your supervisor without deploying:

```bash
# in one terminal:
.\run.ps1

# in another terminal (after installing cloudflared from https://github.com/cloudflare/cloudflared):
cloudflared tunnel --url http://localhost:8000
```

Cloudflare prints a public URL like `https://random-words-123.trycloudflare.com` that anyone can open. Only works while your laptop is on.

---

## environment variables

| name | required | default | what it does |
|---|---|---|---|
| `GROQ_API_KEY` | optional | empty | Enables hybrid LLM extraction. Without it, the system falls back to spaCy + rules. Get a free key at <https://console.groq.com>. |
| `GROQ_MODEL` | optional | `llama-3.1-8b-instant` | Any Groq-hosted model name. |
| `PORT` | optional | `7860` on Docker, `8000` locally | Port the FastAPI app binds to. |

---

## quick local docker test

Before deploying, confirm the container runs on your machine:

```bash
docker build -t the-commonplace .
docker run --rm -p 8000:7860 -e GROQ_API_KEY=$env:GROQ_API_KEY the-commonplace
```

(use `$GROQ_API_KEY` instead of `$env:GROQ_API_KEY` on macOS/linux)

Open <http://localhost:8000>. If this works, deployment will work.

---

## persistence note

The system writes one knowledge-graph file per visitor session under
`data/sessions/` inside the container.

- **On Hugging Face Spaces (free tier):** the filesystem is *ephemeral* — files persist across short restarts but are wiped on a rebuild. For a live demo that's fine (each visitor already gets a fresh, isolated graph). For permanent storage, point `data/` at a real volume (HF Persistent Storage is a paid add-on).
- **On Render:** also ephemeral on the free tier. Same caveat.
- **For real persistence:** swap the JSON backend for a hosted database (SQLite on a volume, Neo4j Aura free tier, Supabase, etc.) — the `KnowledgeGraph` class in `backend/graph.py` has a clean interface for this.
