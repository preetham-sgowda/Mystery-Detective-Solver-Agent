# 🕵️ Mystery Detective Solver Agent

An interactive AI-powered murder mystery solver built with LangGraph ReAct agent, 
ChatGroq (LLaMA 3.3 70B), FAISS RAG, and Streamlit.

---

## 🏗️ Architecture

```
Streamlit UI
    ↓
LangGraph ReAct Agent (create_react_agent)
    ↓
@tool functions: clue_analyzer | suspect_profiler | memory_lookup | web_search
    ↓
ChatGroq (llama-3.3-70b-versatile) + MemorySaver
    ↓
FAISS RAG (sentence-transformers all-MiniLM-L6-v2)
```

---

## ⚡ Quick Start

### 1. Clone / copy the project

```bash
mkdir mystery_detective && cd mystery_detective
# place app.py and requirements.txt here
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~80 MB)
> on first run. This is cached automatically for subsequent runs.

### 4. Set your GROQ API key

**Option A — Environment variable (recommended):**

```bash
# Mac / Linux
export GROQ_API_KEY="gsk_your_key_here"

# Windows CMD
set GROQ_API_KEY=gsk_your_key_here

# Windows PowerShell
$env:GROQ_API_KEY="gsk_your_key_here"
```

**Option B — Enter it directly in the sidebar** when the app loads.

Get a free key at: https://console.groq.com

### 5. Run the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🎮 How to Use

1. **Fill in Case Files** (left sidebar):
   - *Case Description* — describe the crime scene and circumstances
   - *Evidence & Clues* — list all clues found
   - *Suspects* — name each suspect (one per line)

2. Click **🔍 Analyze Case** — the detective will:
   - Index clues into FAISS
   - Run ReAct reasoning (tool calls visible in logs)
   - Ask clarifying questions

3. Chat freely in the **Interrogation Room** to provide more info

4. Click **🎭 Reveal Culprit** to get the final verdict:
   - Culprit name
   - Full reasoning
   - Confidence percentage

---

## 🧪 Sample Mystery to Try

**Case Description:**
Taha was found dead in his locked study at Blackwood Manor. 
The time of death is estimated between 9 PM and 11 PM. 
A half-empty glass of whiskey sat on his desk. No signs of forced entry.

**Clues:**
- A monogrammed handkerchief with the initials "N.U." was found under the desk
- The poison used was cyanide, tasteless and odorless when dissolved in alcohol
- The manor's back door was unlocked despite the butler claiming to have locked it at 8 PM
- A recent will discovered shows Taha changed his beneficiary last week
- The kitchen staff reported hearing an argument between Taha and his niece Naushin Usman at 8:30 PM
- Footprints in the garden match a woman's shoe size 6

**Suspects:**
- Naushin Usman (niece, recently cut from the will)
- Manohar the butler (employed for 20 years, recently passed over for a raise)
- Dr. Saniya (family physician, secret affair with the lord)
- Jameel (estranged son, returned from abroad last week)

---

## 📦 Project Structure

```
mystery_detective/
├── app.py              # All application code (single file)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🔧 Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: faiss` | Run `pip install faiss-cpu` |
| `AuthenticationError` from Groq | Double-check your GROQ_API_KEY |
| Slow first load | sentence-transformers model is downloading (~80 MB) |
| `duckduckgo_search` rate limit | Wait 30 seconds and retry; DDG has free tier limits |
| Streamlit version errors | Upgrade: `pip install --upgrade streamlit` |

---

## 🧠 Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| UI | Streamlit | Chat interface & controls |
| Agent | LangGraph `create_react_agent` | ReAct reasoning loop |
| LLM | ChatGroq + llama-3.3-70b | Inference |
| Memory | `MemorySaver` | Conversation continuity |
| RAG | FAISS + sentence-transformers | Clue retrieval |
| Search | duckduckgo-search | External context |
