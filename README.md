# 🕵️ Mystery Detective Solver Agent

An interactive, AI-powered forensic investigation system built with **LangChain**, **Groq (LLaMA 3.3)**, **FAISS RAG**, and **Streamlit**. Designed with a premium noir aesthetic, this agent analyzes clues, profiles suspects, and reveals the culprit with sharp logical reasoning.

---

## 🏗️ Architecture

The system uses a **Hybrid RAG + One-Shot Reasoning** flow to maximize token efficiency while maintaining deep investigative context.

```
Streamlit UI (Noir Aesthetic & PDF Export)
    ↓
One-Shot Reasoning Engine (Direct LLM Invoke)
    ↓
FAISS Vector Store (Sentence-Transformers)
    ↓
Context Injection (Clues + Chat History + RAG Results)
    ↓
ChatGroq (llama-3-3-70b / llama-3-1-8b)
```

---

## ✨ Features

- **🔍 Smart Case Analysis**: Automatically indexes clues into a FAISS vector store for semantic retrieval.
- **🎭 Detective AI**: Methodical reasoning in a sharp, noir-inspired style.
- **💬 Interrogation Room**: Interactive chat to ask follow-up questions or provide new evidence.
- **⚖️ Structured Verdicts**: Every analysis ends with a JSON-backed verdict including the culprit, reasoning, and confidence score.
- **📄 PDF Case Reports**: Export the entire investigation log and final verdict to a professional PDF report.
- **🕶️ Premium UI**: Custom dark-walnut noir theme with glassmorphism and vintage typography.

---

## ⚡ Quick Start

### 1. Project Setup

```bash
# Clone or create the directory
mkdir mystery_detective && cd mystery_detective

# Install dependencies
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` downloads the `all-MiniLM-L6-v2` model (~80 MB) on first run, which is then cached.

### 2. Configure API Key

**Option A — Environment Variable:**
```bash
export GROQ_API_KEY="gsk_your_key_here"  # Mac/Linux
set GROQ_API_KEY=gsk_your_key_here      # Windows CMD
$env:GROQ_API_KEY="gsk_your_key_here"   # PowerShell
```

**Option B — Sidebar Input:**
Enter your key directly into the app's sidebar when it loads. Get a free key at [console.groq.com](https://console.groq.com).

### 3. Run the App

```bash
streamlit run app.py
```

---

## 🎮 How to Use

1.  **Fill in Case Files** (Sidebar):
    - **Case Description**: The "What, When, Where" of the crime.
    - **Evidence & Clues**: List findings (monogrammed handkerchiefs, poison types, etc.).
    - **Suspects**: Name the potential culprits.
2.  Click **🔍 Analyze Case**: The detective indexes the evidence and performs an initial review.
3.  **Interrogate**: Use the chat box to ask for specific alibis or deeper analysis of certain clues.
4.  Click **🎭 Reveal Culprit**: Get the final deduction and the official verdict.
5.  **Export**: Download the **PDF Report** for the official case file.

---

## 🧠 Tech Stack

| Component | Library | Purpose |
| :--- | :--- | :--- |
| **UI** | Streamlit | Responsive dashboard & Noir styling |
| **Logic** | LangChain | Orchestration & Tooling |
| **LLM** | Groq (LLaMA 3.3 70B) | High-speed, high-quality reasoning |
| **Vector DB** | FAISS | Semantic clue retrieval (RAG) |
| **Embeddings** | Sentence-Transformers | Clue indexing (all-MiniLM-L6-v2) |
| **Export** | FPDF | Professional PDF generation |
| **Search** | DuckDuckGo | External investigative context |

---

## 🔧 Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **Rate Limit Exceeded** | Switch to `llama-3.1-8b-instant` in the sidebar or wait for the cooldown. |
| **FAISS/Numpy Errors** | Ensure you have `faiss-cpu` installed: `pip install faiss-cpu`. |
| **PDF Format Issues** | The exporter cleans special characters automatically; stick to standard ASCII where possible. |
| **Missing API Key** | Ensure `GROQ_API_KEY` is in your `.env` or set in the sidebar. |

---

## 📦 Project Structure

```text
Mystery-Detective-Solver-Agent/
├── app.py              # Main application logic & UI
├── requirements.txt    # Python dependencies
├── README.md           # Documentation
└── .env               # (Optional) API key storage
```
