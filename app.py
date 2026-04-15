"""
🕵️ Mystery Detective Solver Agent
====================================
A LangGraph ReAct agent that analyzes murder mysteries,
reasons step-by-step, and reveals the culprit.

Stack: Streamlit + LangGraph + ChatGroq + FAISS + DuckDuckGo
"""

import os
import json
import re
import numpy as np
import streamlit as st
from typing import Optional
from dotenv import load_dotenv
load_dotenv()
# LangChain / LangGraph
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# FAISS + Embeddings
import faiss
from sentence_transformers import SentenceTransformer

# DuckDuckGo Search
from duckduckgo_search import DDGS

# PDF Export
from fpdf import FPDF

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🕵️ Mystery Detective Solver",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS  – noir / investigation aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=Crimson+Pro:wght@300;400;600&family=Courier+Prime:wght@400;700&display=swap');

/* ══════════════════════════════════════════
   ROOT PALETTE  –  parchment-noir
   Deep brown backgrounds, warm ivory text,
   amber gold accents, crimson highlights
   ══════════════════════════════════════════ */
:root {
    --bg:         #1c1410;   /* very dark walnut brown */
    --surface:    #231a13;   /* sidebar / elevated panels */
    --card:       #2b2018;   /* cards, message bubbles */
    --card-hover: #342618;
    --border:     #4a3520;   /* warm brown border */
    --border-lit: #7a5c30;   /* brighter border on focus */
    --accent:     #d4a843;   /* rich amber gold */
    --accent-dim: #a07830;   /* muted gold */
    --accent-glow:rgba(212,168,67,0.25);
    --red:        #c94040;   /* investigation red */
    --red-glow:   rgba(201,64,64,0.3);
    --text:       #f0e6d0;   /* warm ivory – HIGH CONTRAST */
    --text-sec:   #c8b89a;   /* secondary text */
    --muted:      #8a7055;   /* muted labels */
    --success:    #5aad7a;
    --ink:        #0e0a06;   /* near-black for button text */
}

/* ══ Reset & Base ══ */
*, *::before, *::after { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }

html, body, .stApp, .main, .block-container {
    background: var(--bg) !important;
    color: var(--text) !important;
}
.stApp {
    font-family: 'Crimson Pro', Georgia, serif;
    font-size: 16px;
    line-height: 1.7;
}

/* ── Subtle noise grain overlay ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: 0.03;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
    background-size: 200px 200px;
}

/* ══ TYPOGRAPHY ══ */
h1, h2, h3, h4 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: var(--accent) !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
}
h1 { font-size: 2.2rem !important; letter-spacing: 3px; text-shadow: 0 0 40px var(--accent-glow); }
h3 { font-size: 1.2rem !important; letter-spacing: 1px; color: var(--text-sec) !important; font-weight: 400 !important; }

p, li, span, div { color: var(--text) !important; }
label, .stMarkdown p { color: var(--text-sec) !important; font-size: 0.85rem !important; letter-spacing: 0.5px; }
strong { color: var(--accent) !important; font-weight: 700 !important; }
code { background: var(--card) !important; color: var(--accent) !important; padding: 0.1em 0.4em; border-radius: 3px; font-family: 'Courier Prime', monospace; }

/* ══ SIDEBAR ══ */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 2px solid var(--border) !important;
    padding-top: 1rem;
}
[data-testid="stSidebar"] > div { padding: 1rem 1.25rem !important; }
[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'Playfair Display', serif !important;
    font-size: 1rem !important;
    color: var(--accent) !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 0.4rem !important;
    margin-bottom: 0.75rem !important;
}
/* Sidebar section label */
[data-testid="stSidebar"] label {
    color: var(--accent-dim) !important;
    font-family: 'Courier Prime', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    display: block !important;
    margin-bottom: 0.25rem !important;
}

/* ══ INPUT FIELDS ══ */
.stTextArea textarea, .stTextInput input {
    background: #1e160e !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--text) !important;
    font-family: 'Crimson Pro', Georgia, serif !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    padding: 0.6rem 0.8rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    caret-color: var(--accent) !important;
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder {
    color: var(--muted) !important;
    font-style: italic !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-glow), inset 0 1px 3px rgba(0,0,0,0.4) !important;
    outline: none !important;
}

/* ══ BUTTONS ══ */
.stButton > button {
    background: linear-gradient(180deg, #2e1f0a 0%, #1e1208 100%) !important;
    border: 1.5px solid var(--accent) !important;
    border-radius: 3px !important;
    color: var(--accent) !important;
    font-family: 'Courier Prime', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4), inset 0 1px 0 rgba(212,168,67,0.1) !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: linear-gradient(180deg, var(--accent) 0%, var(--accent-dim) 100%) !important;
    color: var(--ink) !important;
    border-color: var(--accent) !important;
    box-shadow: 0 0 20px var(--accent-glow), 0 4px 12px rgba(0,0,0,0.5) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ══ CHAT MESSAGES ══ */
[data-testid="stChatMessage"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    margin-bottom: 0.85rem !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3) !important;
    transition: border-color 0.2s !important;
}
[data-testid="stChatMessage"]:hover { border-color: var(--border-lit) !important; }
[data-testid="stChatMessage"] p {
    color: var(--text) !important;
    font-family: 'Crimson Pro', Georgia, serif !important;
    font-size: 1rem !important;
    line-height: 1.7 !important;
    margin: 0 !important;
}
/* Avatar background */
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] { background: #3a2510 !important; }
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] { background: #2a1e0a !important; }

/* ══ EXPANDER ══ */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 5px !important;
    overflow: hidden !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stExpander"] summary,
.streamlit-expanderHeader {
    background: #221810 !important;
    color: var(--accent) !important;
    font-family: 'Courier Prime', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 0.7rem 1rem !important;
    border: none !important;
    text-transform: uppercase !important;
}
[data-testid="stExpander"] summary:hover { background: var(--card-hover) !important; }
[data-testid="stExpander"] > div > div {
    background: #1e1510 !important;
    border-top: 1px solid var(--border) !important;
    padding: 1rem !important;
}
[data-testid="stExpander"] p, [data-testid="stExpander"] li {
    color: var(--text-sec) !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
}

/* ══ METRICS ══ */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-top: 3px solid var(--accent) !important;
    border-radius: 5px !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-family: 'Courier Prime', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    text-shadow: 0 0 12px var(--accent-glow) !important;
}

/* ══ DIVIDERS ══ */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
    opacity: 1 !important;
}

/* ══ SPINNER ══ */
.stSpinner > div { border-top-color: var(--accent) !important; }
.stSpinner p { color: var(--text-sec) !important; font-family: 'Courier Prime', monospace !important; font-size: 0.8rem !important; letter-spacing: 1px !important; }

/* ══ CHAT INPUT BAR ══ */
[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border-top: 1px solid var(--border) !important;
}
[data-testid="stChatInput"] textarea {
    background: #201710 !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-family: 'Crimson Pro', Georgia, serif !important;
    font-size: 1rem !important;
    caret-color: var(--accent) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted) !important; font-style: italic !important; }
[data-testid="stChatInput"] textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px var(--accent-glow) !important; }
[data-testid="stChatInput"] button {
    background: var(--accent) !important;
    border-radius: 4px !important;
    color: var(--ink) !important;
}

/* ══ ALERTS / WARNINGS ══ */
[data-testid="stAlert"] {
    background: #2a1a10 !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--accent) !important;
    border-radius: 4px !important;
    color: var(--text-sec) !important;
}

/* ══ PASSWORD INPUT ══ */
[data-testid="stTextInput"] input[type="password"] {
    font-family: 'Courier Prime', monospace !important;
    font-size: 0.9rem !important;
    letter-spacing: 2px !important;
}

/* ══ SELECTBOX / DROPDOWN ══ */
[data-testid="stSelectbox"] > div { background: var(--card) !important; border-color: var(--border) !important; }

/* ══ SCROLLBAR ══ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-lit); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-dim); }

/* ══ RESULT CARD ══ */
.result-card {
    background: linear-gradient(135deg, #241508 0%, #1e180e 50%, #1a1a12 100%);
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 2rem 2.5rem;
    margin: 1.25rem 0;
    box-shadow: 0 0 40px var(--accent-glow), 0 4px 20px rgba(0,0,0,0.5);
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--accent), var(--red), var(--accent), transparent);
}
.result-card h2 { margin-top: 0 !important; font-size: 1rem !important; color: var(--muted) !important; letter-spacing: 3px; font-family: 'Courier Prime', monospace !important; font-weight: 400 !important; }
.culprit-name {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 900;
    color: var(--red) !important;
    text-shadow: 0 0 30px var(--red-glow);
    display: block;
    margin: 0.4rem 0 1rem;
    line-height: 1.1;
}
.badge {
    display: inline-block;
    background: var(--accent);
    color: var(--ink);
    font-family: 'Courier Prime', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 3px;
    padding: 0.25rem 0.75rem;
    border-radius: 2px;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    box-shadow: 0 0 10px var(--accent-glow);
}
.reasoning-text {
    font-family: 'Crimson Pro', Georgia, serif;
    color: var(--text) !important;
    line-height: 1.9;
    font-size: 1.05rem;
    border-left: 3px solid var(--accent);
    padding-left: 1.25rem;
    margin: 1rem 0;
    background: rgba(0,0,0,0.2);
    padding: 1rem 1rem 1rem 1.5rem;
    border-radius: 0 4px 4px 0;
}

/* ══ MAIN CONTENT AREA ══ */
.main .block-container {
    padding: 2rem 2.5rem !important;
    max-width: 960px !important;
}

/* ══ EMPTY STATE ══ */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    border: 1px dashed var(--border);
    border-radius: 8px;
    background: rgba(0,0,0,0.15);
    margin: 2rem 0;
}
.empty-state .icon { font-size: 4rem; display: block; margin-bottom: 1rem; filter: drop-shadow(0 0 12px var(--accent-glow)); }
.empty-state .title {
    font-family: 'Courier Prime', monospace !important;
    letter-spacing: 4px;
    font-size: 0.85rem;
    color: var(--muted) !important;
    text-transform: uppercase;
    display: block;
    margin-bottom: 0.5rem;
}
.empty-state .subtitle {
    font-family: 'Crimson Pro', serif !important;
    font-size: 1rem;
    color: var(--text-sec) !important;
    font-style: italic;
}

/* ══ PAGE HEADER ══ */
.page-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.page-header .subtitle {
    font-family: 'Courier Prime', monospace;
    color: var(--muted) !important;
    font-size: 0.72rem;
    letter-spacing: 5px;
    text-transform: uppercase;
    display: block;
    margin-top: -0.5rem;
}
.page-header .tagline {
    font-family: 'Crimson Pro', serif;
    color: var(--text-sec) !important;
    font-size: 1rem;
    font-style: italic;
    display: block;
    margin-top: 0.25rem;
}

/* Section headers in main area */
.section-label {
    font-family: 'Courier Prime', monospace !important;
    color: var(--muted) !important;
    font-size: 0.7rem !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem !important;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FAISS Vector Store (module-level singleton)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


class FAISSClueStore:
    """Simple in-memory FAISS store for clue retrieval."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.texts: list[str] = []

    def add(self, texts: list[str], model: SentenceTransformer):
        if not texts:
            return
        embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")
        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query: str, model: SentenceTransformer, k: int = 3) -> list[str]:
        if self.index.ntotal == 0:
            return []
        q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
        k = min(k, self.index.ntotal)
        _, indices = self.index.search(q_emb, k)
        return [self.texts[i] for i in indices[0] if i < len(self.texts)]

# Global instance to avoid Streamlit session_state thread-safety issues during LangGraph execution.
GLOBAL_CLUE_STORE = FAISSClueStore()

# ─────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],
        "case_loaded": False,
        "case_description": "",
        "clues": "",
        "suspects": "",
        "clue_store": GLOBAL_CLUE_STORE,
        "memory": MemorySaver(),
        "thread_id": "detective-session-001",
        "final_result": None,
        "agent": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# Force agent rebuild so updated prompt/token settings always take effect
st.session_state.agent = None

# ─────────────────────────────────────────────
# Groq LLM
# ─────────────────────────────────────────────
GROQ_MODELS = {
    "llama-3.1-8b-instant":    "LLaMA 3.1 8B Instant  (500k TPD — recommended)",
    "llama-3.3-70b-versatile": "LLaMA 3.3 70B Versatile (100k TPD — high quality)",
    "llama3-70b-8192":         "LLaMA 3 70B  (6k TPM)",
    "mixtral-8x7b-32768":      "Mixtral 8×7B 32k  (5k TPM)",
}

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    # Optional: allow user input (but no hardcoded key)
    if not api_key:
        api_key = st.session_state.get("groq_api_key")

    if not api_key:
        st.warning("Please enter your GROQ API Key")
        return None

    model = st.session_state.get("selected_model", "llama-3.1-8b-instant")

    return ChatGroq(
        model=model,
        api_key=api_key,
        temperature=0.3,
        max_tokens=8192,
    )


# ─────────────────────────────────────────────
# Tool definitions  (@tool decorator)
# ─────────────────────────────────────────────

@tool
def clue_analyzer(clues: str) -> str:
    """
    Analyze crime scene clues and extract meaningful investigative insights.
    Input: a string describing clues found at the scene or during investigation.
    Returns structured analysis with patterns, inconsistencies, and key findings.
    """
    if not clues.strip():
        return "No clues provided for analysis."
    return (
        f"[CLUE ANALYSIS]\n"
        f"Raw clues reviewed: {clues}\n\n"
        f"Key observations:\n"
        f"• Physical evidence should be cross-referenced with suspect access.\n"
        f"• Timing inconsistencies are critical — note any alibi gaps.\n"
        f"• Motive indicators: financial gain, jealousy, revenge, self-preservation.\n"
        f"• Look for contradictions between witness statements and physical evidence.\n"
        f"Proceed to suspect_profiler with these insights."
    )


@tool
def suspect_profiler(suspects: str) -> str:
    """
    Generate investigative profiles and possible motives for each suspect.
    Input: comma-separated or newline-separated list of suspects with optional descriptions.
    Returns profiles with motive hypotheses and risk rankings.
    """
    if not suspects.strip():
        return "No suspects provided to profile."
    names = [s.strip() for s in re.split(r"[,\n]+", suspects) if s.strip()]
    profiles = []
    for name in names:
        profiles.append(
            f"▸ {name}: Profile generated. "
            f"Potential motives include opportunity, personal conflict, and material gain. "
            f"Alibi and whereabouts require verification. Cross-reference with physical clues."
        )
    return "[SUSPECT PROFILES]\n" + "\n".join(profiles)


@tool
def memory_lookup(query: str) -> str:
    """
    Retrieve relevant past clues and case fragments from the FAISS vector database.
    Input: a search query describing what you're looking for.
    Returns the most semantically relevant stored clues.
    """
    model = load_embedding_model()
    store: FAISSClueStore = GLOBAL_CLUE_STORE
    results = store.search(query, model, k=3)
    if not results:
        return "No relevant clues found in memory. The case database may be empty."
    return "[MEMORY RETRIEVAL]\nRelevant clues found:\n" + "\n• ".join([""] + results)


@tool
def web_search(query: str) -> str:
    """
    Search DuckDuckGo for investigative context, forensic methods, or background info.
    Input: a concise search query.
    Returns top search snippets to enrich detective reasoning.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No external results found."
        formatted = "\n\n".join(
            f"[{r.get('title', 'Result')}]\n{r.get('body', '')}" for r in results
        )
        return f"[WEB SEARCH: {query}]\n\n{formatted}"
    except Exception as e:
        return f"Web search failed: {e}"


# ─────────────────────────────────────────────
# Run Agent (One-Shot with FAISS Injection)
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are Detective AI, a brilliant and methodical murder mystery solver.
Your reasoning style: sharp, logical, noir.

WORKFLOW:
1. Review the provided Case Database Clues (if any).
2. Synthesize ALL evidence and the user's latest input.
3. Name the culprit — ALWAYS.

CRITICAL RULES:
- You MUST always name a specific culprit. Never say "I conclude that:" and stop.
- Keep your reasoning concise — 3 to 5 sentences maximum before the JSON.
- ALWAYS end every response with the JSON block below, no exceptions.
- Do NOT truncate. Complete the JSON block in full before stopping.

FINAL ANSWER FORMAT — end EVERY response with this exact block:
```json
{
  "culprit": "<Full Name>",
  "reason": "<2-3 sentence explanation citing specific clues>",
  "confidence": "<0-100>%"
}
```

Be decisive. A detective who does not name a suspect has failed.
"""

def run_agent(user_input: str) -> str:
    llm = get_llm()
    if llm is None:
        return "⚠️ GROQ_API_KEY not set. Please enter your API key in the sidebar."

    # Grab FAISS context automatically instead of using a token-heavy LangGraph tool loop
    try:
        model = load_embedding_model()
        store = GLOBAL_CLUE_STORE
        docs = store.search(user_input, model, k=3)
        db_context = "\n".join([f"- {d}" for d in docs]) if docs else "No additional clues found."
    except Exception:
        db_context = "No additional clues found."

    history = [SystemMessage(content=SYSTEM_PROMPT)]
    
    # Grab the last 4 messages to save tokens but retain basic context
    recent_messages = st.session_state.messages[-4:] if len(st.session_state.messages) > 0 else []
    for msg in recent_messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            # Strip out JSON blob from assistant memory to save tokens
            clean = re.sub(r"```json.*?```", "", msg["content"], flags=re.DOTALL).strip()
            clean = re.sub(r'\{[^{}]*"culprit"[^{}]*\}', "", clean, flags=re.DOTALL).strip()
            history.append(AIMessage(content=clean))
            
    enhanced_input = f"{user_input}\n\n[CASE DATABASE CLUES]\n{db_context}"
    history.append(HumanMessage(content=enhanced_input))
    
    try:
        response = llm.invoke(history)
        return response.content
    except Exception as e:
        err = str(e)
        if "rate_limit_exceeded" in err or "429" in err:
            wait_match = re.search(r"Please try again in ([\d\w\.]+)", err)
            wait_time = wait_match.group(1) if wait_match else "a while"
            model_name = st.session_state.get("selected_model", "llama-3.1-8b-instant")
            return (
                f"🚦 **Rate limit reached** for `{model_name}`.\n\n"
                f"- ⏳ Please wait **{wait_time}** before retrying, or\n"
                f"- 🔄 Switch to **LLaMA 3.1 8B Instant** in the sidebar (500k tokens/day limit), or\n"
                f"- 🔑 Use a different Groq API key.\n\n"
                f"*Upgrade at https://console.groq.com/settings/billing for higher limits.*"
            )
        return f"❌ Agent error: {err}"


def extract_json_result(text: str) -> Optional[dict]:
    """Pull the structured JSON block out of the agent's response."""
    pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback: bare JSON object
    match2 = re.search(r'\{[^{}]*"culprit"[^{}]*\}', text, re.DOTALL)
    if match2:
        try:
            return json.loads(match2.group(0))
        except json.JSONDecodeError:
            pass
    return None


def load_case_into_faiss(clues: str):
    """Chunk clues by sentence and store in FAISS."""
    model = load_embedding_model()
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", clues) if len(s.strip()) > 10]
    if sentences:
        st.session_state.clue_store.add(sentences, model)


def generate_case_pdf() -> bytes:
    """Robustly generate a PDF report with aggressive string cleaning and layout safety."""
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    epw = pdf.w - 2 * pdf.l_margin # Effective page width for explicit control
    
    def safe_text(txt):
        if not txt:
            return ""
        txt = str(txt)
        # Normalize line endings
        txt = txt.replace("\r\n", "\n").replace("\r", "\n")
        # Direct character simplification for core Helvetica
        repl = {
            "\u2014": "-", "\u2013": "-", "\u201c": '"', "\u201d": '"',
            "\u2018": "'", "\u2019": "'", "\u2022": "*", "\u2026": "..."
        }
        for k, v in repl.items():
            txt = txt.replace(k, v)
        
        # Kill emojis / specialized unicode that FPDF standard fonts don't handle
        txt = txt.replace("🕵️", "Det.").replace("⚠️", "Warning:").replace("⚖️", "Verdict:")
        txt = txt.encode('latin-1', 'replace').decode('latin-1')
        
        # Aggressive Break: FPDF crashes if a single string (no spaces) is wider than column.
        # We break ANY unbroken token longer than 40 chars.
        parts = []
        for word in txt.split(" "):
            if len(word) > 40:
                # Inject a space every 40 chars
                word = re.sub(r"(\S{40})", r"\1 ", word)
            parts.append(word)
        return " ".join(parts).strip()

    # Title
    pdf.set_font("helvetica", style="B", size=18)
    pdf.cell(epw, 12, "MYSTERY CASE REPORT", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=10)
    pdf.cell(epw, 6, "CONFIDENTIAL - INVESTIGATIVE USE ONLY", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    
    # 1. Case Overview
    pdf.set_font("helvetica", style="B", size=14)
    pdf.cell(epw, 10, "1. Case Overview", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=11)
    pdf.multi_cell(epw, 6, safe_text(st.session_state.case_description))
    pdf.ln(5)

    if st.session_state.suspects.strip():
        pdf.set_font("helvetica", style="B", size=12)
        pdf.cell(epw, 8, "Suspects:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", size=11)
        pdf.multi_cell(epw, 6, safe_text(st.session_state.suspects))
        pdf.ln(3)

    if st.session_state.clues.strip():
        pdf.set_font("helvetica", style="B", size=12)
        pdf.cell(epw, 8, "Initial Clues:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", size=11)
        pdf.multi_cell(epw, 6, safe_text(st.session_state.clues))
        pdf.ln(8)

    # 2. History
    if st.session_state.messages:
        pdf.set_font("helvetica", style="B", size=14)
        pdf.cell(epw, 10, "2. Investigation Log", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        for msg in st.session_state.messages:
            role = "DET." if msg["role"] == "assistant" else "USER"
            content = msg["content"]
            # Scrub raw json
            content = re.sub(r"```json.*?```", "[Result compiled below]", content, flags=re.DOTALL)
            content = re.sub(r'\{[^{}]*"culprit"[^{}]*\}', "[Result compiled below]", content, flags=re.DOTALL)
            
            # Use specific style for Detective
            pdf.set_font("helvetica", style="B", size=10)
            pdf.cell(epw, 6, f"> {role}:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("helvetica", size=10)
            pdf.multi_cell(epw, 5, safe_text(content))
            pdf.ln(4)
        
    # 3. Final Findings
    if st.session_state.final_result:
        res = st.session_state.final_result
        pdf.ln(5)
        pdf.set_fill_color(240, 240, 240)
        pdf.set_font("helvetica", style="B", size=14)
        pdf.cell(epw, 12, "3. Final Verdict", fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
        
        pdf.set_font("helvetica", style="B", size=12)
        pdf.multi_cell(epw, 8, f"CULPRIT: {safe_text(res.get('culprit', 'Unknown'))}")
        pdf.set_font("helvetica", size=11)
        pdf.multi_cell(epw, 6, f"REASONING: {safe_text(res.get('reason', ''))}")
        pdf.multi_cell(epw, 6, f"CONFIDENCE: {safe_text(str(res.get('confidence', '')))}")

    return bytes(pdf.output())


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗝️ API Configuration", unsafe_allow_html=False)
    api_key_input = st.text_input(
        "GROQ API Key",
        type="password",
        value=st.session_state.get("groq_api_key", os.environ.get("GROQ_API_KEY", "")),
        placeholder="gsk_...",
        help="Get your key at console.groq.com",
    )
    if api_key_input:
        st.session_state.groq_api_key = api_key_input
        os.environ["GROQ_API_KEY"] = api_key_input
        st.session_state.agent = None  # Reset agent when key changes

    # Model selector
    prev_model = st.session_state.get("selected_model", "llama-3.1-8b-instant")
    selected_model = st.selectbox(
        "Model",
        options=list(GROQ_MODELS.keys()),
        index=list(GROQ_MODELS.keys()).index(prev_model) if prev_model in GROQ_MODELS else 0,
        format_func=lambda m: GROQ_MODELS[m],
        help="Switch models if you hit a rate limit.",
    )
    if selected_model != prev_model:
        st.session_state.selected_model = selected_model
        st.session_state.agent = None  # Rebuild agent with new model
    else:
        st.session_state.selected_model = selected_model

    st.markdown("---")
    st.markdown("## 📋 Case Files")

    case_description = st.text_area(
        "Case Description",
        value=st.session_state.case_description,
        height=120,
        placeholder="Describe the crime scene, victim, and circumstances...",
    )

    clues_input = st.text_area(
        "Evidence & Clues",
        value=st.session_state.clues,
        height=120,
        placeholder="List all clues, one per line or comma-separated...",
    )

    suspects_input = st.text_area(
        "Suspects",
        value=st.session_state.suspects,
        height=100,
        placeholder="Name each suspect, one per line...",
    )

    col1, col2 = st.columns(2)

    with col1:
        analyze_btn = st.button("🔍 Analyze Case", use_container_width=True)

    with col2:
        reveal_btn = st.button("🎭 Reveal Culprit", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📁 FAISS Memory")
    clue_count = st.session_state.clue_store.index.ntotal
    st.metric("Stored Clue Vectors", clue_count)

    if st.button("🗑️ Reset Case", use_container_width=True):
        for key in ["messages", "case_loaded", "case_description", "clues",
                    "suspects", "clue_store", "memory", "final_result", "agent"]:
            if key in st.session_state:
                del st.session_state[key]
        init_state()
        st.rerun()

    if st.session_state.messages:
        st.markdown("---")
        try:
            pdf_bytes = generate_case_pdf()
            st.download_button(
                label="📄 Export Case to PDF",
                data=pdf_bytes,
                file_name="Mystery_Case_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
                help="Download the entire case file and investigation log as a PDF."
            )
        except Exception as e:
            st.error(f"Could not generate PDF: {e}")

    st.markdown("---")
    active_model = st.session_state.get("selected_model", "llama-3.1-8b-instant")
    st.markdown(
        f"<div style='font-family:Courier Prime,monospace;color:var(--muted);font-size:0.72rem;"
        f"letter-spacing:1px;line-height:2;border-top:1px solid var(--border);padding-top:0.75rem;'>"
        f"📡 Model: {active_model}<br>"
        f"🧠 Engine: LangGraph ReAct<br>"
        f"💾 Memory: MemorySaver + FAISS<br>"
        f"🔎 Search: DuckDuckGo"
        f"</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1 style="margin-bottom:0.1rem;">🕵️ Mystery Detective Solver</h1>
    <span class="subtitle">— AI Powered Investigation System —</span>
    <span class="tagline">LangGraph · ChatGroq LLaMA 3.3 · FAISS RAG · ReAct Reasoning</span>
</div>
""", unsafe_allow_html=True)

# ── Handle ANALYZE CASE button ──
if analyze_btn:
    if not case_description.strip() and not clues_input.strip():
        st.error("Please provide at least a case description or some clues.")
    else:
        # Persist sidebar values
        st.session_state.case_description = case_description
        st.session_state.clues = clues_input
        st.session_state.suspects = suspects_input

        # Load clues into FAISS
        if clues_input.strip():
            with st.spinner("Indexing evidence into FAISS vector store…"):
                load_case_into_faiss(clues_input)

        # Build analysis prompt
        prompt = (
            f"NEW CASE OPENED.\n\n"
            f"CASE DESCRIPTION:\n{case_description}\n\n"
            f"EVIDENCE & CLUES:\n{clues_input}\n\n"
            f"SUSPECTS:\n{suspects_input}\n\n"
            f"Begin your investigation. Use your tools. "
            f"Ask me any clarifying questions you need."
        )

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.case_loaded = True

        with st.spinner("🔍 Detective AI is examining the evidence…"):
            response = run_agent(prompt)

        st.session_state.messages.append({"role": "assistant", "content": response})

        # Check if result already revealed
        result = extract_json_result(response)
        if result:
            st.session_state.final_result = result

        st.rerun()

# ── Handle REVEAL CULPRIT button ──
if reveal_btn:
    if not st.session_state.case_loaded:
        st.warning("Please analyze the case first before revealing the culprit.")
    else:
        reveal_prompt = (
            "Based on all the evidence analyzed, suspect profiles, and clues retrieved, "
            "it is time to REVEAL THE CULPRIT. "
            "Provide your final deduction with full reasoning. "
            "End with the structured JSON result."
        )
        st.session_state.messages.append({"role": "user", "content": reveal_prompt})

        with st.spinner("🎭 The detective is drawing final conclusions…"):
            response = run_agent(reveal_prompt)

        st.session_state.messages.append({"role": "assistant", "content": response})

        result = extract_json_result(response)
        if result:
            st.session_state.final_result = result

        st.rerun()

# ─────────────────────────────────────────────
# Final Result Banner
# ─────────────────────────────────────────────
if st.session_state.final_result:
    r = st.session_state.final_result
    culprit = r.get("culprit", "Unknown")
    reason = r.get("reason", "")
    confidence_raw = str(r.get("confidence", "0%")).replace("%", "").strip()
    try:
        confidence_val = int(float(confidence_raw))
    except ValueError:
        confidence_val = 0

    st.markdown(
        f"""
        <div class="result-card">
            <span class="badge">Case Solved</span>
            <h2 style="margin-bottom:0.25rem;">The Culprit Is…</h2>
            <span class="culprit-name">{culprit}</span>
            <div class="reasoning-text">{reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("🎯 Culprit", culprit)
    m2.metric("📊 Confidence", f"{confidence_val}%")
    m3.metric("🧠 Method", "ReAct + RAG")

    st.markdown("---")

# ─────────────────────────────────────────────
# Chat History
# ─────────────────────────────────────────────
if st.session_state.messages:
    st.markdown('<p class="section-label">📋 Investigation Log</p>', unsafe_allow_html=True)
    for i, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            with st.chat_message("user", avatar="🧑‍💼"):
                # Collapse long system prompts
                if len(content) > 400:
                    with st.expander("📋 Case Brief Submitted"):
                        st.markdown(content)
                else:
                    st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🕵️"):
                # Strip fenced ```json``` blocks AND bare {...} JSON objects from display text
                clean = re.sub(r"```json.*?```", "", content, flags=re.DOTALL)
                clean = re.sub(r"\{[^{}]*\"culprit\"[^{}]*\}", "", clean, flags=re.DOTALL)
                clean = clean.strip()

                # Try to extract structured result from this specific message
                msg_result = extract_json_result(content)

                if clean:
                    with st.expander(f"🔎 Detective's Reasoning (Step {i // 2 + 1})", expanded=(i == len(st.session_state.messages) - 1)):
                        st.markdown(clean)

                # Render verdict as styled bullet points if this message has a JSON result
                if msg_result:
                    culprit = msg_result.get("culprit", "Unknown")
                    reason = msg_result.get("reason", "")
                    confidence = str(msg_result.get("confidence", "?"))
                    st.markdown(
                        f"""
                        <div style="background:linear-gradient(135deg,#241508,#1e180e);
                                    border:1px solid var(--accent);border-radius:6px;
                                    padding:1rem 1.25rem;margin-top:0.75rem;
                                    box-shadow:0 0 20px rgba(212,168,67,0.15);">
                            <div style="font-family:'Courier Prime',monospace;font-size:0.65rem;
                                        letter-spacing:3px;color:var(--accent);text-transform:uppercase;
                                        margin-bottom:0.6rem;">⚖️ Verdict</div>
                            <ul style="margin:0;padding-left:1.25rem;list-style:none;">
                                <li style="margin-bottom:0.4rem;">
                                    <span style="color:var(--accent);font-weight:700;">🎯 Culprit:</span>
                                    <span style="color:#c94040;font-family:'Playfair Display',serif;
                                                 font-size:1.1rem;font-weight:700;margin-left:0.4rem;">{culprit}</span>
                                </li>
                                <li style="margin-bottom:0.4rem;">
                                    <span style="color:var(--accent);font-weight:700;">📝 Reason:</span>
                                    <span style="color:var(--text-sec);margin-left:0.4rem;font-style:italic;">{reason}</span>
                                </li>
                                <li>
                                    <span style="color:var(--accent);font-weight:700;">📊 Confidence:</span>
                                    <span style="color:var(--text);margin-left:0.4rem;">{confidence}</span>
                                </li>
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ─────────────────────────────────────────────
# Chat Input  (free-form follow-up questions)
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-label">💬 Interrogation Room</p>', unsafe_allow_html=True)

user_query = st.chat_input(
    "Ask the detective a question, provide more info, or request re-analysis…"
)

if user_query:
    if not st.session_state.case_loaded:
        st.warning("Load a case first using the sidebar before chatting.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.spinner("🕵️ Detective is thinking…"):
            response = run_agent(user_query)
        st.session_state.messages.append({"role": "assistant", "content": response})

        result = extract_json_result(response)
        if result:
            st.session_state.final_result = result

        st.rerun()

# ─────────────────────────────────────────────
# Empty state
# ─────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <span class="icon">🔍</span>
        <span class="title">No Active Investigation</span>
        <span class="subtitle">
            Fill in the Case Files in the sidebar and click <strong>Analyze Case</strong> to open a new investigation.
        </span>
    </div>
    """, unsafe_allow_html=True)
