import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

# ── State shared across all agents ──────────────────────────────────────────
class DreamState(TypedDict):
    dream_text: str
    emotion_result: str
    symbol_result: str
    pattern_result: str
    final_report: str

# ── LLM factory ─────────────────────────────────────────────────────────────
def get_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.7,
    )

# ── Agent 1: Emotion Detector ────────────────────────────────────────────────
def emotion_detector(state: DreamState, llm) -> DreamState:
    prompt = f"""You are an Emotion Detector agent. Analyze the dream below and identify:
1. Primary emotion (e.g. fear, joy, anxiety, wonder, sadness)
2. Emotional intensity (Low / Medium / High)
3. One sentence about what the emotion suggests.

Dream: {state['dream_text']}

Respond in this exact format:
Primary Emotion: <emotion>
Intensity: <level>
Insight: <one sentence>"""

    response = llm.invoke(prompt)
    return {**state, "emotion_result": response.content.strip()}

# ── Agent 2: Symbol Decoder ──────────────────────────────────────────────────
def symbol_decoder(state: DreamState, llm) -> DreamState:
    prompt = f"""You are a Dream Symbol Decoder agent. Identify 2-3 key dream symbols from the text below.
For each symbol, give its classic psychological meaning in one sentence.

Dream: {state['dream_text']}

Respond in this exact format:
Symbol 1: <symbol> — <meaning>
Symbol 2: <symbol> — <meaning>
Symbol 3: <symbol> — <meaning> (if present, else skip)"""

    response = llm.invoke(prompt)
    return {**state, "symbol_result": response.content.strip()}

# ── Agent 3: Pattern Tracker ─────────────────────────────────────────────────
def pattern_tracker(state: DreamState, llm) -> DreamState:
    prompt = f"""You are a Dream Pattern Tracker agent. Based on this single dream, identify:
1. The dominant theme (e.g. conflict, escape, transformation, pursuit)
2. A likely life area it reflects (e.g. work stress, relationships, self-doubt)
3. A short actionable suggestion for the dreamer.

Dream: {state['dream_text']}

Respond in this exact format:
Theme: <theme>
Life Area: <area>
Suggestion: <one actionable sentence>"""

    response = llm.invoke(prompt)
    return {**state, "pattern_result": response.content.strip()}

# ── Agent 4: Insight Narrator ────────────────────────────────────────────────
def insight_narrator(state: DreamState, llm) -> DreamState:
    prompt = f"""You are the Insight Narrator. Synthesize the analysis below into a warm, 
encouraging 3-paragraph personal dream reflection for the user.

Dream: {state['dream_text']}

Emotion Analysis:
{state['emotion_result']}

Symbol Analysis:
{state['symbol_result']}

Pattern Analysis:
{state['pattern_result']}

Write a 3-paragraph reflection:
- Paragraph 1: What the dream is saying emotionally
- Paragraph 2: What the symbols and patterns reveal  
- Paragraph 3: An encouraging closing message with a journaling prompt"""

    response = llm.invoke(prompt)
    return {**state, "final_report": response.content.strip()}

# ── Build the LangGraph ──────────────────────────────────────────────────────
def build_graph(api_key: str):
    llm = get_llm(api_key)

    def run_emotion(state):   return emotion_detector(state, llm)
    def run_symbols(state):   return symbol_decoder(state, llm)
    def run_patterns(state):  return pattern_tracker(state, llm)
    def run_narrator(state):  return insight_narrator(state, llm)

    graph = StateGraph(DreamState)

    graph.add_node("emotion_detector",  run_emotion)
    graph.add_node("symbol_decoder",    run_symbols)
    graph.add_node("pattern_tracker",   run_patterns)
    graph.add_node("insight_narrator",  run_narrator)

    # emotion → symbol → pattern → narrator (sequential pipeline)
    graph.set_entry_point("emotion_detector")
    graph.add_edge("emotion_detector", "symbol_decoder")
    graph.add_edge("symbol_decoder",   "pattern_tracker")
    graph.add_edge("pattern_tracker",  "insight_narrator")
    graph.add_edge("insight_narrator", END)

    return graph.compile()

# ── Public function called by Flask ─────────────────────────────────────────
def analyze_dream(dream_text: str, api_key: str) -> dict:
    graph = build_graph(api_key)
    initial_state: DreamState = {
        "dream_text":     dream_text,
        "emotion_result": "",
        "symbol_result":  "",
        "pattern_result": "",
        "final_report":   "",
    }
    result = graph.invoke(initial_state)
    return {
        "emotion":  result["emotion_result"],
        "symbols":  result["symbol_result"],
        "patterns": result["pattern_result"],
        "report":   result["final_report"],
    }