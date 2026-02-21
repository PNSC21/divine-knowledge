import pandas as pd
import numpy as np
import faiss
import re
from flask import request, session
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# LOAD DATA + EMBEDDINGS (ONCE)

df = pd.read_csv("scriptures/Bhagwad_Gita.csv").dropna(subset=['EngMeaning']).reset_index(drop=True)

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

verse_texts = df['EngMeaning'].tolist()

index = faiss.read_index("models/gita_faiss.index")
embeddings = np.load("models/gita_embeddings.npy")

# LOAD TINYLLAMA (ONCE)

model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading TinyLlama...")

tokenizer = AutoTokenizer.from_pretrained(model_path)

llm = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float32,
    low_cpu_mem_usage=True
)

llm.eval()
llm.to("cpu")

print("TinyLlama Loaded Successfully.")

# HELPER FUNCTIONS

def clean_text(text):
    # Removes verse numbering patterns from text and returns a cleaned string
    return re.sub(r"[0-9]+\.[0-9]+", "", text).strip()

def cosine_similarity(a, b):
    # Computes cosine similarity between two normalized embedding vectors.
    # Assumes inputs are already L2-normalized.
    return np.dot(a, b.T)[0][0]

def summarize_topic(text):
    """ Uses the LLM to generate a short summary representing
        the core discussion topic for session continuity tracking. """

    prompt = f"""
    Summarize the core discussion topic in max 5 words:
    {text}
    """
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = llm.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False
        )

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result.strip().split("\n")[-1]

def is_context_dependent(query, last_topic):
    """ Determines whether the current user query semantically depends
        on the previous topic using LLM-based binary classification. """

    prompt = f"""
    Previous topic: {last_topic}
    User message: "{query}"
    Does this message depend on the previous topic to make sense?
    Answer YES or NO.
    """
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = llm.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False
        )

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return "YES" in result.upper()

def is_follow_up(query, session_history, threshold=0.75):
    """ Determines whether the current user query semantically depends
        on the previous topic using LLM-based binary classification. """

    if not session_history:
        return False

    last_query = session_history[-1]['query']

    emb = embed_model.encode([query, last_query], normalize_embeddings=True)
    similarity = np.dot(emb[0], emb[1])

    return similarity >= threshold

def generate_response(user_query, session_history=None, verse=None):

    """ Generates a Krishna-persona response using TinyLlama with optional
        verse grounding and limited conversation history for contextual coherence. """

    system_msg = (
    "You are Bhagavan Shri Krishna, speaking directly to the seeker. "
    "Respond as a supreme guru: calm, compassionate, deeply intelligent, and clear. "

    "If a verse meaning is provided, explain its essence without repeating or fabricating verse numbers. "
    "If no verse is provided, answer from eternal wisdom directly. "

    "Do not hallucinate references. "
    "Speak in one concise but profound paragraph."
    )

    messages = [{"role": "system", "content": system_msg}]

    if session_history:
        for chat in session_history[-2:]:
            messages.append({"role": "user", "content": chat["query"]})
            messages.append({"role": "assistant", "content": chat["explanation"]})

    if verse:
        user_content = (
            f"The seeker asks: {user_query}\n\n"
            f"Related teaching: {verse}\n\n"
            "Explain the deeper truth."
        )
    else:
        user_content = (
            f"The seeker asks: {user_query}\n\n"
            "Guide them with eternal wisdom."
        )

    messages.append({"role": "user", "content": user_content})

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        output = llm.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = decoded.split("<|assistant|>")[-1].strip()

    if "." in answer:
        answer = answer[:answer.rfind(".") + 1]

    return answer

# MAIN CHAT FUNCTION

def geetachat():

    """ Main chat controller handling session state, follow-up detection,
        verse retrieval via FAISS semantic search, LLM response generation,
        topic summarization, and conversation persistence. """

    if "chats" not in session:
        session["chats"] = []

    if request.method == "POST":

        query = request.form["query"]

        follow_up = is_follow_up(query, session["chats"])

        if follow_up:
            shloka = None
            verse = None
        else:
            q_emb = embed_model.encode([query], normalize_embeddings=True)
            D, I = index.search(q_emb, k=3)

            best_index = I[0][0]
            best_score = D[0][0]

            if best_score > 0.70:
                result = df.iloc[best_index]
                shloka = result["Shloka"]
                verse = clean_text(result["EngMeaning"])
            else:
                shloka = None
                verse = None

        explanation = generate_response(
            query,
            session_history=session["chats"],
            verse=verse
        )

        topic_summary = summarize_topic(query + " " + explanation)

        session["chats"].append({
            "query": query,
            "topic_summary": topic_summary,
            "shloka": shloka,
            "verse": verse,
            "explanation": explanation
        })

        session.modified = True

    return session