from flask import Flask, render_template, request, session, redirect, url_for
from GitaChat import *
import pandas as pd
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

df = pd.read_csv("scriptures/Bhagwad_Gita.csv")

def clean_meaning(text):
    if pd.isna(text):
        return ""

    text = str(text).strip()

    text = re.sub(r'^.*?\d+\.\d+.*?\s*', '', text)
    text = re.sub(r'^\.\s*', '', text)

    return text.strip()

def format_word_meaning(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove starting verse number like 10.28
    text = re.sub(r'^\d+\.\d+\s*', '', text)

    # Separate commentary if present
    commentary = ""
    if "Commentary" in text:
        main_part, commentary = text.split("Commentary", 1)
    else:
        main_part = text

    # Split using '?' because that is real separator
    parts = main_part.split("?")

    cleaned_parts = []

    for part in parts:
        part = part.strip()
        if part:
            cleaned_parts.append(part)

    formatted = "<br>".join(cleaned_parts)

    if commentary:
        formatted += "<br><br><strong>Explanation:</strong><br><br>" + commentary.strip()

    return formatted

# Homepage
@app.route("/")
def home():
    return render_template("home.html")

# Bhagavad Gita main page
@app.route("/gita")
def gita_home():
    chapters = sorted(df["Chapter"].unique())
    return render_template("gita_home.html", chapters=chapters)

# Chapter page
@app.route("/chapter/<int:chapter_id>")
def chapter(chapter_id):
    chapter_data = df[df["Chapter"] == chapter_id]
    verses = chapter_data["Verse"].tolist()
    return render_template("chapter.html",
                           chapter_id=chapter_id,
                           verses=verses)

# Verse page
@app.route("/chapter/<int:chapter_id>/verse/<int:verse_id>")
def verse(chapter_id, verse_id):
    verse_data = df[
        (df["Chapter"] == chapter_id) &
        (df["Verse"] == verse_id)
    ].iloc[0]

    verse_data["HinMeaning"] = clean_meaning(verse_data["HinMeaning"])
    verse_data["EngMeaning"] = clean_meaning(verse_data["EngMeaning"])
    verse_data["WordMeaning"] = format_word_meaning(verse_data["WordMeaning"])

    return render_template("verse.html", verse=verse_data)

@app.route("/gitagpt", methods=["GET", "POST"])
def gitagpt():
    geetachat()
    return render_template('BhagwadGita.html', chats=session.get('chats', []))

@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('gitagpt'))

if __name__ == "__main__":
    app.run(debug=False, port=5000)
