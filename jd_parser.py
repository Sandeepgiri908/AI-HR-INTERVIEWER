import json
import os
import tempfile

import pdfplumber
from docx import Document
from PIL import Image
import easyocr

from ai_engine import ask_ai


# -----------------------------
# PDF text extraction
# -----------------------------
def read_pdf(file):
    text = ""

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text.strip()


# -----------------------------
# DOCX text extraction
# -----------------------------
def read_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()


# -----------------------------
# TXT text extraction
# -----------------------------
def read_txt(file):
    return file.read().decode("utf-8", errors="ignore").strip()


# -----------------------------
# Image OCR extraction
# -----------------------------
def read_image(file):
    suffix = os.path.splitext(file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file.read())
        temp_path = temp_file.name

    try:
        reader = easyocr.Reader(["en"], gpu=False)
        result = reader.readtext(temp_path, detail=0)
        text = "\n".join(result)
        return text.strip()

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# -----------------------------
# Extract JD text
# -----------------------------
def extract_jd_text(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return read_pdf(uploaded_file)

    elif filename.endswith(".docx"):
        return read_docx(uploaded_file)

    elif filename.endswith(".txt"):
        return read_txt(uploaded_file)

    elif filename.endswith((".png", ".jpg", ".jpeg")):
        return read_image(uploaded_file)

    else:
        return ""


# -----------------------------
# Analyze JD using Groq
# -----------------------------
def analyze_jd(uploaded_file):
    jd_text = extract_jd_text(uploaded_file)

    if not jd_text:
        return {
            "job_title": "",
            "experience": "",
            "education": "",
            "location": "",
            "mandatory_skills": [],
            "good_to_have": [],
            "responsibilities": [],
            "keywords": [],
            "raw_text": ""
        }

    prompt = f"""
You are an expert HR recruiter.

Analyze the following Job Description and extract structured information.

Return ONLY valid JSON.
Do not add explanation.
Do not add markdown.

Required JSON format:

{{
  "job_title": "",
  "experience": "",
  "education": "",
  "location": "",
  "mandatory_skills": [],
  "good_to_have": [],
  "responsibilities": [],
  "keywords": []
}}

Job Description:
{jd_text}
"""

    response = ask_ai(prompt)

    try:
        data = json.loads(response)

    except Exception:
        data = {
            "job_title": "",
            "experience": "",
            "education": "",
            "location": "",
            "mandatory_skills": [],
            "good_to_have": [],
            "responsibilities": [],
            "keywords": []
        }

    data["raw_text"] = jd_text
    return data