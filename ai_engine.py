import os
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def calculate_experience_from_text(resume_text):
    current_date = datetime.now()
    work_text = resume_text

    if "Work Experience" in resume_text:
        work_text = resume_text.split("Work Experience", 1)[1]

        for marker in ["Skills", "Education", "Certificates", "Projects"]:
            if marker in work_text:
                work_text = work_text.split(marker, 1)[0]
                break

    pattern = r"(\d{4})/(\d{2})\s*[–-]\s*(present|Present|\d{4}/\d{2})"
    matches = re.findall(pattern, work_text)

    total_months = 0

    for start_year, start_month, end_value in matches:
        start_date = datetime(int(start_year), int(start_month), 1)

        if end_value.lower() == "present":
            end_date = current_date
        else:
            end_year, end_month = end_value.split("/")
            end_date = datetime(int(end_year), int(end_month), 1)

        months = (end_date.year - start_date.year) * 12 + (
            end_date.month - start_date.month
        )

        if months > 0:
            total_months += months

    years = total_months // 12
    months = total_months % 12

    return f"{years} years {months} months"


def extract_email(resume_text):
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", resume_text)
    return match.group(0) if match else "Not Found"


def extract_phone(resume_text):
    match = re.search(r"\b[6-9]\d{9}\b", resume_text)
    return match.group(0) if match else "Not Found"


def extract_basic_name(resume_text):
    lines = [line.strip() for line in resume_text.split("\n") if line.strip()]
    return lines[0] if lines else "Not Found"


def ask_ai(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a careful AI assistant. Return only the requested output."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1
    )

    return response.choices[0].message.content


def clean_json_response(response):
    response = response.strip()
    response = re.sub(r"```json|```", "", response).strip()

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        response = match.group(0)

    return response


def extract_resume_skills_ai(resume_text):
    prompt = f"""
You are an expert resume parser.

Extract only the technical skills, tools, programming languages, databases,
cloud platforms, frameworks, and libraries from this resume.

Return ONLY valid JSON.

Format:
{{
  "skills": []
}}

Resume Text:
{resume_text}
"""

    try:
        response = ask_ai(prompt)
        cleaned = clean_json_response(response)
        data = json.loads(cleaned)
        return data.get("skills", [])
    except Exception:
        return []


def structured_resume_parser(resume_text):
    experience = calculate_experience_from_text(resume_text)
    skills = extract_resume_skills_ai(resume_text)

    structured_data = {
        "name": extract_basic_name(resume_text),
        "email": extract_email(resume_text),
        "phone": extract_phone(resume_text),
        "total_experience": experience,
        "skills": skills
    }

    return structured_data


def analyze_resume(resume_text):
    structured_data = structured_resume_parser(resume_text)

    prompt = f"""
You are an expert HR resume screening assistant.

Use this factual data exactly:
Name: {structured_data["name"]}
Email: {structured_data["email"]}
Phone: {structured_data["phone"]}
Total Experience: {structured_data["total_experience"]}
Skills: {structured_data["skills"]}

Now analyze the resume.

Return output in this format:

Candidate Name:
Email:
Phone:
Total Experience:
Current/Recent Role:
Technical Skills:
Tools:
Cloud Platforms:
Databases:
Programming Languages:
Education:
Projects:
Certifications:
Strengths:
Weaknesses:
Short Summary:

Resume Text:
{resume_text}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a careful HR resume analyzer. Do not hallucinate facts."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1
    )

    return response.choices[0].message.content, structured_data