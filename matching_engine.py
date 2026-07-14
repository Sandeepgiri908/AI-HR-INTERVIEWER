import json
import re
from ai_engine import ask_ai


def clean_json_response(response):
    response = response.strip()
    response = re.sub(r"```json|```", "", response).strip()

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        response = match.group(0)

    return response


def extract_years(text):
    text = str(text).lower()

    match = re.search(r"(\d+)\s*years?", text)
    if match:
        return int(match.group(1))

    match = re.search(r"(\d+)\s*-\s*(\d+)", text)
    if match:
        return int(match.group(1))

    return 0


def llm_skill_matching(resume_skills, jd_skills):
    prompt = f"""
You are an expert ATS skill matcher.

Compare each JD skill with candidate resume skills.

Return ONLY valid JSON.

Format:
{{
  "skill_matches": [
    {{
      "jd_skill": "",
      "matched_resume_skill": "",
      "status": "Matched or Missing",
      "confidence": 0.0
    }}
  ]
}}

Rules:
- Check every JD skill.
- Use semantic matching.
- Example: ML = Machine Learning.
- Example: PowerBI = Power BI.
- Example: GenAI = Generative AI.
- Example: PostgreSQL = Postgres.
- If skill is clearly present, status = Matched.
- If not present, status = Missing.
- confidence should be between 0 and 1.

Resume Skills:
{resume_skills}

JD Skills:
{jd_skills}
"""

    try:
        response = ask_ai(prompt)
        cleaned = clean_json_response(response)
        data = json.loads(cleaned)
        return data.get("skill_matches", [])
    except Exception:
        return []


def calculate_skill_score(skill_matches):
    if not skill_matches:
        return 0, [], []

    total = len(skill_matches)
    matched = []

    for item in skill_matches:
        if str(item.get("status", "")).lower() == "matched":
            matched.append(item)

    score = round((len(matched) / total) * 100)

    matched_skills = [
        item.get("jd_skill", "")
        for item in skill_matches
        if str(item.get("status", "")).lower() == "matched"
    ]

    missing_skills = [
        item.get("jd_skill", "")
        for item in skill_matches
        if str(item.get("status", "")).lower() == "missing"
    ]

    return score, matched_skills, missing_skills


def calculate_experience_score(resume_exp, jd_exp):
    candidate_years = extract_years(resume_exp)

    jd_text = str(jd_exp).lower()
    range_match = re.search(r"(\d+)\s*[-to]+\s*(\d+)", jd_text)

    if range_match:
        min_exp = int(range_match.group(1))

        if candidate_years >= min_exp:
            return 100
        return round((candidate_years / min_exp) * 100)

    required_years = extract_years(jd_exp)

    if required_years == 0:
        return 70

    if candidate_years >= required_years:
        return 100

    return round((candidate_years / required_years) * 100)


def calculate_education_score(resume_data, jd_data):
    resume_text = str(resume_data).lower()
    jd_education = str(jd_data.get("education", "")).lower()

    bachelor_keywords = ["bachelor", "b.tech", "btech", "be", "b.e", "degree", "graduation"]
    master_keywords = ["master", "m.tech", "mtech", "mba", "mca", "msc", "m.sc"]

    if not jd_education:
        return 70

    if any(k in jd_education for k in master_keywords):
        if any(k in resume_text for k in master_keywords):
            return 100
        elif any(k in resume_text for k in bachelor_keywords):
            return 70
        return 30

    if any(k in jd_education for k in bachelor_keywords):
        if any(k in resume_text for k in bachelor_keywords + master_keywords):
            return 100
        return 30

    return 70


def generate_recommendation(score):
    if score >= 85:
        return "Strong Match", "High"
    elif score >= 70:
        return "Good Match", "High"
    elif score >= 50:
        return "Average Match", "Medium"
    else:
        return "Weak Match", "Low"


def generate_ai_summary(resume_data, jd_data, matched_skills, missing_skills):
    prompt = f"""
You are an HR screening assistant.

Generate candidate strengths, weaknesses and short summary.

Return ONLY valid JSON.

Format:
{{
  "candidate_strengths": [],
  "candidate_weaknesses": [],
  "summary": ""
}}

Resume Data:
{resume_data}

JD Data:
{jd_data}

Matched Skills:
{matched_skills}

Missing Skills:
{missing_skills}
"""

    try:
        response = ask_ai(prompt)
        cleaned = clean_json_response(response)
        return json.loads(cleaned)
    except Exception:
        return {
            "candidate_strengths": [],
            "candidate_weaknesses": [],
            "summary": ""
        }


def calculate_match(resume_data, jd_data):
    resume_skills = resume_data.get("skills", [])

    if isinstance(resume_skills, str):
        resume_skills = [s.strip() for s in resume_skills.split(",") if s.strip()]

    jd_skills = []
    jd_skills.extend(jd_data.get("mandatory_skills", []))
    jd_skills.extend(jd_data.get("good_to_have", []))

    skill_matches = llm_skill_matching(resume_skills, jd_skills)

    skill_score, matched_skills, missing_skills = calculate_skill_score(skill_matches)

    experience_score = calculate_experience_score(
        resume_data.get("total_experience", ""),
        jd_data.get("experience", "")
    )

    education_score = calculate_education_score(resume_data, jd_data)

    overall_score = round(
        (skill_score * 0.60) +
        (experience_score * 0.25) +
        (education_score * 0.15)
    )

    hiring_recommendation, interview_probability = generate_recommendation(overall_score)

    ai_data = generate_ai_summary(
        resume_data,
        jd_data,
        matched_skills,
        missing_skills
    )

    return {
        "overall_match_score": overall_score,
        "skill_match_score": skill_score,
        "experience_match_score": experience_score,
        "education_match_score": education_score,
        "skill_match_details": skill_matches,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "candidate_strengths": ai_data.get("candidate_strengths", []),
        "candidate_weaknesses": ai_data.get("candidate_weaknesses", []),
        "hiring_recommendation": hiring_recommendation,
        "interview_probability": interview_probability,
        "summary": ai_data.get("summary", "")
    }