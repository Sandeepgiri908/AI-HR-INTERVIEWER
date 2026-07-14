import re


SKILL_SYNONYMS = {
    "py": "python",
    "python programming": "python",
    "core python": "python",

    "ml": "machine learning",
    "machine learning": "machine learning",

    "dl": "deep learning",
    "deep learning": "deep learning",

    "genai": "generative ai",
    "gen ai": "generative ai",
    "generative ai": "generative ai",

    "llm": "large language models",
    "llms": "large language models",
    "large language model": "large language models",
    "large language models": "large language models",

    "powerbi": "power bi",
    "power bi": "power bi",
    "microsoft power bi": "power bi",

    "postgres": "postgresql",
    "postgre sql": "postgresql",
    "postgresql": "postgresql",

    "ms sql": "sql server",
    "mssql": "sql server",
    "sql server": "sql server",

    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "scikit-learn": "scikit-learn",

    "tf": "tensorflow",
    "tensorflow": "tensorflow",

    "torch": "pytorch",
    "pytorch": "pytorch",

    "aws cloud": "aws",
    "amazon web services": "aws",
    "aws": "aws",

    "gcp": "google cloud",
    "google cloud platform": "google cloud",
    "google cloud": "google cloud",

    "azure cloud": "azure",
    "microsoft azure": "azure",
    "azure": "azure",

    "langchain": "langchain",
    "lang chain": "langchain",

    "rag": "retrieval augmented generation",
    "retrieval augmented generation": "retrieval augmented generation",

    "xg boost": "xgboost",
    "xgboost": "xgboost",

    "light gbm": "lightgbm",
    "lightgbm": "lightgbm",

    "cat boost": "catboost",
    "catboost": "catboost"
}


def clean_skill(skill):
    skill = str(skill).lower().strip()
    skill = skill.replace("_", " ")
    skill = re.sub(r"[^a-z0-9+#. ]", "", skill)
    skill = re.sub(r"\s+", " ", skill).strip()
    return skill


def normalize_skill(skill):
    cleaned = clean_skill(skill)

    if cleaned in SKILL_SYNONYMS:
        return SKILL_SYNONYMS[cleaned]

    return cleaned


def normalize_skill_list(skills):
    normalized = []

    for skill in skills:
        final_skill = normalize_skill(skill)
        if final_skill and final_skill not in normalized:
            normalized.append(final_skill)

    return normalized