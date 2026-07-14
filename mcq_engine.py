import json
import re
import time
from typing import Any, Dict, List, Optional

from ai_engine import ask_ai


DEFAULT_TOTAL_QUESTIONS = 10


PERSONAL_QUESTION_PATTERNS = (
    "how many years",
    "years of experience",
    "tell me about yourself",
    "what is your experience",
    "which skill do you have",
    "what skills do you have",
    "what are your strengths",
    "what are your weaknesses",
    "expected salary",
    "salary expectation",
    "why should we hire",
    "career goal",
    "preferred role",
    "describe yourself",
    "where do you see yourself",
)


def _clean_ai_response(raw_response: Any) -> str:
    """Convert the AI response to plain text and remove code fences."""

    if raw_response is None:
        return ""

    # Support AI response objects that contain a text attribute.
    if hasattr(raw_response, "text"):
        text = str(raw_response.text).strip()
    else:
        text = str(raw_response).strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    return text.strip()


def _parse_json_array(
    raw_response: Any,
) -> List[Dict[str, Any]]:
    """Extract and parse the JSON array returned by the AI."""

    text = _clean_ai_response(raw_response)

    if not text:
        print("Empty AI response received.")
        return []

    start_position = text.find("[")
    end_position = text.rfind("]")

    if (
        start_position == -1
        or end_position == -1
        or end_position <= start_position
    ):
        print(
            "JSON parsing skipped because a complete "
            "JSON array was not found."
        )
        print("Raw AI response:")
        print(text)
        return []

    json_text = text[start_position:end_position + 1]

    try:
        parsed_data = json.loads(json_text)

    except json.JSONDecodeError as error:
        print(f"JSON parsing failed: {error}")
        print("Raw AI response:")
        print(text)
        return []

    if not isinstance(parsed_data, list):
        print("AI response JSON is not a list.")
        return []

    return [
        item
        for item in parsed_data
        if isinstance(item, dict)
    ]


def _normalize_text(value: Any) -> str:
    """Remove extra spaces and safely convert values to text."""

    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()


def _normalize_key(value: Any) -> str:
    """Create a normalized key for duplicate checking."""

    text = _normalize_text(value).casefold()

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def _is_technical_question(question: str) -> bool:
    """Reject personal, HR and nontechnical questions."""

    lowered_question = question.casefold()

    return not any(
        pattern in lowered_question
        for pattern in PERSONAL_QUESTION_PATTERNS
    )


def _validate_question(
    item: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Validate and normalize one MCQ."""

    question = _normalize_text(
        item.get("question")
    )

    options = item.get("options")

    correct_answer = _normalize_text(
        item.get("correct_answer")
        or item.get("answer")
    )

    category = _normalize_text(
        item.get("category")
    ) or "Technical"

    difficulty = _normalize_text(
        item.get("difficulty")
    ) or "Intermediate"

    if not question:
        return None

    if not _is_technical_question(question):
        print(
            f"Rejected nontechnical question: {question}"
        )
        return None

    if not isinstance(options, list):
        return None

    if len(options) != 4:
        return None

    cleaned_options = [
        _normalize_text(option)
        for option in options
    ]

    if any(
        not option
        for option in cleaned_options
    ):
        return None

    # All four options must be unique.
    normalized_options = [
        option.casefold()
        for option in cleaned_options
    ]

    if len(set(normalized_options)) != 4:
        return None

    # Correct answer must exactly match one option,
    # ignoring uppercase/lowercase differences.
    matching_option = next(
        (
            option
            for option in cleaned_options
            if option.casefold()
            == correct_answer.casefold()
        ),
        None,
    )

    # Also support answers such as A, B, C, D.
    if matching_option is None:
        answer_letter_mapping = {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3,
            "option a": 0,
            "option b": 1,
            "option c": 2,
            "option d": 3,
        }

        answer_index = answer_letter_mapping.get(
            correct_answer.casefold()
        )

        if answer_index is not None:
            matching_option = cleaned_options[
                answer_index
            ]

    if matching_option is None:
        return None

    allowed_difficulties = {
        "easy": "Easy",
        "intermediate": "Intermediate",
        "medium": "Intermediate",
        "hard": "Hard",
    }

    difficulty = allowed_difficulties.get(
        difficulty.casefold(),
        "Intermediate",
    )

    return {
        "question": question,
        "options": cleaned_options,
        "correct_answer": matching_option,
        "category": category,
        "difficulty": difficulty,
    }


def _compact_json(
    data: Any,
    limit: int = 12000,
) -> str:
    """Convert structured data to compact prompt text."""

    try:
        text = json.dumps(
            data,
            ensure_ascii=False,
            default=str,
        )

    except (TypeError, ValueError):
        text = str(data)

    if len(text) > limit:
        return text[:limit] + "..."

    return text


def _flatten_values(data: Any) -> List[str]:
    """Recursively extract text values from dictionaries and lists."""

    values: List[str] = []

    if isinstance(data, dict):
        for value in data.values():
            values.extend(
                _flatten_values(value)
            )

    elif isinstance(data, (list, tuple, set)):
        for value in data:
            values.extend(
                _flatten_values(value)
            )

    elif data is not None:
        text = _normalize_text(data)

        if text:
            values.append(text)

    return values


def get_candidate_skills(
    candidate_data: Dict[str, Any],
) -> List[str]:
    """
    Extract technical skills from candidate data.

    Supports multiple possible keys because resume-analysis
    output formats may differ.
    """

    possible_skill_keys = (
        "skills",
        "technical_skills",
        "technical skills",
        "programming_languages",
        "programming languages",
        "tools",
        "technologies",
        "frameworks",
        "databases",
        "libraries",
        "software",
        "cloud",
        "platforms",
    )

    extracted_skills: List[str] = []

    for key, value in candidate_data.items():
        normalized_key = str(key).strip().casefold()

        if normalized_key in possible_skill_keys:
            extracted_skills.extend(
                _flatten_values(value)
            )

    # Sometimes all useful information is nested.
    if not extracted_skills:
        for key in possible_skill_keys:
            value = candidate_data.get(key)

            if value:
                extracted_skills.extend(
                    _flatten_values(value)
                )

    cleaned_skills: List[str] = []
    seen_skills = set()

    for skill_entry in extracted_skills:
        # Split comma-separated skill strings.
        individual_skills = re.split(
            r"[,;|/\n]+",
            skill_entry,
        )

        for skill in individual_skills:
            skill = _normalize_text(skill)

            if not skill:
                continue

            normalized_skill = skill.casefold()

            if normalized_skill not in seen_skills:
                seen_skills.add(normalized_skill)
                cleaned_skills.append(skill)

    return cleaned_skills


def _build_prompt(
    candidate_data: Dict[str, Any],
    jd_data: Optional[Dict[str, Any]],
    match_result: Optional[Dict[str, Any]],
    question_count: int,
    excluded_questions: List[str],
) -> str:
    """Create the MCQ-generation prompt."""

    excluded_text = "\n".join(
        f"- {question}"
        for question in excluded_questions[-40:]
    )

    if not excluded_text:
        excluded_text = "None"

    candidate_skills = get_candidate_skills(
        candidate_data
    )

    skill_text = ", ".join(candidate_skills)

    return f"""
You are an expert technical interviewer creating a recruitment screening test.

Generate exactly {question_count} NEW technical multiple-choice questions.

CANDIDATE'S EXTRACTED TECHNICAL SKILLS:
{skill_text}

CANDIDATE STRUCTURED RESUME DATA:
{_compact_json(candidate_data)}

JOB DESCRIPTION DATA:
{_compact_json(jd_data or {})}

RESUME-JD MATCH INFORMATION:
{_compact_json(match_result or {})}

QUESTIONS ALREADY GENERATED.
DO NOT REPEAT OR PARAPHRASE THEM:
{excluded_text}

STRICT RULES:

1. Generate only objective technical questions.

2. Base the questions primarily on the candidate's technical skills,
   technologies, tools, frameworks, databases, programming languages,
   machine-learning methods and project technologies.

3. The questions must be suitable for testing actual technical knowledge.

4. Include practical, conceptual, debugging, output-based and
   scenario-based technical questions.

5. Do not ask:
   - How many years of experience do you have?
   - Which skills do you have?
   - Tell me about yourself.
   - What are your strengths or weaknesses?
   - Why should we hire you?
   - Salary or career-goal questions.
   - Any HR or behavioural questions.

6. Do not ask questions whose answers depend on the candidate's personal
   experience, opinion or resume wording.

7. Each question must have exactly four non-empty, technically meaningful
   and unique options.

8. Exactly one option must be correct.

9. The value of "correct_answer" must exactly match one of the four
   option strings.

10. Avoid duplicate questions and avoid testing the same concept repeatedly.

11. Prefer approximately:
    - 20 percent Easy
    - 60 percent Intermediate
    - 20 percent Hard

12. Cover multiple candidate skills instead of creating all questions
    from only one technology.

13. Return only one valid JSON array.

14. Do not include Markdown, headings, explanations, comments or code fences.

REQUIRED JSON FORMAT:

[
  {{
    "question": "Technical question text",
    "options": [
      "Option 1",
      "Option 2",
      "Option 3",
      "Option 4"
    ],
    "correct_answer": "Option 1",
    "category": "Python",
    "difficulty": "Intermediate"
  }}
]
""".strip()


def remove_duplicate_questions(
    questions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Remove exact and strongly similar duplicate questions."""

    unique_questions: List[Dict[str, Any]] = []
    seen_question_keys = set()

    for question_data in questions:
        question_text = question_data.get(
            "question",
            "",
        )

        question_key = _normalize_key(
            question_text
        )

        if not question_key:
            continue

        if question_key in seen_question_keys:
            continue

        # Compare word overlap to catch near duplicates.
        current_words = set(
            question_key.split()
        )

        is_near_duplicate = False

        for existing_question in unique_questions:
            existing_key = _normalize_key(
                existing_question.get(
                    "question",
                    "",
                )
            )

            existing_words = set(
                existing_key.split()
            )

            if not current_words or not existing_words:
                continue

            common_words = (
                current_words & existing_words
            )

            similarity = len(common_words) / max(
                len(current_words),
                len(existing_words),
            )

            if similarity >= 0.85:
                is_near_duplicate = True
                break

        if is_near_duplicate:
            continue

        seen_question_keys.add(question_key)
        unique_questions.append(question_data)

    return unique_questions


def generate_llm_mcqs(
    jd_data: Optional[Dict[str, Any]],
    candidate_data: Dict[str, Any],
    required_count: int,
    existing_questions: Optional[
        List[Dict[str, Any]]
    ] = None,
    match_result: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Generate and validate one batch of questions."""

    existing_questions = existing_questions or []

    excluded_questions = [
        question.get("question", "")
        for question in existing_questions
        if question.get("question")
    ]

    prompt = _build_prompt(
        candidate_data=candidate_data,
        jd_data=jd_data,
        match_result=match_result,
        question_count=required_count,
        excluded_questions=excluded_questions,
    )

    try:
        raw_response = ask_ai(prompt)

    except Exception as error:
        print(
            f"AI request failed: {error}"
        )
        return []

    print("\nRaw MCQ AI response:")
    print(raw_response)

    parsed_questions = _parse_json_array(
        raw_response
    )

    valid_questions: List[Dict[str, Any]] = []

    for item in parsed_questions:
        validated_question = _validate_question(
            item
        )

        if validated_question is not None:
            valid_questions.append(
                validated_question
            )

    return remove_duplicate_questions(
        valid_questions
    )


def generate_mcqs(
    jd_data: Optional[Dict[str, Any]] = None,
    candidate_data: Optional[Dict[str, Any]] = None,
    match_result: Optional[Dict[str, Any]] = None,
    total_questions: int = DEFAULT_TOTAL_QUESTIONS,
    maximum_attempts: int = 4,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Generate technical MCQs.

    This function accepts total_questions, so app.py can call:

    generate_mcqs(
        jd_data=jd_data,
        candidate_data=candidate_data,
        match_result=match_result,
        total_questions=20
    )
    """

    # Backward compatibility for alternate argument names.
    if candidate_data is None:
        candidate_data = (
            kwargs.get("resume_data")
            or kwargs.get("candidate_analysis")
            or kwargs.get("resume_analysis")
        )

    if jd_data is None:
        jd_data = (
            kwargs.get("job_data")
            or kwargs.get("job_description_data")
        )

    if not isinstance(candidate_data, dict):
        raise ValueError(
            "Candidate data is missing or invalid. "
            "Pass the resume analysis dictionary as candidate_data."
        )

    try:
        total_questions = int(total_questions)

    except (TypeError, ValueError):
        total_questions = DEFAULT_TOTAL_QUESTIONS

    if total_questions <= 0:
        raise ValueError(
            "total_questions must be greater than zero."
        )

    total_questions = min(
        total_questions,
        50,
    )

    candidate_skills = get_candidate_skills(
        candidate_data
    )

    if not candidate_skills:
        raise ValueError(
            "No technical skills were extracted from the resume. "
            "Check the candidate_data structure printed in the terminal."
        )

    print(
        "\nCandidate skills used for MCQ generation:"
    )

    print(candidate_skills)

    final_questions: List[Dict[str, Any]] = []

    maximum_attempts = max(
        1,
        int(maximum_attempts),
    )

    for attempt in range(maximum_attempts):
        remaining_count = (
            total_questions
            - len(final_questions)
        )

        if remaining_count <= 0:
            break

        # Ask for a buffer because the validator may reject
        # malformed or duplicate questions.
        if attempt == 0:
            request_count = min(
                total_questions + 4,
                30,
            )
        else:
            request_count = min(
                remaining_count + 3,
                12,
            )

        print(
            f"\nMCQ attempt {attempt + 1}/{maximum_attempts}: "
            f"{len(final_questions)} valid questions available. "
            f"Requesting {request_count} questions."
        )

        new_questions = generate_llm_mcqs(
            jd_data=jd_data,
            candidate_data=candidate_data,
            required_count=request_count,
            existing_questions=final_questions,
            match_result=match_result,
        )

        final_questions.extend(
            new_questions
        )

        final_questions = remove_duplicate_questions(
            final_questions
        )

        print(
            f"Valid unique questions after attempt "
            f"{attempt + 1}: {len(final_questions)}"
        )

        if len(final_questions) >= total_questions:
            break

        # Small pause to avoid immediately hitting an API
        # rate limit between retry requests.
        time.sleep(1)

    if len(final_questions) < total_questions:
        raise RuntimeError(
            f"Only {len(final_questions)} valid technical questions "
            f"were generated after {maximum_attempts} attempts. "
            f"The remaining questions were rejected because of "
            f"duplicate questions, invalid options, missing answers "
            f"or invalid JSON structure. Check the VS Code terminal "
            f"for the raw AI responses."
        )

    return final_questions[
        :total_questions
    ]