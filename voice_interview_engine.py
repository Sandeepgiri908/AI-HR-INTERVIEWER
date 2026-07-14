import json
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq

from ai_engine import ask_ai


load_dotenv()

DEFAULT_INTERVIEW_QUESTIONS = 5

NON_TECHNICAL_PATTERNS = (
    "tell me about yourself",
    "introduce yourself",
    "salary expectation",
    "expected salary",
    "why should we hire",
    "what are your strengths",
    "what are your weaknesses",
    "where do you see yourself",
    "how many years of experience",
    "what is your experience",
    "which skills do you have",
)


def _get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY was not found. Add it to your .env file."
        )

    return Groq(api_key=api_key)


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_ai_response(raw_response: Any) -> str:
    if raw_response is None:
        return ""

    if hasattr(raw_response, "text"):
        text = str(raw_response.text)
    else:
        text = str(raw_response)

    text = text.strip()
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```$", "", text)

    return text.strip()


def _parse_json(raw_response: Any) -> Any:
    text = _clean_ai_response(raw_response)

    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    array_start = text.find("[")
    array_end = text.rfind("]")

    if (
        array_start != -1
        and array_end != -1
        and array_end > array_start
    ):
        try:
            return json.loads(
                text[array_start:array_end + 1]
            )
        except json.JSONDecodeError:
            pass

    object_start = text.find("{")
    object_end = text.rfind("}")

    if (
        object_start != -1
        and object_end != -1
        and object_end > object_start
    ):
        try:
            return json.loads(
                text[object_start:object_end + 1]
            )
        except json.JSONDecodeError:
            pass

    print("Could not parse AI JSON response:")
    print(text)

    return None


def _compact_json(data: Any, limit: int = 10000) -> str:
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


def _is_technical_question(question: str) -> bool:
    lowered_question = question.casefold()

    return not any(
        pattern in lowered_question
        for pattern in NON_TECHNICAL_PATTERNS
    )


def _question_key(question: str) -> str:
    question = question.casefold()
    question = re.sub(r"[^a-z0-9\s]", "", question)
    return re.sub(r"\s+", " ", question).strip()


def generate_voice_interview_questions(
    candidate_data: Dict[str, Any],
    jd_data: Optional[Dict[str, Any]] = None,
    match_result: Optional[Dict[str, Any]] = None,
    total_questions: int = DEFAULT_INTERVIEW_QUESTIONS,
) -> List[Dict[str, str]]:
    if not isinstance(candidate_data, dict):
        raise ValueError(
            "Candidate data is missing or invalid."
        )

    try:
        total_questions = int(total_questions)
    except (TypeError, ValueError):
        total_questions = DEFAULT_INTERVIEW_QUESTIONS

    total_questions = max(1, min(total_questions, 10))

    prompt = f"""
You are an expert technical interviewer.

Generate exactly {total_questions} technical and project-based
voice interview questions.

CANDIDATE RESUME DATA:
{_compact_json(candidate_data)}

JOB DESCRIPTION DATA:
{_compact_json(jd_data or {})}

RESUME-JD MATCH RESULT:
{_compact_json(match_result or {})}

STRICT RULES:

1. Ask only technical and project-related questions.
2. Give priority to candidate projects, programming languages,
   databases, machine learning methods, frameworks, tools,
   cloud platforms, algorithms and technologies.
3. At least half of the questions must be based on practical
   implementation or candidate projects.
4. Questions should require explanation of how, why, implementation,
   evaluation, debugging, limitations or improvements.
5. Do not ask HR, personal, salary, experience-duration,
   strength, weakness or self-introduction questions.
6. Each question should be answerable in about 20 to 40 seconds.
7. Do not generate duplicate or nearly duplicate questions.
8. Return only a valid JSON array.
9. Do not include Markdown or code fences.

REQUIRED FORMAT:

[
  {{
    "question": "Explain how you evaluated the model in your project.",
    "category": "Machine Learning Project",
    "difficulty": "Intermediate",
    "expected_topics": "Metrics, validation, overfitting and business impact"
  }}
]
""".strip()

    raw_response = ask_ai(prompt)

    print("\nRaw voice interview question response:")
    print(raw_response)

    parsed_response = _parse_json(raw_response)

    if not isinstance(parsed_response, list):
        raise RuntimeError(
            "The AI did not return a valid question list."
        )

    final_questions: List[Dict[str, str]] = []
    seen_questions = set()

    for item in parsed_response:
        if not isinstance(item, dict):
            continue

        question = _normalize_text(
            item.get("question")
        )
        category = _normalize_text(
            item.get("category")
        ) or "Technical"
        difficulty = _normalize_text(
            item.get("difficulty")
        ) or "Intermediate"
        expected_topics = _normalize_text(
            item.get("expected_topics")
        )

        if not question:
            continue

        if not _is_technical_question(question):
            continue

        key = _question_key(question)

        if not key or key in seen_questions:
            continue

        seen_questions.add(key)

        final_questions.append(
            {
                "question": question,
                "category": category,
                "difficulty": difficulty,
                "expected_topics": expected_topics,
            }
        )

        if len(final_questions) >= total_questions:
            break

    if len(final_questions) < total_questions:
        raise RuntimeError(
            f"Only {len(final_questions)} valid voice interview "
            f"questions were generated."
        )

    return final_questions


def transcribe_audio(audio_file: Any) -> str:
    if audio_file is None:
        raise ValueError(
            "No audio recording was provided."
        )

    client = _get_groq_client()
    audio_bytes = audio_file.getvalue()

    if not audio_bytes:
        raise ValueError(
            "The recorded audio file is empty."
        )

    suffix = ".wav"
    audio_name = getattr(audio_file, "name", "")

    if audio_name:
        extension = os.path.splitext(audio_name)[1]

        if extension:
            suffix = extension

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temporary_file:
            temporary_file.write(audio_bytes)
            temporary_path = temporary_file.name

        with open(temporary_path, "rb") as audio_stream:
            transcription = (
                client.audio.transcriptions.create(
                    file=audio_stream,
                    model="whisper-large-v3-turbo",
                    language="en",
                    response_format="json",
                    temperature=0.0,
                )
            )

        transcript = _normalize_text(
            getattr(transcription, "text", "")
        )

        if not transcript:
            raise RuntimeError(
                "Speech transcription returned empty text."
            )

        return transcript

    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def evaluate_voice_answer(
    question_data: Dict[str, Any],
    candidate_answer: str,
    candidate_data: Optional[Dict[str, Any]] = None,
    jd_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    question = _normalize_text(
        question_data.get("question")
    )
    expected_topics = _normalize_text(
        question_data.get("expected_topics")
    )
    candidate_answer = _normalize_text(candidate_answer)

    if not candidate_answer:
        return {
            "score": 0,
            "technical_correctness": 0,
            "explanation_quality": 0,
            "practical_understanding": 0,
            "clarity_relevance": 0,
            "feedback": "No valid answer was recorded.",
            "strengths": [],
            "missing_points": [
                "The candidate did not provide an answer."
            ],
            "ideal_answer_summary": "",
        }

    prompt = f"""
You are a strict and unbiased senior technical interviewer.

Evaluate the candidate's answer only against the technical requirements
of the question. Different candidates and answers must receive genuinely
different scores based on their quality.

QUESTION:
{question}

EXPECTED TECHNICAL TOPICS:
{expected_topics}

CANDIDATE ANSWER:
{candidate_answer}

CANDIDATE RESUME CONTEXT:
{_compact_json(candidate_data or {}, limit=6000)}

JOB DESCRIPTION CONTEXT:
{_compact_json(jd_data or {}, limit=4000)}

SCORING RUBRIC:

1. Technical correctness: 0 to 50
   - 0–10: mostly incorrect or unrelated
   - 11–25: limited understanding with major gaps
   - 26–40: mostly correct with some missing details
   - 41–50: technically strong and complete

2. Explanation quality: 0 to 20
   - Consider structure, reasoning and depth.

3. Practical understanding: 0 to 20
   - Consider implementation, debugging, limitations,
     trade-offs and real-world usage.

4. Clarity and relevance: 0 to 10
   - Consider whether the answer directly addresses the question.

STRICT EVALUATION RULES:

1. Do not use a fixed, default or repeated score.
2. Judge this specific candidate answer independently.
3. The final score must equal the sum of all four component scores.
4. Penalize incorrect statements, vague language, missing concepts,
   repetition and irrelevant information.
5. Strengths must mention specific correct points from this answer.
6. Missing points must mention specific concepts absent or incorrect
   in this answer.
7. Do not give generic weaknesses such as
   "needs more detail" without naming the missing technical detail.
8. Generate the ideal answer independently using the question,
   expected topics and established technical knowledge.
9. The ideal answer must not be a rewritten or polished version of
   the candidate's answer.
10. If the candidate gives incorrect information, the ideal answer
    must correct it.
11. Return only a valid JSON object without Markdown or code fences.

REQUIRED JSON STRUCTURE:

{{
  "technical_correctness": <integer from 0 to 50>,
  "explanation_quality": <integer from 0 to 20>,
  "practical_understanding": <integer from 0 to 20>,
  "clarity_relevance": <integer from 0 to 10>,
  "score": <sum of the four component scores>,
  "feedback": "<specific feedback based on this answer>",
  "strengths": [
    "<specific technically correct point stated by the candidate>"
  ],
  "missing_points": [
    "<specific missing or incorrect technical concept>"
  ],
  "ideal_answer_summary": "<an independent technically correct ideal answer of approximately 80 to 150 words>"
}}
""".strip()
    raw_response = ask_ai(prompt)

    print("\nRaw voice answer evaluation response:")
    print(raw_response)

    parsed_response = _parse_json(raw_response)

    if not isinstance(parsed_response, dict):
        raise RuntimeError(
            "The AI did not return valid answer evaluation JSON."
        )

    def safe_score(value: Any, maximum: int) -> int:
        try:
            numeric_value = int(round(float(value)))
        except (TypeError, ValueError):
            numeric_value = 0

        return max(0, min(numeric_value, maximum))

    technical_correctness = safe_score(
        parsed_response.get("technical_correctness"),
        50,
    )
    explanation_quality = safe_score(
        parsed_response.get("explanation_quality"),
        20,
    )
    practical_understanding = safe_score(
        parsed_response.get("practical_understanding"),
        20,
    )
    clarity_relevance = safe_score(
        parsed_response.get("clarity_relevance"),
        10,
    )

    calculated_score = (
        technical_correctness
        + explanation_quality
        + practical_understanding
        + clarity_relevance
    )

    strengths = parsed_response.get("strengths", [])
    missing_points = parsed_response.get("missing_points", [])

    if not isinstance(strengths, list):
        strengths = [str(strengths)]

    if not isinstance(missing_points, list):
        missing_points = [str(missing_points)]

    return {
        "score": calculated_score,
        "technical_correctness": technical_correctness,
        "explanation_quality": explanation_quality,
        "practical_understanding": practical_understanding,
        "clarity_relevance": clarity_relevance,
        "feedback": _normalize_text(
            parsed_response.get("feedback")
        ),
        "strengths": [
            _normalize_text(item)
            for item in strengths
            if _normalize_text(item)
        ],
        "missing_points": [
            _normalize_text(item)
            for item in missing_points
            if _normalize_text(item)
        ],
        "ideal_answer_summary": _normalize_text(
            parsed_response.get("ideal_answer_summary")
        ),
    }


def calculate_final_voice_result(
    interview_answers: List[Dict[str, Any]],
    qualification_score: float = 60.0,
) -> Dict[str, Any]:
    if not interview_answers:
        return {
            "score": 0.0,
            "status": "Not Qualified",
            "answered_questions": 0,
            "total_questions": 0,
            "strengths": [],
            "improvement_areas": [],
            "recommendation": (
                "No interview answers were submitted."
            ),
        }

    valid_scores = []
    all_strengths: List[str] = []
    all_missing_points: List[str] = []

    for answer_data in interview_answers:
        evaluation = answer_data.get("evaluation", {})

        try:
            score = float(
                evaluation.get("score", 0)
            )
        except (TypeError, ValueError):
            score = 0.0

        valid_scores.append(
            max(0.0, min(score, 100.0))
        )

        all_strengths.extend(
            evaluation.get("strengths", [])
        )
        all_missing_points.extend(
            evaluation.get("missing_points", [])
        )

    final_score = round(
        sum(valid_scores) / len(valid_scores),
        2,
    )

    if final_score >= qualification_score:
        status = "Qualified"
        recommendation = (
            "Candidate demonstrated sufficient technical understanding "
            "and can proceed to the next recruitment round."
        )
    else:
        status = "Not Qualified"
        recommendation = (
            "Candidate requires stronger technical understanding "
            "before proceeding to the next recruitment round."
        )

    unique_strengths = list(
        dict.fromkeys(
            item for item in all_strengths if item
        )
    )
    unique_improvements = list(
        dict.fromkeys(
            item for item in all_missing_points if item
        )
    )

    return {
        "score": final_score,
        "status": status,
        "answered_questions": len(interview_answers),
        "total_questions": len(interview_answers),
        "strengths": unique_strengths[:8],
        "improvement_areas": unique_improvements[:8],
        "recommendation": recommendation,
    }
