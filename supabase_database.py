import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from supabase import Client, create_client


load_dotenv()


def _get_secret(name: str) -> Optional[str]:
    """
    Read credentials from Streamlit secrets first,
    then fall back to environment variables.
    """

    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass

    return os.getenv(name)


def get_supabase_client() -> Client:
    """
    Create the Supabase client.
    """

    supabase_url = _get_secret("SUPABASE_URL")
    supabase_key = _get_secret("SUPABASE_KEY")

    if not supabase_url:
        raise ValueError(
            "SUPABASE_URL was not found."
        )

    if not supabase_key:
        raise ValueError(
            "SUPABASE_KEY was not found."
        )

    return create_client(
        supabase_url,
        supabase_key,
    )


def _safe_number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def create_candidate_record(
    candidate_name: str,
    candidate_email: str,
    resume_data: Dict[str, Any],
    jd_data: Dict[str, Any],
    match_result: Dict[str, Any],
    eligible_for_mcq: bool,
) -> Dict[str, Any]:
    """
    Create the candidate record after resume screening.
    """

    client = get_supabase_client()

    match_score = _safe_number(
        match_result.get("overall_match_score")
    )

    record = {
        "candidate_name": (
            candidate_name.strip()
            or "Unknown Candidate"
        ),
        "candidate_email": (
            candidate_email.strip()
            or None
        ),
        "resume_match_score": match_score,
        "resume_match_result": match_result,
        "resume_data": resume_data,
        "jd_data": jd_data,
        "eligible_for_mcq": eligible_for_mcq,
        "final_status": (
            "MCQ Qualified"
            if eligible_for_mcq
            else "Rejected After Screening"
        ),
        "updated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    response = (
        client.table("candidate_interviews")
        .insert(record)
        .execute()
    )

    if not response.data:
        raise RuntimeError(
            "Supabase did not return the inserted candidate record."
        )

    return response.data[0]


def update_mcq_result(
    record_id: int,
    mcq_result: Dict[str, Any],
    eligible_for_voice: bool,
) -> Dict[str, Any]:
    """
    Save the MCQ result.
    """

    client = get_supabase_client()

    score = _safe_number(
        mcq_result.get("score_percentage")
    )

    correct_answers = int(
        mcq_result.get(
            "correct_answers",
            mcq_result.get("correct_count", 0),
        )
        or 0
    )

    total_questions = int(
        mcq_result.get("total_questions", 0)
        or 0
    )

    update_data = {
        "mcq_score": score,
        "mcq_correct_answers": correct_answers,
        "mcq_total_questions": total_questions,
        "mcq_result": mcq_result,
        "mcq_completed": True,
        "eligible_for_voice": eligible_for_voice,
        "final_status": (
            "Voice Interview Qualified"
            if eligible_for_voice
            else "Rejected After MCQ"
        ),
        "updated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    response = (
        client.table("candidate_interviews")
        .update(update_data)
        .eq("id", record_id)
        .execute()
    )

    if not response.data:
        raise RuntimeError(
            "MCQ result was not updated in Supabase."
        )

    return response.data[0]


def update_voice_result(
    record_id: int,
    voice_answers: List[Dict[str, Any]],
    voice_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Save and lock the voice interview result.
    """

    client = get_supabase_client()

    score = _safe_number(
        voice_result.get("score")
    )

    status = str(
        voice_result.get(
            "status",
            "Not Qualified",
        )
    )

    update_data = {
        "voice_score": score,
        "voice_status": status,
        "voice_answers": voice_answers,
        "voice_result": voice_result,
        "voice_completed": True,
        "final_status": (
            "Selected for HR Round"
            if status == "Qualified"
            else "Rejected After Voice Interview"
        ),
        "updated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    response = (
        client.table("candidate_interviews")
        .update(update_data)
        .eq("id", record_id)
        .eq("voice_completed", False)
        .execute()
    )

    if not response.data:
        raise RuntimeError(
            "Voice interview is already completed or "
            "the database record was not found."
        )

    return response.data[0]


def get_candidate_record(
    record_id: int,
) -> Optional[Dict[str, Any]]:
    """
    Read one candidate record.
    """

    client = get_supabase_client()

    response = (
        client.table("candidate_interviews")
        .select("*")
        .eq("id", record_id)
        .limit(1)
        .execute()
    )

    if not response.data:
        return None

    return response.data[0]


def get_all_candidates() -> List[Dict[str, Any]]:
    """
    Return all candidate records for dashboard and history.
    """

    client = get_supabase_client()

    response = (
        client.table("candidate_interviews")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )

    return response.data or []