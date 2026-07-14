import os
import time

import pandas as pd
import streamlit as st

from resume_parser import extract_resume_text
from ai_engine import analyze_resume
from jd_parser import analyze_jd
from matching_engine import calculate_match
from mcq_engine import generate_mcqs
from mcq_scoring import calculate_mcq_score
from voice_interview_engine import (
    generate_voice_interview_questions,
    transcribe_audio,
    evaluate_voice_answer,
    calculate_final_voice_result,
)
from supabase_database import (
    create_candidate_record,
    update_mcq_result,
    update_voice_result,
    get_candidate_record,
    get_all_candidates,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Recruitment Platform",
    page_icon="🤖",
    layout="wide"
)

os.makedirs("uploads", exist_ok=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

DEFAULT_SESSION_VALUES = {
    "candidate_name": "",
    "candidate_email": "",
    "resume_text": "",
    "resume_data": None,
    "resume_analysis": None,
    "jd_data": None,
    "match_result": None,
    "eligible_for_mcq": False,
    "mcqs": None,
    "mcq_submitted": False,
    "mcq_result": None,
    "eligible_for_voice": False,
    "voice_questions": [],
    "voice_answers": [],
    "voice_question_index": 0,
    "voice_interview_started": False,
    "voice_interview_completed": False,
    "voice_final_result": None,
    "voice_interview_start_time": None,
    "candidate_record_id": None,
    "voice_result_saved": False
}

for key, default_value in DEFAULT_SESSION_VALUES.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_uploaded_file(uploaded_file, prefix=""):
    """
    Save an uploaded Streamlit file inside the uploads folder.
    """
    if uploaded_file is None:
        return None

    safe_filename = os.path.basename(uploaded_file.name)
    filename = f"{prefix}{safe_filename}"
    file_path = os.path.join("uploads", filename)

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return file_path


def reset_after_new_screening():
    """
    Remove previous MCQ and voice-round results whenever
    a new screening is performed.
    """
    st.session_state["mcqs"] = None
    st.session_state["mcq_submitted"] = False
    st.session_state["mcq_result"] = None
    st.session_state["eligible_for_voice"] = False
    st.session_state["voice_questions"] = []
    st.session_state["voice_answers"] = []
    st.session_state["voice_question_index"] = 0
    st.session_state["voice_interview_started"] = False
    st.session_state["voice_interview_completed"] = False
    st.session_state["voice_final_result"] = None
    st.session_state["voice_interview_start_time"] = None
    st.session_state["candidate_record_id"] = None
    st.session_state["voice_result_saved"] = False


def reset_mcq_test():
    """
    Clear the current MCQ test and previously selected answers.
    """
    st.session_state["mcqs"] = None
    st.session_state["mcq_submitted"] = False
    st.session_state["mcq_result"] = None
    st.session_state["eligible_for_voice"] = False
    st.session_state["voice_questions"] = []
    st.session_state["voice_answers"] = []
    st.session_state["voice_question_index"] = 0
    st.session_state["voice_interview_started"] = False
    st.session_state["voice_interview_completed"] = False
    st.session_state["voice_final_result"] = None
    st.session_state["voice_interview_start_time"] = None
    st.session_state["voice_result_saved"] = False

    keys_to_delete = [
        key for key in st.session_state
        if key.startswith("mcq_answer_")
    ]

    for key in keys_to_delete:
        del st.session_state[key]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🤖 AI Recruitment Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Candidate Screening",
        "MCQ Round",
        "AI Voice Interview",
        #"HR Dashboard",
        #"Interview History"
    ]
)

st.sidebar.divider()

if st.session_state.get("match_result"):
    match_score = st.session_state["match_result"].get(
        "overall_match_score",
        0
    )
    st.sidebar.metric("Resume Match", f"{match_score}%")

if st.session_state.get("mcq_result"):
    mcq_score = st.session_state["mcq_result"].get(
        "score_percentage",
        0
    )
    st.sidebar.metric("MCQ Score", f"{mcq_score}%")


# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":
    st.title("🤖 AI Recruitment Platform")

    st.subheader(
        "Smart Resume Screening, JD Matching, Technical MCQ "
        "Assessment and AI Voice Interview"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Resume Parser", "Ready")
    col2.metric("AI Analyzer", "Ready")
    col3.metric("JD Matching", "Ready")
    col4.metric("Technical MCQ", "Ready")

    st.divider()

    st.markdown(
        """
        ### Recruitment Process

        **Round 1 — Candidate Screening**

        Upload the candidate's resume and job description. The AI extracts
        structured information and calculates the resume-to-JD match score.

        **Round 2 — Technical MCQ Assessment**

        Candidates obtaining at least **50% resume match** can attempt
        10 technical MCQs generated from their resume skills and projects.

        **Round 3 — AI Voice Interview**

        Candidates obtaining at least **60% in the MCQ test** qualify for
        the AI voice interview.
        """
    )


# ============================================================
# CANDIDATE SCREENING PAGE
# ============================================================

elif page == "Candidate Screening":
    st.title("Candidate Screening")

    candidate_name = st.text_input(
        "Candidate Name",
        value=st.session_state.get("candidate_name", "")
    )

    candidate_email = st.text_input(
        "Candidate Email",
        value=st.session_state.get("candidate_email", "")
    )

    col1, col2 = st.columns(2)

    with col1:
        uploaded_resume = st.file_uploader(
            "Upload Resume",
            type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
            key="resume_uploader"
        )

    with col2:
        jd_file = st.file_uploader(
            "Upload Job Description",
            type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
            key="jd_uploader"
        )

    # --------------------------------------------------------
    # Extract resume text
    # --------------------------------------------------------

    if uploaded_resume is not None:
    try:
        import os
        import tempfile

        file_extension = os.path.splitext(uploaded_resume.name)[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        ) as temp_file:
            temp_file.write(uploaded_resume.getbuffer())
            resume_path = temp_file.name

        resume_text = extract_resume_text(resume_path)

        if os.path.exists(resume_path):
            os.remove(resume_path)

        if resume_text and resume_text.strip():
            st.session_state["resume_text"] = resume_text
            st.success("Resume uploaded and text extracted successfully.")
        else:
            st.session_state["resume_text"] = ""
            st.error("No readable text could be extracted from the resume.")

    except Exception as error:
        st.session_state["resume_text"] = ""
        st.error(f"Resume extraction failed: {error}")

if st.session_state.get("resume_text"):
    with st.expander("View extracted resume text"):
        st.text_area(
            "Extracted Resume",
            value=st.session_state["resume_text"],
            height=300,
            disabled=True,
            label_visibility="collapsed"
        )

    # --------------------------------------------------------
    # Buttons
    # --------------------------------------------------------

    button_col1, button_col2 = st.columns(2)

    with button_col1:
        analyze_resume_button = st.button(
            "Analyze Resume",
            use_container_width=True
        )

    with button_col2:
        start_screening_button = st.button(
            "Start Screening",
            type="primary",
            use_container_width=True
        )

    # --------------------------------------------------------
    # Resume analysis only
    # --------------------------------------------------------

    if analyze_resume_button:
        resume_text = st.session_state.get("resume_text", "")

        if uploaded_resume is None and not resume_text:
            st.warning("Please upload a resume.")

        elif not resume_text:
            st.warning("Resume text could not be extracted.")

        else:
            try:
                with st.spinner("AI is analyzing the resume..."):
                    resume_analysis, structured_data = analyze_resume(
                        resume_text
                    )

                st.session_state["resume_analysis"] = resume_analysis
                st.session_state["resume_data"] = structured_data

                st.success("Resume analysis completed successfully.")

            except Exception as error:
                st.error(f"Resume analysis failed: {error}")

    # --------------------------------------------------------
    # Full candidate screening
    # --------------------------------------------------------

    if start_screening_button:
        resume_text = st.session_state.get("resume_text", "")

        if not candidate_name.strip():
            st.warning("Please enter the candidate name.")

        elif uploaded_resume is None and not resume_text:
            st.warning("Please upload the candidate's resume.")

        elif jd_file is None:
            st.warning("Please upload the job description.")

        elif not resume_text:
            st.warning("Resume text could not be extracted.")

        else:
            try:
                reset_after_new_screening()

                st.session_state["candidate_name"] = candidate_name.strip()
                st.session_state["candidate_email"] = candidate_email.strip()

                with st.spinner("Analyzing the candidate's resume..."):
                    resume_analysis, resume_data = analyze_resume(
                        resume_text
                    )

                with st.spinner("Analyzing the job description..."):
                    jd_path = save_uploaded_file(
                        jd_file,
                        prefix="jd_"
                    )

                    # Use whichever input format your jd_parser accepts.
                    # First try file path. If that fails, try uploaded file.
                    try:
                        jd_data = analyze_jd(jd_path)
                    except (TypeError, AttributeError):
                        jd_data = analyze_jd(jd_file)

                if not isinstance(resume_data, dict):
                    raise ValueError(
                        "Resume analysis did not return structured data."
                    )

                if not isinstance(jd_data, dict):
                    raise ValueError(
                        "JD analysis did not return structured data."
                    )

                with st.spinner("Calculating resume-to-JD match..."):
                    match_result = calculate_match(
                        resume_data,
                        jd_data
                    )

                if not isinstance(match_result, dict):
                    raise ValueError(
                        "Matching engine did not return a valid result."
                    )

                match_score = float(
                    match_result.get("overall_match_score", 0) or 0
                )

                st.session_state["resume_analysis"] = resume_analysis
                st.session_state["resume_data"] = resume_data
                st.session_state["jd_data"] = jd_data
                st.session_state["match_result"] = match_result
                st.session_state["eligible_for_mcq"] = match_score >= 50

                with st.spinner("Saving screening result to Supabase..."):
                    candidate_record = create_candidate_record(
                        candidate_name=st.session_state["candidate_name"],
                        candidate_email=st.session_state["candidate_email"],
                        resume_data=resume_data,
                        jd_data=jd_data,
                        match_result=match_result,
                        eligible_for_mcq=st.session_state[
                            "eligible_for_mcq"
                        ],
                    )

                st.session_state["candidate_record_id"] = (
                    candidate_record["id"]
                )

                st.success(
                    "Screening completed and saved successfully."
                )

            except ValueError as error:
                st.error(str(error))

            except RuntimeError as error:
                st.error(str(error))

            except Exception as error:
                st.error(f"Candidate screening failed: {error}")

    # --------------------------------------------------------
    # Display stored analysis
    # --------------------------------------------------------

    if st.session_state.get("resume_analysis"):
        st.divider()
        st.subheader("AI Resume Analysis")
        st.write(st.session_state["resume_analysis"])

    if st.session_state.get("match_result"):
        st.divider()

        resume_data = st.session_state.get("resume_data", {})
        jd_data = st.session_state.get("jd_data", {})
        match_result = st.session_state.get("match_result", {})

        st.subheader("Candidate Details")

        details_col1, details_col2 = st.columns(2)

        details_col1.write(
            f"**Name:** {st.session_state.get('candidate_name', '')}"
        )

        details_col2.write(
            f"**Email:** "
            f"{st.session_state.get('candidate_email', '') or 'Not provided'}"
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            with st.expander("Resume Extracted Data", expanded=False):
                st.json(resume_data)

        with result_col2:
            with st.expander("JD Extracted Data", expanded=False):
                st.json(jd_data)

        st.subheader("Resume vs JD Match Result")
        st.json(match_result)

        match_score = float(
            match_result.get("overall_match_score", 0) or 0
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        metric_col1.metric(
            "Overall Match Score",
            f"{match_score:.1f}%"
        )

        metric_col2.metric(
            "Hiring Recommendation",
            str(
                match_result.get(
                    "hiring_recommendation",
                    "Not available"
                )
            )
        )

        metric_col3.metric(
            "Interview Probability",
            str(
                match_result.get(
                    "interview_probability",
                    "Not available"
                )
            )
        )

        if match_score >= 50:
            st.session_state["eligible_for_mcq"] = True

            st.success(
                "Candidate qualified for the Technical MCQ Round. "
                "Open the MCQ Round page from the sidebar."
            )
        else:
            st.session_state["eligible_for_mcq"] = False

            st.error(
                "Candidate did not achieve the required 50% match score "
                "and is not eligible for the MCQ round."
            )


# ============================================================
# MCQ ROUND PAGE
# ============================================================

elif page == "MCQ Round":
    st.title("Technical MCQ Round")

    if not st.session_state.get("match_result"):
        st.warning(
            "Please complete candidate screening before starting "
            "the MCQ round."
        )

    elif not st.session_state.get("eligible_for_mcq", False):
        match_score = st.session_state["match_result"].get(
            "overall_match_score",
            0
        )

        st.error(
            f"Candidate match score is {match_score}%. "
            "A minimum score of 50% is required."
        )

    else:
        st.success(
            f"{st.session_state.get('candidate_name', 'Candidate')} "
            "is eligible for the technical MCQ round."
        )

        match_score = st.session_state["match_result"].get(
            "overall_match_score",
            0
        )

        info_col1, info_col2, info_col3 = st.columns(3)

        info_col1.metric("Resume Match", f"{match_score}%")
        info_col2.metric("Number of Questions", "10")
        info_col3.metric("Qualification Score", "60%")

        st.info(
            "The questions will be based only on technical skills and "
            "project technologies extracted from the candidate's resume."
        )

        generate_col, reset_col = st.columns(2)

        with generate_col:
            generate_button = st.button(
                "Generate 10 Technical MCQs",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.get("mcqs") is not None
            )

        with reset_col:
            reset_button = st.button(
                "Generate New Test",
                use_container_width=True,
                disabled=st.session_state.get("mcqs") is None
            )

        if reset_button:
            reset_mcq_test()
            st.rerun()

        # ----------------------------------------------------
        # Generate MCQs
        # ----------------------------------------------------

        if generate_button:
            resume_data = st.session_state.get("resume_data")
            jd_data = st.session_state.get("jd_data")
            match_result = st.session_state.get("match_result")

            if not isinstance(resume_data, dict):
                st.error(
                    "Resume data is missing. Please run candidate "
                    "screening again."
                )

            elif not isinstance(jd_data, dict):
                st.error(
                    "JD data is missing. Please run candidate "
                    "screening again."
                )

            else:
                try:
                    with st.spinner(
                        "Generating and validating 10 technical MCQs..."
                    ):
                        mcq_questions = generate_mcqs(
                            candidate_data=resume_data,
                            jd_data=jd_data,
                            match_result=match_result,
                            total_questions=10
                        )

                    if not isinstance(mcq_questions, list):
                        raise RuntimeError(
                            "MCQ engine did not return a question list."
                        )

                    if len(mcq_questions) != 10:
                        raise RuntimeError(
                            f"Expected 10 questions, but received "
                            f"{len(mcq_questions)}."
                        )

                    st.session_state["mcqs"] = mcq_questions
                    st.session_state["mcq_submitted"] = False
                    st.session_state["mcq_result"] = None
                    st.session_state["eligible_for_voice"] = False

                    st.success(
                        "10 technical MCQs generated successfully."
                    )

                    st.rerun()

                except ValueError as error:
                    st.error(str(error))

                except RuntimeError as error:
                    st.error(str(error))

                except Exception as error:
                    st.error(f"MCQ generation failed: {error}")

        # ----------------------------------------------------
        # Display MCQ test
        # ----------------------------------------------------

        mcqs = st.session_state.get("mcqs")

        if mcqs:
            st.divider()
            st.subheader("Answer all questions")

            if not st.session_state.get("mcq_submitted", False):
                with st.form("technical_mcq_form"):
                    for index, question_data in enumerate(mcqs):
                        question = question_data.get(
                            "question",
                            "Question unavailable"
                        )

                        options = question_data.get("options", [])
                        category = question_data.get(
                            "category",
                            "Technical"
                        )
                        difficulty = question_data.get(
                            "difficulty",
                            "Intermediate"
                        )

                        st.markdown(
                            f"### Question {index + 1}"
                        )

                        st.caption(
                            f"Category: {category} | "
                            f"Difficulty: {difficulty}"
                        )

                        st.write(question)

                        if len(options) == 4:
                            st.radio(
                                "Choose one answer",
                                options,
                                index=None,
                                key=f"mcq_answer_{index}"
                            )
                        else:
                            st.error(
                                "This question does not contain "
                                "four valid options."
                            )

                        st.divider()

                    submit_mcq_button = st.form_submit_button(
                        "Submit MCQ Test",
                        type="primary",
                        use_container_width=True
                    )

                if submit_mcq_button:
                    unanswered_questions = []
                    user_answers = {}

                    for index in range(len(mcqs)):
                        answer_key = f"mcq_answer_{index}"
                        selected_answer = st.session_state.get(answer_key)

                        if selected_answer is None:
                            unanswered_questions.append(index + 1)
                        else:
                            user_answers[index] = selected_answer

                    if unanswered_questions:
                        question_numbers = ", ".join(
                            map(str, unanswered_questions)
                        )

                        st.warning(
                            "Please answer all questions. "
                            f"Unanswered question numbers: "
                            f"{question_numbers}"
                        )

                    else:
                        try:
                            result = calculate_mcq_score(
                                mcqs,
                                user_answers
                            )

                            if not isinstance(result, dict):
                                raise ValueError(
                                    "MCQ scoring did not return "
                                    "a valid result."
                                )

                            score_percentage = float(
                                result.get("score_percentage", 0) or 0
                            )

                            # Ensure qualification uses the required
                            # score of 60%, even if older scoring code
                            # uses another threshold.
                            qualified = score_percentage >= 60

                            result["qualified_for_voice_round"] = qualified

                            st.session_state["mcq_result"] = result
                            st.session_state["mcq_submitted"] = True
                            st.session_state["eligible_for_voice"] = qualified

                            record_id = st.session_state.get(
                                "candidate_record_id"
                            )

                            if record_id:
                                with st.spinner(
                                    "Saving MCQ result to Supabase..."
                                ):
                                    update_mcq_result(
                                        record_id=record_id,
                                        mcq_result=result,
                                        eligible_for_voice=qualified,
                                    )

                            st.rerun()

                        except Exception as error:
                            st.error(
                                f"MCQ score calculation failed: {error}"
                            )

            # ------------------------------------------------
            # Display MCQ result
            # ------------------------------------------------

            if st.session_state.get("mcq_submitted", False):
                result = st.session_state.get("mcq_result", {})

                score_percentage = float(
                    result.get("score_percentage", 0) or 0
                )

                correct_answers = result.get(
                    "correct_answers",
                    result.get("correct_count", 0)
                )

                total_questions = result.get(
                    "total_questions",
                    len(mcqs)
                )

                st.subheader("MCQ Result")

                result_col1, result_col2, result_col3 = st.columns(3)

                result_col1.metric(
                    "MCQ Score",
                    f"{score_percentage:.1f}%"
                )

                result_col2.metric(
                    "Correct Answers",
                    f"{correct_answers}/{total_questions}"
                )

                result_col3.metric(
                    "Status",
                    (
                        "Qualified"
                        if score_percentage >= 60
                        else "Not Qualified"
                    )
                )

                with st.expander("View complete scoring result"):
                    st.json(result)

                if score_percentage >= 60:
                    st.session_state["eligible_for_voice"] = True

                    st.success(
                        "Candidate qualified for the AI Voice "
                        "Interview Round."
                    )
                else:
                    st.session_state["eligible_for_voice"] = False

                    st.error(
                        "Candidate did not achieve the required "
                        "60% MCQ score."
                    )


# ============================================================
# AI VOICE INTERVIEW PAGE
# ============================================================

elif page == "AI Voice Interview":
    st.title("🎙️ AI Voice Interview Round")

    if not st.session_state.get("mcq_submitted", False):
        st.warning(
            "Please complete and submit the technical MCQ round first."
        )
        st.stop()

    if not st.session_state.get("eligible_for_voice", False):
        score = 0

        if st.session_state.get("mcq_result"):
            score = st.session_state["mcq_result"].get(
                "score_percentage",
                0
            )

        st.error(
            f"Candidate scored {score}% in the MCQ test. "
            "A minimum score of 60% is required for the voice round."
        )
        st.stop()

    st.success(
        f"{st.session_state.get('candidate_name', 'Candidate')} "
        "qualified for the AI Voice Interview."
    )

    resume_data = st.session_state.get("resume_data")
    jd_data = st.session_state.get("jd_data", {})
    match_result = st.session_state.get("match_result", {})

    if not isinstance(resume_data, dict):
        st.error(
            "Resume data is missing. Please complete candidate "
            "screening again."
        )
        st.stop()

    record_id = st.session_state.get("candidate_record_id")

    if record_id:
        try:
            stored_candidate = get_candidate_record(record_id)

            if (
                stored_candidate
                and stored_candidate.get("voice_completed")
            ):
                st.session_state["voice_interview_started"] = True
                st.session_state["voice_interview_completed"] = True
                st.session_state["voice_final_result"] = (
                    stored_candidate.get("voice_result") or {}
                )
                st.session_state["voice_answers"] = (
                    stored_candidate.get("voice_answers") or []
                )
                st.session_state["voice_result_saved"] = True

        except Exception as error:
            st.warning(
                f"Could not verify interview lock from Supabase: {error}"
            )

    # --------------------------------------------------------
    # Start interview
    # --------------------------------------------------------

    if not st.session_state.get("voice_interview_started", False):
        st.info(
            "The interview contains 5 technical and project-based "
            "questions. Record one answer for each question."
        )

        if st.button(
            "Start Voice Interview",
            type="primary",
            use_container_width=True
        ):
            try:
                with st.spinner(
                    "Generating technical voice interview questions..."
                ):
                    questions = generate_voice_interview_questions(
                        candidate_data=resume_data,
                        jd_data=jd_data,
                        match_result=match_result,
                        total_questions=5
                    )

                if not isinstance(questions, list) or not questions:
                    raise RuntimeError(
                        "Voice interview engine did not return questions."
                    )

                st.session_state["voice_questions"] = questions
                st.session_state["voice_answers"] = []
                st.session_state["voice_question_index"] = 0
                st.session_state["voice_interview_started"] = True
                st.session_state["voice_interview_completed"] = False
                st.session_state["voice_final_result"] = None
                st.session_state["voice_interview_start_time"] = time.time()

                st.rerun()

            except Exception as error:
                st.error(
                    f"Voice interview generation failed: {error}"
                )

        st.stop()

    # --------------------------------------------------------
    # Display completed result
    # --------------------------------------------------------

    if st.session_state.get("voice_interview_completed", False):
        result = st.session_state.get("voice_final_result")

        st.success("Voice interview completed successfully.")

        if not isinstance(result, dict):
            st.error("Final voice interview result is unavailable.")
            st.stop()

        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric(
            "Voice Interview Score",
            f"{float(result.get('score', 0)):.1f}%"
        )

        result_col2.metric(
            "Questions Answered",
            result.get("answered_questions", 0)
        )

        result_col3.metric(
            "Status",
            result.get("status", "Not Available")
        )

        if result.get("status") == "Qualified":
            st.success(
                "Candidate qualified for the next recruitment round."
            )
        else:
            st.error(
                "Candidate did not qualify for the next recruitment round."
            )

        st.subheader("Overall Recommendation")
        st.write(
            result.get(
                "recommendation",
                "No recommendation is available."
            )
        )

        strengths = result.get("strengths", [])

        if strengths:
            st.subheader("Strengths")

            for strength in strengths:
                st.write(f"✅ {strength}")

        improvement_areas = result.get("improvement_areas", [])

        if improvement_areas:
            st.subheader("Areas for Improvement")

            for area in improvement_areas:
                st.write(f"⚠️ {area}")

        st.divider()
        st.subheader("Question-wise Evaluation")

        for index, answer_data in enumerate(
            st.session_state.get("voice_answers", []),
            start=1
        ):
            evaluation = answer_data.get("evaluation", {})

            with st.expander(
                f"Question {index} — "
                f"Score: {evaluation.get('score', 0)}/100"
            ):
                st.markdown(
                    f"**Question:** {answer_data.get('question', '')}"
                )

                st.markdown("**Candidate Answer:**")
                st.write(answer_data.get("transcript", ""))

                st.markdown("**AI Feedback:**")
                st.write(
                    evaluation.get(
                        "feedback",
                        "No feedback available."
                    )
                )

                score_col1, score_col2 = st.columns(2)

                score_col1.metric(
                    "Technical Correctness",
                    f"{evaluation.get('technical_correctness', 0)}/50"
                )

                score_col1.metric(
                    "Practical Understanding",
                    f"{evaluation.get('practical_understanding', 0)}/20"
                )

                score_col2.metric(
                    "Explanation Quality",
                    f"{evaluation.get('explanation_quality', 0)}/20"
                )

                score_col2.metric(
                    "Clarity and Relevance",
                    f"{evaluation.get('clarity_relevance', 0)}/10"
                )

                ideal_answer = evaluation.get(
                    "ideal_answer_summary",
                    ""
                )

                if ideal_answer:
                    st.markdown("**Ideal Answer Summary:**")
                    st.write(ideal_answer)

        st.info(
            "This interview attempt has been submitted and locked. "
            "The candidate cannot retake the interview."
        )

        st.stop()

    # --------------------------------------------------------
    # Display current question
    # --------------------------------------------------------

    questions = st.session_state.get("voice_questions", [])
    current_index = st.session_state.get("voice_question_index", 0)
    total_questions = len(questions)

    if not questions:
        st.error(
            "Voice questions are unavailable. Restart the interview."
        )
        st.stop()

    if current_index >= total_questions:
        final_result = calculate_final_voice_result(
            interview_answers=st.session_state.get(
                "voice_answers",
                []
            ),
            qualification_score=60
        )

        record_id = st.session_state.get("candidate_record_id")

        if record_id and not st.session_state.get(
            "voice_result_saved",
            False
        ):
            with st.spinner(
                "Saving and locking voice interview result..."
            ):
                update_voice_result(
                    record_id=record_id,
                    voice_answers=st.session_state.get(
                        "voice_answers",
                        []
                    ),
                    voice_result=final_result,
                )

            st.session_state["voice_result_saved"] = True

        st.session_state["voice_final_result"] = final_result
        st.session_state["voice_interview_completed"] = True
        st.rerun()

    current_question = questions[current_index]

    progress_value = (
        current_index / total_questions
        if total_questions
        else 0
    )

    st.progress(
        progress_value,
        text=(
            f"Question {current_index + 1} "
            f"of {total_questions}"
        )
    )

    start_time = st.session_state.get(
        "voice_interview_start_time"
    )

    if start_time:
        elapsed_seconds = int(time.time() - start_time)
        minutes = elapsed_seconds // 60
        seconds = elapsed_seconds % 60

        st.caption(
            f"Elapsed interview time: "
            f"{minutes:02d}:{seconds:02d}"
        )

    st.subheader(f"Question {current_index + 1}")
    st.info(current_question.get("question", ""))

    question_col1, question_col2 = st.columns(2)

    question_col1.caption(
        f"Category: "
        f"{current_question.get('category', 'Technical')}"
    )

    question_col2.caption(
        f"Difficulty: "
        f"{current_question.get('difficulty', 'Intermediate')}"
    )

    st.write(
        "Click the microphone, record your answer, review the audio, "
        "and click **Submit Answer**."
    )

    recorded_audio = st.audio_input(
        "Record your answer",
        sample_rate=16000,
        key=f"voice_audio_{current_index}"
    )

    if recorded_audio is not None:
        st.audio(recorded_audio)

    if st.button(
        "Submit Answer",
        type="primary",
        use_container_width=True,
        disabled=recorded_audio is None
    ):
        if recorded_audio is None:
            st.warning("Please record your answer first.")

        else:
            try:
                with st.spinner(
                    "Converting speech to text..."
                ):
                    transcript = transcribe_audio(recorded_audio)

                with st.spinner(
                    "Evaluating the technical answer..."
                ):
                    evaluation = evaluate_voice_answer(
                        question_data=current_question,
                        candidate_answer=transcript,
                        candidate_data=resume_data,
                        jd_data=jd_data
                    )

                answer_record = {
                    "question_number": current_index + 1,
                    "question": current_question.get(
                        "question",
                        ""
                    ),
                    "category": current_question.get(
                        "category",
                        "Technical"
                    ),
                    "difficulty": current_question.get(
                        "difficulty",
                        "Intermediate"
                    ),
                    "transcript": transcript,
                    "evaluation": evaluation
                }

                st.session_state["voice_answers"].append(
                    answer_record
                )

                st.session_state["voice_question_index"] += 1
                st.rerun()

            except Exception as error:
                st.error(
                    f"Voice answer processing failed: {error}"
                )


# ============================================================
# HR DASHBOARD PAGE
# ============================================================

elif page == "HR Dashboard":
    st.title("📊 HR Dashboard")

    try:
        candidates = get_all_candidates()
    except Exception as error:
        st.error(f"Could not load dashboard data from Supabase: {error}")
        st.stop()

    if not candidates:
        st.info("No candidate records are available yet.")
        st.stop()

    dashboard_df = pd.DataFrame(candidates)

    for column in ["resume_match_score", "mcq_score", "voice_score"]:
        if column in dashboard_df.columns:
            dashboard_df[column] = pd.to_numeric(
                dashboard_df[column], errors="coerce"
            )

    total_candidates = len(dashboard_df)
    mcq_shortlisted = int(dashboard_df.get("eligible_for_mcq", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    voice_shortlisted = int(dashboard_df.get("eligible_for_voice", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    selected_candidates = int((dashboard_df.get("final_status", pd.Series(dtype=str)) == "Selected for HR Round").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Candidates", total_candidates)
    c2.metric("MCQ Shortlisted", mcq_shortlisted)
    c3.metric("Voice Shortlisted", voice_shortlisted)
    c4.metric("Selected for HR", selected_candidates)

    avg_match = dashboard_df.get("resume_match_score", pd.Series(dtype=float)).mean()
    avg_mcq = dashboard_df.get("mcq_score", pd.Series(dtype=float)).mean()
    avg_voice = dashboard_df.get("voice_score", pd.Series(dtype=float)).mean()

    s1, s2, s3 = st.columns(3)
    s1.metric("Average Resume Match", f"{0 if pd.isna(avg_match) else avg_match:.1f}%")
    s2.metric("Average MCQ Score", f"{0 if pd.isna(avg_mcq) else avg_mcq:.1f}%")
    s3.metric("Average Voice Score", f"{0 if pd.isna(avg_voice) else avg_voice:.1f}%")

    st.divider()

    f1, f2 = st.columns(2)
    candidate_search = f1.text_input("Search candidate", placeholder="Enter name or email")
    status_options = ["All"]
    if "final_status" in dashboard_df.columns:
        status_options += sorted(dashboard_df["final_status"].dropna().astype(str).unique().tolist())
    selected_status = f2.selectbox("Filter by final status", status_options)

    filtered_df = dashboard_df.copy()
    if candidate_search:
        q = candidate_search.casefold()
        names = filtered_df.get("candidate_name", pd.Series("", index=filtered_df.index)).fillna("").astype(str)
        emails = filtered_df.get("candidate_email", pd.Series("", index=filtered_df.index)).fillna("").astype(str)
        filtered_df = filtered_df[names.str.casefold().str.contains(q, regex=False) | emails.str.casefold().str.contains(q, regex=False)]
    if selected_status != "All":
        filtered_df = filtered_df[filtered_df["final_status"] == selected_status]

    ch1, ch2 = st.columns(2)
    with ch1:
        st.subheader("Hiring Funnel")
        funnel_df = pd.DataFrame({
            "Stage": ["Total Candidates", "MCQ Shortlisted", "Voice Shortlisted", "Selected for HR"],
            "Candidates": [total_candidates, mcq_shortlisted, voice_shortlisted, selected_candidates],
        }).set_index("Stage")
        st.bar_chart(funnel_df)
    with ch2:
        st.subheader("Final Status Distribution")
        if "final_status" in dashboard_df.columns:
            status_df = dashboard_df["final_status"].fillna("Unknown").value_counts().rename_axis("Status").to_frame("Candidates")
            st.bar_chart(status_df)

    sc1, sc2 = st.columns(2)
    with sc1:
        st.subheader("Resume and MCQ Scores")
        cols = [c for c in ["resume_match_score", "mcq_score"] if c in dashboard_df.columns]
        if cols:
            temp = dashboard_df[["candidate_name"] + cols].copy()
            temp["candidate_name"] = temp["candidate_name"].fillna("Unknown").astype(str)
            st.bar_chart(temp.set_index("candidate_name"))
    with sc2:
        st.subheader("Voice Interview Scores")
        if "voice_score" in dashboard_df.columns and dashboard_df["voice_score"].notna().any():
            temp = dashboard_df[["candidate_name", "voice_score"]].dropna(subset=["voice_score"]).copy()
            temp["candidate_name"] = temp["candidate_name"].fillna("Unknown").astype(str)
            st.bar_chart(temp.set_index("candidate_name"))
        else:
            st.info("No completed voice interview scores yet.")

    st.divider()
    st.subheader("Candidate Records")
    display_columns = [c for c in ["candidate_name", "candidate_email", "resume_match_score", "mcq_score", "voice_score", "voice_status", "final_status", "created_at"] if c in filtered_df.columns]
    st.dataframe(filtered_df[display_columns], use_container_width=True, hide_index=True)

    st.download_button(
        "Download Dashboard CSV",
        data=filtered_df[display_columns].to_csv(index=False),
        file_name="candidate_interview_dashboard.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if not filtered_df.empty:
        st.subheader("Candidate Detail")
        options = {}
        for _, row in filtered_df.iterrows():
            label = f"{row.get('candidate_name', 'Unknown')} — {row.get('candidate_email', '')}"
            options[label] = row.get("id")
        selected_label = st.selectbox("Select candidate", list(options.keys()))
        selected_id = options[selected_label]
        selected = next((x for x in candidates if x.get("id") == selected_id), None)
        if selected:
            d1, d2, d3 = st.columns(3)
            d1.metric("Resume Match", f"{float(selected.get('resume_match_score') or 0):.1f}%")
            d2.metric("MCQ Score", f"{float(selected.get('mcq_score') or 0):.1f}%")
            d3.metric("Voice Score", f"{float(selected.get('voice_score') or 0):.1f}%")
            st.write(f"**Final Status:** {selected.get('final_status', 'Unknown')}")
            with st.expander("Resume Match Result"):
                st.json(selected.get("resume_match_result", {}))
            with st.expander("MCQ Result"):
                st.json(selected.get("mcq_result", {}))
            with st.expander("Voice Interview Result"):
                st.json(selected.get("voice_result", {}))
            with st.expander("Voice Question-wise Answers"):
                answers = selected.get("voice_answers", []) or []
                if not answers:
                    st.info("No voice interview answers are available.")
                for i, answer in enumerate(answers, start=1):
                    st.markdown(f"**Question {i}:** {answer.get('question', '')}")
                    st.write(f"Answer: {answer.get('transcript', '')}")
                    st.write(f"Score: {answer.get('evaluation', {}).get('score', 0)}/100")
                    st.divider()


# ============================================================
# INTERVIEW HISTORY PAGE
# ============================================================

elif page == "Interview History":
    st.title("🗂️ Interview History")

    try:
        history_records = get_all_candidates()
    except Exception as error:
        st.error(f"Could not load interview history from Supabase: {error}")
        st.stop()

    if not history_records:
        st.info("No interview history is available.")
        st.stop()

    history_df = pd.DataFrame(history_records)
    display_columns = [c for c in ["candidate_name", "candidate_email", "resume_match_score", "mcq_score", "voice_score", "voice_status", "final_status", "created_at"] if c in history_df.columns]

    search_text = st.text_input("Search interview history", placeholder="Candidate name or email")
    filtered_history = history_df.copy()
    if search_text:
        q = search_text.casefold()
        names = filtered_history.get("candidate_name", pd.Series("", index=filtered_history.index)).fillna("").astype(str)
        emails = filtered_history.get("candidate_email", pd.Series("", index=filtered_history.index)).fillna("").astype(str)
        filtered_history = filtered_history[names.str.casefold().str.contains(q, regex=False) | emails.str.casefold().str.contains(q, regex=False)]

    st.dataframe(filtered_history[display_columns], use_container_width=True, hide_index=True)
    st.download_button(
        "Download Interview History CSV",
        data=filtered_history[display_columns].to_csv(index=False),
        file_name="interview_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if not filtered_history.empty:
        candidate_map = {}
        for _, row in filtered_history.iterrows():
            label = f"{row.get('candidate_name', 'Unknown')} — {row.get('candidate_email', '')}"
            candidate_map[label] = row.get("id")
        selected_label = st.selectbox("Open complete interview record", list(candidate_map.keys()))
        selected_id = candidate_map[selected_label]
        selected_record = next((r for r in history_records if r.get("id") == selected_id), None)
        if selected_record:
            st.subheader(selected_record.get("candidate_name", "Candidate"))
            c1, c2, c3 = st.columns(3)
            c1.metric("Resume Match", f"{float(selected_record.get('resume_match_score') or 0):.1f}%")
            c2.metric("MCQ Score", f"{float(selected_record.get('mcq_score') or 0):.1f}%")
            c3.metric("Voice Score", f"{float(selected_record.get('voice_score') or 0):.1f}%")
            st.write(f"**Final Status:** {selected_record.get('final_status', 'Unknown')}")
            with st.expander("Complete Stored Record"):
                st.json(selected_record)
