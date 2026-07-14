def calculate_mcq_score(mcqs, user_answers):
    total = len(mcqs)
    correct = 0

    for i, q in enumerate(mcqs):
        selected = user_answers.get(i)
        actual = q.get("correct_answer")

        if selected == actual:
            correct += 1

    percentage = round((correct / total) * 100, 2) if total > 0 else 0

    result = {
        "total_questions": total,
        "correct_answers": correct,
        "wrong_answers": total - correct,
        "score_percentage": percentage,
        "qualified_for_voice_round": percentage >= 60
    }

    return result