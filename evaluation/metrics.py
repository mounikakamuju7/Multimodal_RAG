def simple_relevance_score(answer, context):
    score = 0
    for chunk in context:
        if chunk[:50] in answer:
            score += 1
    return score / len(context)