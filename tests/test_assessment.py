from companyscope.assessment import assess_attention


def test_attention_assessment_explains_each_rule():
    features = {
        "company_status": "active",
        "accounts_overdue": True,
        "confirmation_statement_overdue": True,
        "days_since_latest_filing": 800,
        "has_insolvency_history_link": True,
    }

    assessment = assess_attention(features)

    assert assessment.score == 65
    assert assessment.band == "high-attention"
    assert len(assessment.reasons) == 4
    assert "not a credit rating" in assessment.disclaimer


def test_routine_company_has_no_hidden_penalty():
    features = {
        "company_status": "active",
        "accounts_overdue": False,
        "confirmation_statement_overdue": False,
        "days_since_latest_filing": 120,
        "has_insolvency_history_link": False,
    }

    assessment = assess_attention(features)

    assert assessment.score == 0
    assert assessment.band == "routine"
    assert assessment.reasons == ["No configured attention flags were triggered"]
