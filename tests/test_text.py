"""Conversation analysis: catch the funnel, stay quiet on ordinary chat."""

import pytest

from sentinel.text.pipeline import analyse_conversation, band

PIG_BUTCHERING = [
    {"sender": "suspect", "text": "Hi, is this Dr. Linda? My assistant gave me this number."},
    {"sender": "victim", "text": "No, wrong number."},
    {"sender": "suspect", "text": "So sorry to bother you! Must be destiny. Do you have WhatsApp? wa.me/15415550199"},
    {"sender": "suspect", "text": "I was doing my market analysis with my uncle. Guaranteed returns on this DeFi dapp."},
    {"sender": "suspect", "text": "Just a small test deposit activates your VIP node. Send USDT to TJRabPrwbZy45sbavfcjinPJC18kjpRTv8"},
    {"sender": "suspect", "text": "Keep this between us, your bank will not understand. The pool closes in 15 minutes."},
]

ORDINARY = [
    {"sender": "suspect", "text": "hey! how was your weekend"},
    {"sender": "suspect", "text": "I went hiking with my sister saturday, rained the entire time lol"},
    {"sender": "suspect", "text": "are we still on for coffee thursday?"},
    {"sender": "suspect", "text": "cool, see you then"},
]


def test_pig_butchering_funnel_is_critical():
    r = analyse_conversation(PIG_BUTCHERING)
    assert r["risk_band"] == "CRITICAL"
    assert "pig_butchering" in r["identified_playbooks"]


def test_ordinary_chat_stays_minimal():
    """False positives here cost real people real relationships."""
    r = analyse_conversation(ORDINARY)
    assert r["risk_band"] == "MINIMAL"
    assert r["identified_playbooks"] == ["none_matched"]
    assert r["risk_score"] == 0.0


def test_every_finding_quotes_the_text_it_matched():
    """A finding a user cannot check against their own chat is not evidence."""
    r = analyse_conversation(PIG_BUTCHERING)
    assert r["matched_phrases"]
    for m in r["matched_phrases"]:
        assert m["quote"]
        source = PIG_BUTCHERING[m["message_index"]]["text"].lower()
        assert m["quote"].lower() in source


def test_payment_identifiers_are_extracted_for_reporting():
    r = analyse_conversation(PIG_BUTCHERING)
    ids = r["identifiers"]
    assert ids["tron_address"] == ["TJRabPrwbZy45sbavfcjinPJC18kjpRTv8"]
    assert ids["whatsapp_link"] == ["wa.me/15415550199"]


def test_escalation_velocity_notices_an_early_ask():
    r = analyse_conversation(PIG_BUTCHERING)
    assert r["escalation"]["early"] is True
    assert r["escalation"]["first_ask_at_message"] == 2


def test_same_ask_much_later_scores_lower():
    """Timing is the point: a request on day ten is not a request on message two."""
    filler = [{"sender": "suspect", "text": f"good morning, hope you slept well {i}"} for i in range(12)]
    late = filler + [{"sender": "suspect", "text": "shall we move to telegram?"}]
    early = [{"sender": "suspect", "text": "shall we move to telegram?"}] + filler
    assert (
        analyse_conversation(late)["escalation"]["score"]
        < analyse_conversation(early)["escalation"]["score"]
    )


def test_repeating_one_phrase_does_not_manufacture_a_critical_score():
    """Saturation guard: ten copies of one hit must not equal a real funnel."""
    spam = [{"sender": "suspect", "text": "telegram telegram telegram"} for _ in range(10)]
    r = analyse_conversation(spam)
    assert r["risk_band"] != "CRITICAL"


def test_advice_is_specific_to_what_was_found():
    r = analyse_conversation(
        [
            {"sender": "suspect", "text": "This is the fraud department, read back the 6-digit code."},
            {"sender": "suspect", "text": "Install AnyDesk so we can secure your account."},
        ]
    )
    assert "account_takeover" in r["identified_playbooks"]
    assert any("verification code" in a for a in r["what_to_do"])


def test_scoring_method_is_declared_uncalibrated():
    """The text path is a rubric, and must never claim otherwise."""
    r = analyse_conversation(ORDINARY)
    assert "not statistically calibrated" in r["scoring"]["method"]


def test_unknown_suspect_is_an_error_not_a_silent_zero():
    with pytest.raises(ValueError):
        analyse_conversation(ORDINARY, suspect="nobody")


def test_empty_conversation_rejected():
    with pytest.raises(ValueError):
        analyse_conversation([])


def test_bands_are_monotone():
    assert band(0.9) == "CRITICAL"
    assert band(0.6) == "HIGH"
    assert band(0.3) == "ELEVATED"
    assert band(0.15) == "LOW"
    assert band(0.0) == "MINIMAL"
