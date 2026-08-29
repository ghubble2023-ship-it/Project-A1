"""Gift-card romance fraud: the funnel the original rule set could not see.

These tests are derived from a real transcript in the project's Drive corpus
(`Chat string`), paraphrased here so the fixtures carry no real identifiers.
The original rule set fired exactly one rule on that conversation -- `Apple
Card` -- and scored an unambiguous, completed gift-card scam as ELEVATED, three
bands below the top.

The gap mattered because the rules were written for investment fraud (pig
butchering, task scams, account takeover) and gift-card romance fraud runs a
different funnel: locate, isolate, errand, pretext, rail, and then the request
that actually moves the money -- a photograph of the card.

The specificity tests at the bottom are the ones that must not be allowed to
rot. Every rule added here is a new opportunity to flag an ordinary
conversation, and a scam detector that cries wolf on family arguments is worse
than none.
"""

from __future__ import annotations

import pytest

from sentinel.text.pipeline import analyse_conversation


def _conv(pairs):
    return [{"sender": s, "text": t} for s, t in pairs]


SCAM = _conv([
    ("suspect", "Hello"),
    ("suspect", "How are you doing"),
    ("me", "Good, you?"),
    ("suspect", "Where are you located"),
    ("me", "Central Oregon"),
    ("suspect", "What your zip code so I can know maybe you are nearby me here"),
    ("me", "97xxx"),
    ("suspect", "Ok did you live alone"),
    ("me", "I do"),
    ("suspect", "Will you like us to have fun together today"),
    ("suspect", "Can you get me something at the store so I can come over now"),
    ("suspect", "Just get it I have no gas in my car"),
    ("suspect", "Just get a Apple Card at the store so I can get gas in my car"),
    ("me", "Ok i have it"),
    ("suspect", "Can I see a picture of it"),
    ("me", "?"),
    ("suspect", "Are you there"),
    ("suspect", "You are ignoring me"),
    ("suspect", "What of the card you get"),
])


def test_the_whole_funnel_is_detected_not_just_the_payment_rail():
    result = analyse_conversation(SCAM)
    playbooks = set(result["identified_playbooks"])
    assert {"payment_pressure", "grooming", "coercion"} <= playbooks


def test_a_completed_gift_card_scam_scores_at_least_high():
    result = analyse_conversation(SCAM)
    assert result["risk_score"] >= 0.5, result["risk_score"]
    assert result["risk_band"] in {"HIGH", "CRITICAL"}


def test_the_proof_of_purchase_request_is_caught():
    """The card in the victim's hand costs the fraudster nothing. The photo is
    the theft, so this rule is the one that must never regress."""
    result = analyse_conversation(SCAM)
    rules = {p["playbook"] + ":" + p["finding"] for p in result["matched_phrases"]}
    assert any("card numbers" in r for r in rules), rules


def test_findings_quote_the_sender_verbatim():
    result = analyse_conversation(SCAM)
    for phrase in result["matched_phrases"]:
        source = SCAM[phrase["message_index"]]["text"]
        assert phrase["quote"].lower() in source.lower()


@pytest.mark.parametrize(
    "text,expect_rule",
    [
        ("what's your zip code", "grooming"),
        ("are you home alone right now", "grooming"),
        ("my car broke down and I need money for a taxi", "payment_pressure"),
        ("can you pick up something at the store", "payment_pressure"),
        ("send me a photo of the card", "payment_pressure"),
        ("scratch off the silver and read me the numbers on the back", "payment_pressure"),
        ("why aren't you replying, don't you care about me", "coercion"),
    ],
)
def test_each_new_rule_fires_on_its_own_tradecraft(text, expect_rule):
    """One line in isolation may be too weak to *name* a playbook -- a single
    grooming probe saturates to 0.20, below the naming threshold. It must still
    be attributed, and reported as a weak match rather than contradicted."""
    result = analyse_conversation(_conv([("suspect", text)]))
    named = set(result["identified_playbooks"]) | set(result["weak_matches"])
    assert expect_rule in named, result["matched_phrases"]
    assert {p["playbook"] for p in result["matched_phrases"]} <= named


# --- specificity: ordinary life must not trip these -------------------------

BENIGN = _conv([
    ("suspect", "I'm just past even wanting to go do this"),
    ("me", "I bet you look great"),
    ("suspect", "I can't even do my hair and do anything"),
    ("me", "take him to buy more to go get a birthday present after work if "
           "that's what you want me to do, I don't know why you can't go before "
           "the end of the day before dinner time"),
    ("me", "Well I guess I can't even call to talk to Walter anymore either"),
    ("suspect", "I was just getting out of the shower. No one said you couldn't "
                "talk to Walter."),
    ("suspect", "My phone wasn't even in the bathroom with me. It was sitting in "
                "the bedroom on the charger and I don't have my phone on ring."),
    ("me", "Im back inside ill see you guys when I get off"),
])


def test_a_real_domestic_argument_is_not_a_scam():
    """Emotionally charged, mentions buying a present, includes an accusation of
    not answering the phone. Every one of those neighbours a new rule."""
    result = analyse_conversation(BENIGN)
    assert result["risk_band"] == "MINIMAL"
    assert result["risk_score"] == 0.0, result["matched_phrases"]


@pytest.mark.parametrize("text", [
    "can you grab milk from the store on your way home",
    "where are you located? I'll come meet you",
    "I ran out of gas on the way to work, what a day",
    "did you see the picture I sent",
    "why aren't you answering, dinner is getting cold",
])
def test_single_innocuous_lines_do_not_reach_a_warning_band(text):
    """Any one of these can appear in a scam. On its own it is a Tuesday."""
    result = analyse_conversation(_conv([("suspect", text)]))
    assert result["risk_band"] in {"MINIMAL", "LOW", "ELEVATED"}, (
        text, result["risk_score"]
    )


def test_escalation_velocity_no_longer_cliffs_to_zero():
    """A payment ask twenty messages in is slower than one at message two, but
    it is not uninformative -- the old linear rule scored both the same as an
    ask five hundred messages in."""
    def ask_at(position):
        pad = [("suspect", "hey how are you")] * (position - 1)
        return analyse_conversation(
            _conv(pad + [("suspect", "just get a gift card at the store")])
        )["escalation"]["score"]

    early, mid, late = ask_at(1), ask_at(21), ask_at(101)
    assert early > mid > late > 0.0
    assert early == pytest.approx(1.0)


def test_report_never_contradicts_itself():
    """Every playbook credited with a phrase is named somewhere -- either as
    identified or as a weak match. The two fields must partition, not overlap."""
    for conv in (SCAM, BENIGN):
        result = analyse_conversation(conv)
        named = set(result["identified_playbooks"]) - {"none_matched"}
        weak = set(result["weak_matches"])
        assert not (named & weak)
        assert {p["playbook"] for p in result["matched_phrases"]} <= (named | weak)
