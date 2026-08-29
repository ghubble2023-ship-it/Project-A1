"""Scam playbook patterns, as data.

Keeping these as a table rather than as branches in code means the detection
surface can be reviewed, extended and argued about by people who do not read
Python -- which for fraud patterns is most of the people who know them best.

Each rule carries:

``playbook``  the operation it belongs to
``stage``     where in that operation's funnel it appears (1 = earliest)
``severity``  0-1, how strongly this phrasing alone implicates fraud
``label``     the plain-English name shown to a user

Severity is deliberately conservative for early-stage rules. "Wrong number,
sorry" is how a genuine misdial opens too; it is only damning in company.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    key: str
    playbook: str
    stage: int
    severity: float
    label: str
    pattern: re.Pattern

    @staticmethod
    def make(key, playbook, stage, severity, label, pattern) -> "Rule":
        return Rule(
            key, playbook, stage, severity, label, re.compile(pattern, re.IGNORECASE)
        )


RULES: list[Rule] = [
    # -- Pig butchering (sha zhu pan): long-con romance into fake investment --
    Rule.make(
        "pb_icebreaker", "pig_butchering", 1, 0.25,
        "Manufactured 'wrong number' opening",
        r"\b(?:wrong number|sorry to bother|my assistant gave me (?:this|your)|"
        r"saved this number|is this (?:mr|mrs|ms|dr)\b)",
    ),
    Rule.make(
        "pb_fate", "pig_butchering", 1, 0.20,
        "Instant destiny / fate framing",
        r"\b(?:must be (?:fate|destiny)|meant to meet|god brought us|"
        r"fate that we (?:met|connected))",
    ),
    Rule.make(
        "pb_wealth", "pig_butchering", 2, 0.35,
        "Unsolicited wealth signalling",
        r"\b(?:financial freedom|passive income|my uncle taught me|"
        r"insider (?:signal|tip)s?|market analysis|trading (?:gold|forex)|"
        r"quantitative trading|my mentor)",
    ),
    Rule.make(
        "pb_platform", "pig_butchering", 3, 0.60,
        "Steering to a controlled trading platform",
        r"\b(?:connect your wallet|dapp\b|defi platform|node contract|"
        r"liquidity (?:node|pool|mining)|vip node|guaranteed (?:returns|profit|yield)|"
        r"usdt\b|trc-?20|erc-?20)",
    ),
    Rule.make(
        "pb_deposit", "pig_butchering", 3, 0.65,
        "Deposit / top-up to unlock funds",
        r"\b(?:small test deposit|recharge (?:your )?account|deposit to activate|"
        r"top ?up (?:your )?(?:balance|account)|minimum deposit)",
    ),

    # -- Task / 'optimisation' scams: negative-balance extortion --------------
    Rule.make(
        "task_offer", "task_scam", 1, 0.40,
        "App-rating / task work offer",
        r"\b(?:rate apps|app optimi[sz]ation|merchant optimi[sz]ation|"
        r"hotel ratings|data optimi[sz]ation|daily tasks?|task commission|"
        r"complete \d+ (?:orders|tasks))",
    ),
    Rule.make(
        "task_trap", "task_scam", 3, 0.70,
        "Negative-balance / combo-task trap",
        r"\b(?:combo task|negative balance|recharge to unlock|funds? (?:are )?frozen|"
        r"withdrawal threshold|vip merchant level|unlock your withdrawal)",
    ),

    # -- Account takeover: 2FA interception and remote access -----------------
    Rule.make(
        "ato_code", "account_takeover", 2, 0.75,
        "Request for a one-time verification code",
        r"\b(?:read (?:me |back )?the code|cancellation code|verification code|"
        r"6[- ]digit code|sms code|otp\b|code (?:i|we) (?:just )?sent)",
    ),
    Rule.make(
        "ato_remote", "account_takeover", 2, 0.80,
        "Request to install remote-access software",
        r"\b(?:anydesk|teamviewer|rustdesk|quicksupport|ultraviewer|"
        r"screen ?share|remote (?:desktop|support) app)",
    ),
    Rule.make(
        "ato_safe_account", "account_takeover", 3, 0.85,
        "'Move your money to a safe account'",
        r"\b(?:safe account|secure account|move (?:your )?(?:money|funds) "
        r"to (?:a )?(?:safe|secure|new)|fraud department|"
        r"(?:your )?account (?:is|has been) compromised)",
    ),

    # -- Recovery fraud: re-victimising people who already lost money ---------
    Rule.make(
        "recovery", "recovery_fraud", 1, 0.75,
        "Offer to recover previously lost funds",
        r"\b(?:recover (?:your )?(?:crypto|funds|bitcoin|lost money|scammed)|"
        r"blockchain (?:tracing|forensics)|white ?hat hacker|"
        r"(?:interpol|fbi) (?:recovery|specialist)|smart contract clawback|"
        r"(?:retrieval|recovery|unlock|gas tracking) fee)",
    ),

    # -- Payment rails that cannot be reversed --------------------------------
    Rule.make(
        "pay_irreversible", "payment_pressure", 3, 0.70,
        "Push toward an irreversible payment rail",
        r"\b(?:gift ?cards?|steam card|google play card|apple card|"
        r"western union|moneygram|wire transfer|zelle|cash ?app|"
        r"bitcoin atm|crypto atm|btc address|wallet address)",
    ),
    # The theft itself. A gift card in the victim's hand costs the fraudster
    # nothing; the money moves only when they photograph the numbers. Asking to
    # *see* the card is therefore the most diagnostic moment in the whole
    # exchange, and it is the point at which the loss becomes irreversible.
    # The referent is mandatory. An early draft allowed a bare "can I see the
    # picture", which fires on "did you see the picture I sent" -- an ordinary
    # sentence -- and pushed a single innocuous message to HIGH. Requiring the
    # card/code object costs one hit on the reference transcript ("Can I see the
    # picture", said three messages after the card was named) and removes the
    # false positive. Context-free rules cannot have both; precision wins,
    # because the surrounding funnel still fires.
    Rule.make(
        "pay_proof", "payment_pressure", 4, 0.85,
        "Request for the card numbers, code or a photo of the receipt",
        r"\b(?:(?:can|could|let) (?:i|me) see (?:a |the )?(?:pic(?:ture)?|photo|image)"
        r" of (?:it|them|the card|the code|the back|the receipt)|"
        r"send (?:me )?(?:a |the )?(?:pic(?:ture)?|photo) of (?:it|the card|the code|the back)|"
        r"scratch (?:off )?the (?:back|silver|panel)|"
        r"(?:numbers?|code|pin) on the (?:back|card)|"
        r"read (?:me )?the (?:numbers?|code)|"
        r"what(?:'?s| is| of) the (?:card|code|number)|"
        r"redeem(?:ing)? code|card details)",
    ),
    # The pretext that justifies the errand. Distinct from the rail itself:
    # the fraudster needs a reason the victim must buy something *now*, and a
    # transport failure is the most common because it is unverifiable and
    # implies imminent arrival.
    Rule.make(
        "pay_pretext", "payment_pressure", 3, 0.45,
        "Unverifiable stranded / transport pretext for needing money now",
        r"\b(?:no gas in my car|out of gas|need gas money|"
        r"my car (?:broke|broke down|won'?t start)|"
        r"stuck at the (?:airport|station|border)|"
        r"need (?:money )?for (?:a )?(?:ticket|taxi|uber|cab|fare)|"
        r"customs (?:fee|charge|clearance)|delivery fee to release)",
    ),
    Rule.make(
        "pay_errand", "payment_pressure", 3, 0.40,
        "Request that the target buy something on the sender's behalf",
        r"\b(?:(?:can|could|will) you (?:get|buy|pick up|grab) (?:me|it|something|that)"
        r"(?: at| from)? (?:the )?(?:store|shop|walmart|target|cvs|walgreens)?|"
        r"just get it for me|get it and i'?ll pay you back|"
        r"(?:go|stop) (?:to|by) the store for me)",
    ),

    # -- Grooming: mapping the target before the ask --------------------------
    # Low severity on their own. A new acquaintance may reasonably ask where you
    # live. These matter as *company* -- an isolation probe next to a payment
    # push is a different object from either alone -- which is exactly what the
    # stage-span bonus in the classifier is for.
    Rule.make(
        "groom_locate", "grooming", 1, 0.25,
        "Early probe for precise location",
        r"\b(?:what(?:'?s| is)? your zip ?code|what zip|your postcode|"
        r"which (?:city|town) (?:are you|do you live) in|"
        r"where exactly (?:are you|do you live)|are you near(?:by)?)",
    ),
    Rule.make(
        "groom_isolation", "grooming", 2, 0.50,
        "Probe for whether the target is alone",
        r"\b(?:d(?:o|id) you live alone|are you (?:home )?alone|"
        r"anyone (?:else )?(?:there|home) with you|"
        r"is (?:anyone|someone) (?:with|around) you|"
        r"do you live (?:by yourself|with anyone))",
    ),

    # -- Coercion: isolation and artificial urgency ---------------------------
    Rule.make(
        "iso_secrecy", "coercion", 2, 0.55,
        "Instruction to keep it secret from family or bank",
        r"\b(?:keep this between us|don'?t tell (?:your |any)?"
        r"(?:family|spouse|husband|wife|bank|anyone|kids)|"
        r"they (?:won'?t|wouldn'?t) understand|strictly confidential)",
    ),
    # Pressure applied to silence rather than to a deadline. When a target stops
    # replying mid-ask, fraudsters switch from seduction to grievance, because
    # guilt reliably restarts a stalled conversation.
    Rule.make(
        "iso_guilt", "coercion", 3, 0.40,
        "Guilt or accusation when the target stops replying",
        r"\b(?:you(?:'?re| are) ignoring me|why (?:are|arent|aren'?t) you "
        r"(?:answering|replying|responding)|"
        r"(?:i thought )?you said you (?:would|loved|cared)|"
        r"don'?t you (?:trust|love|care about) me|"
        r"you(?:'?re| are) not serious about (?:me|us))",
    ),
    Rule.make(
        "iso_urgency", "coercion", 2, 0.45,
        "Artificial deadline pressure",
        r"\b(?:act (?:now|fast|within \d+ ?(?:min|hour))|"
        r"(?:offer|window|pool) (?:expires|closes|is closing)|"
        r"only \d+ ?(?:minutes|hours) left|before the market closes|last chance)",
    ),

    # -- Moving off a moderated platform --------------------------------------
    Rule.make(
        "migrate_app", "off_platform", 1, 0.35,
        "Push to move to an unmoderated chat app",
        r"\b(?:telegram|whats ?app|signal app|wechat|viber|line app|"
        r"t\.me/|wa\.me/|hangouts|google ?chat)",
    ),
    Rule.make(
        "migrate_reason", "off_platform", 1, 0.30,
        "Excuse for leaving the platform",
        r"\b(?:i (?:don'?t|rarely) (?:use|check) this app|"
        r"(?:my )?account here (?:is|will be) (?:deleted|deactivated|expiring)|"
        r"let'?s (?:talk|chat|continue) on)",
    ),

    # -- Refusal to verify identity -------------------------------------------
    Rule.make(
        "no_verify", "identity_evasion", 2, 0.55,
        "Deflecting a live video or voice check",
        r"\b(?:camera (?:is )?broken|can'?t (?:do|use) video|"
        r"my (?:phone|camera) (?:doesn'?t|won'?t) work|"
        r"video (?:call )?(?:later|not now)|(?:i'?m |am )?(?:on|at) an oil rig|"
        r"deployed overseas|working offshore)",
    ),
]


#: Financial identifiers worth extracting verbatim so a user can report them.
EXTRACTORS: dict[str, re.Pattern] = {
    "telegram_handle": re.compile(r"\b(?:t\.me/|@)([A-Za-z][A-Za-z0-9_]{4,31})\b"),
    "whatsapp_link": re.compile(r"\bwa\.me/(\+?\d{6,15})\b", re.IGNORECASE),
    "btc_address": re.compile(r"\b(?:bc1[a-z0-9]{25,62}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})\b"),
    "eth_address": re.compile(r"\b0x[a-fA-F0-9]{40}\b"),
    "tron_address": re.compile(r"\bT[A-Za-z0-9]{33}\b"),
    "url": re.compile(r"https?://[^\s<>\"']+"),
}
