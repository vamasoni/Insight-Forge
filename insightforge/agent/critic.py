"""Critic: checks every claim against the evidence it cites.

Layer 1 (deterministic, free, can't be talked out of it):
  - cited evidence exists
  - every number in the claim matches a number the cited evidence contains (after rounding / % scaling)
  - causal language is flagged (all Olist/BIRD data is observational)
  - "significant" requires a cited test with p < 0.05
Layer 2 (LLM): semantic support, run only on claims that pass layer 1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from insightforge import prompts
from insightforge.evidence import CalcEvidence, Claim, EvidenceStore
from insightforge.llm import LLMProvider

CAUSAL = re.compile(
    r"\b(caus(e|es|ed|ing)|driv(e|es|en|ing)|lead(s)? to|led to|result(s|ed)? in|because of|due to|"
    r"as a result of|impact(s|ed)? (on )?|effect of|attributable to|thanks to|boost(s|ed)?|hurt(s)?|"
    r"responsible for|explains why|makes customers|reduc(es|ed) .* by causing)\b", re.I)
HEDGED_CAUSAL = re.compile(r"\b(not|cannot|can't|doesn't|does not|no evidence).{0,40}"
                           r"(caus|driv|lead|result in)", re.I)
SIGNIFICANT = re.compile(r"\b(statistically )?significant(ly)?\b", re.I)
NOT_SIGNIFICANT = re.compile(r"\b(not|no|non-?)\s*(statistically )?significant", re.I)

_NUM = re.compile(r"(?<![\w.])[-+]?(?:R\$\s?|\$)?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(%|pp|k|K|M|million|thousand|x)?")


@dataclass
class NumberMention:
    raw: str
    value: float
    decimals: int
    unit: str


def extract_numbers(text: str) -> list[NumberMention]:
    out = []
    for m in _NUM.finditer(text):
        ip, frac, unit = m.group(1), m.group(2) or "", (m.group(3) or "").lower()
        v = float(ip.replace(",", "") + frac)
        dec = len(frac) - 1 if frac else 0
        if unit in ("k", "thousand"):
            v, dec = v * 1e3, max(0, dec - 3)
        elif unit in ("m", "million"):
            v, dec = v * 1e6, max(0, dec - 6)
        out.append(NumberMention(m.group(0).strip(), v, dec, unit))
    return out


def _is_trivial(n: NumberMention, text: str) -> bool:
    if n.unit in ("%", "pp", "x"):
        return False
    if n.decimals == 0 and 1900 <= n.value <= 2100 and "," not in n.raw:  # years (but not "2,017")
        return True
    if n.decimals == 0 and n.value <= 12:  # "top 5", "3 states", "score of 5"
        return True
    # ids like S1/Q2/F3 are excluded by the regex lookbehind; claim markers too
    return False


def number_supported(n: NumberMention, evidence_nums: list[float]) -> bool:
    """A claim number is supported if some evidence value rounds to it at the claim's own precision.

    No blanket relative tolerance: that let a wrong number pass by landing near an unrelated value in
    the same table. Only k/M-suffixed numbers get a 5% band, since "1.2M" is inherently approximate.
    """
    if n.unit in ("k", "m", "million", "thousand"):
        return any(abs(abs(e) - n.value) <= 0.05 * n.value for e in evidence_nums)
    tol = 0.5 * 10 ** (-n.decimals) + 1e-9
    targets = [(n.value, tol)]
    if n.unit in ("%", "pp"):
        targets.append((n.value / 100, tol / 100))  # 42.1% may be stored as 0.421
    return any(abs(abs(e) - t) <= tt for e in evidence_nums for t, tt in targets)


@dataclass
class DeterministicVerdict:
    status: str  # supported | unsupported | unverified_number | causal_overreach | needs_hedge
    notes: list[str]


def deterministic_check(claim: Claim, store: EvidenceStore) -> DeterministicVerdict:
    notes: list[str] = []
    if not claim.evidence_ids:
        return DeterministicVerdict("unsupported", ["claim cites no evidence"])
    missing = [e for e in claim.evidence_ids if store.get(e) is None]
    if missing:
        return DeterministicVerdict("unsupported", [f"cites evidence that does not exist: {', '.join(missing)}"])

    if CAUSAL.search(claim.text) and not HEDGED_CAUSAL.search(claim.text):
        return DeterministicVerdict("causal_overreach",
                                    [f"causal wording ({CAUSAL.search(claim.text).group(0)!r}) on observational data"])

    if SIGNIFICANT.search(claim.text) and not NOT_SIGNIFICANT.search(claim.text):
        tests = [store.get(e) for e in claim.evidence_ids if isinstance(store.get(e), CalcEvidence)]
        ps = [t.outputs.get("p_value") for t in tests if isinstance(t.outputs.get("p_value"), (int, float))]
        if not ps:
            return DeterministicVerdict("needs_hedge", ["says 'significant' but cites no statistical test"])
        if min(ps) >= 0.05:
            return DeterministicVerdict("unsupported", [f"says 'significant' but cited test has p={min(ps):.3g}"])

    ev_nums = store.numbers_for(claim.evidence_ids)
    bad = [n.raw for n in extract_numbers(claim.text) if not _is_trivial(n, claim.text)
           and not number_supported(n, ev_nums)]
    if bad:
        return DeterministicVerdict("unverified_number",
                                    [f"number(s) not found in cited evidence: {', '.join(bad)}"])
    upstream = list(claim.evidence_ids)
    for eid in claim.evidence_ids:
        ev = store.get(eid)
        if isinstance(ev, CalcEvidence):
            upstream += ev.inputs
    for eid in dict.fromkeys(upstream):
        if getattr(store.get(eid), "truncated", False):
            notes.append(f"{eid} was truncated at the row limit; results computed from it may be incomplete")
    return DeterministicVerdict("supported", notes)


class Critic:
    def __init__(self, llm: LLMProvider | None, use_llm: bool = True):
        self.llm = llm
        self.use_llm = use_llm and llm is not None

    def review(self, store: EvidenceStore) -> list[Claim]:
        passed: list[Claim] = []
        for c in store.claims.values():
            v = deterministic_check(c, store)
            c.status, c.critic_notes = v.status, list(v.notes)
            if v.status == "supported":
                passed.append(c)
        if self.use_llm and passed:
            self._llm_review(passed, store)
        return list(store.claims.values())

    def _llm_review(self, claims: list[Claim], store: EvidenceStore) -> None:
        ids = sorted({e for c in claims for e in c.evidence_ids}, key=lambda s: (s[0], int(s[1:])))
        user = prompts.CRITIC_USER.format(
            claims="\n".join(f"{c.id}: {c.text} (cites {', '.join(c.evidence_ids)})" for c in claims),
            evidence="\n\n".join(store.describe(e) for e in ids))
        try:
            data = self.llm.complete_json(prompts.CRITIC_SYSTEM, user, role="critic", max_tokens=1500)
        except Exception as e:  # the deterministic layer already ran; don't fail the report
            for c in claims:
                c.critic_notes.append(f"LLM critic unavailable: {e}")
            return
        verdicts = {v.get("id"): v for v in (data.get("verdicts", []) if isinstance(data, dict) else [])}
        valid = {"supported", "unsupported", "causal_overreach", "needs_hedge"}
        for c in claims:
            v = verdicts.get(c.id)
            if v and v.get("verdict") in valid:
                c.status = v["verdict"]
                if v.get("reason") and c.status != "supported":
                    c.critic_notes.append(v["reason"])

    @staticmethod
    def feedback(store: EvidenceStore) -> str:
        bad = [c for c in store.claims.values() if c.status != "supported"]
        return "\n".join(f"- {c.id} [{c.status}] \"{c.text}\": {'; '.join(c.critic_notes)}" for c in bad)
