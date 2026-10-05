"""Tests without the model: a fake router stands in for Laya. Run: python -m pytest (or python test_daaf.py)."""
from enum import Enum, IntEnum
from typing import Literal

from daaf import Undecided, decide


class FakeRouter:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def predict(self, state, questions, **kw):
        self.calls.append((state, questions, kw))
        return {"answers": {"answer": self.answer}}

    def predict_batch(self, requests, **kw):
        return [self.predict(r["state"], r["questions"], **kw) for r in requests]


def test_bool():
    r = FakeRouter({"noul": 0.9})

    @decide(router=r)
    def is_scam(message: str) -> bool:
        """Is this a scam?"""

    assert is_scam("pay now") is True
    assert r.calls[0][:2] == ("pay now", {"answer": {"type": "noul", "instructions": "Is this a scam?"}})


def test_literal_and_several_arguments():
    r = FakeRouter({"choice": "2"})

    @decide(router=r)
    def urgency(subject: str, body: str = "") -> Literal[1, 2, 3]:
        """How urgent?"""

    assert urgency("help") == 2  # the text label comes back as the original value
    assert r.calls[0][0] == {"subject": "help", "body": ""}


def test_enum_values_describe_options():
    class Dept(Enum):
        billing = "invoices, refunds"
        support = "bugs, outages"

    r = FakeRouter({"choice": "support"})

    @decide(router=r)
    def dept(text: str) -> Dept:
        """Which department?"""

    assert dept("down") is Dept.support
    assert r.calls[0][1]["answer"]["criteria"] == {"billing": "invoices, refunds", "support": "bugs, outages"}


def test_intenum_is_an_ordered_score():
    class Urgency(IntEnum):
        high = 5
        low = 1

    r = FakeRouter({"score": 0.4, "probabilities": {"0": 0.3, "1": 0.7}})

    @decide(router=r)
    def urgency(text: str) -> Urgency:
        """How urgent?"""

    assert urgency("x") is Urgency.high  # most probable level, not the rounded average
    assert r.calls[0][1]["answer"]["criteria"] == ["low", "high"]  # sorted by value


def test_batch():
    r = FakeRouter({"noul": 0.2})

    @decide(router=r, min_confidence=0.5, model="multilingual")
    def ok(subject: str, body: str) -> bool:
        """Ok?"""

    assert ok.batch([("a", "b"), {"subject": "c", "body": "d"}]) == [False, False]
    assert [c[0] for c in r.calls] == [{"subject": "a", "body": "b"}, {"subject": "c", "body": "d"}]


def test_low_confidence_raises():
    @decide(router=FakeRouter({"noul": 0.55, "confidence": 0.55, "low_confidence": True}), min_confidence=0.8)
    def f(text: str) -> bool:
        """Yes?"""

    try:
        f("x")
        raise AssertionError("expected Undecided")
    except Undecided as e:
        assert e.answer["noul"] == 0.55


def test_bad_definitions_fail_at_decoration():
    for bad in ("def g(t: str) -> bool: pass", 'def g(t: str) -> str:\n    """Q?"""'):
        ns = {}
        exec(bad, ns)
        try:
            decide(ns["g"], router=FakeRouter({}))
            raise AssertionError("expected TypeError")
        except TypeError:
            pass


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_"):
            t()
    print("ok")
