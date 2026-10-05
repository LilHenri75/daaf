"""Decision-as-a-function: write a typed Python function with a docstring, Laya answers it.

    from typing import Literal
    from daaf import decide

    @decide
    def is_scam(message: str) -> bool:
        \"\"\"Is this message a scam trying to get money or credentials?\"\"\"

    @decide
    def department(message: str) -> Literal["billing", "support", "sales"]:
        \"\"\"Which department should handle this message?\"\"\"

    is_scam("Your parcel is held, pay 1.99 EUR here")   # True
    department("I was charged twice")                    # "billing"
"""
import functools
import inspect
from enum import Enum, IntEnum
from typing import Literal, get_args, get_origin, get_type_hints

__all__ = ["decide", "Undecided"]
__version__ = "0.1.0"

_router = None


def _default_router():
    global _router
    if _router is None:
        from laya import Router  # imported lazily: loading torch is slow
        _router = Router()
    return _router


class Undecided(Exception):
    """Laya's confidence fell below the function's `min_confidence`."""

    def __init__(self, function, answer):
        self.answer = answer
        super().__init__(f"{function}: confidence {answer.get('confidence', 0):.2f} too low")


def _question(fn):
    """Build the Laya question from the docstring and return annotation, plus a decoder."""
    doc = inspect.getdoc(fn)
    if not doc:
        raise TypeError(f"{fn.__name__}: the docstring is the question, it must not be empty")
    ret = get_type_hints(fn).get("return")
    if ret is bool:
        return {"type": "noul", "instructions": doc}, lambda a: a["noul"] >= 0.5
    if get_origin(ret) is Literal:
        values = get_args(ret)
        labels = {str(v): v for v in values}
        if len(labels) != len(values):
            raise TypeError(f"{fn.__name__}: Literal values must be distinct once turned into text")
        question = {"type": "choice", "instructions": doc, "criteria": dict.fromkeys(labels)}
        return question, lambda a: labels[a["choice"]]
    if inspect.isclass(ret) and issubclass(ret, IntEnum):
        # Ordered levels: Laya's `score` type, levels sorted by value, names read as text.
        levels = sorted(ret)
        question = {"type": "score", "instructions": doc,
                    "criteria": [m.name.replace("_", " ") for m in levels]}
        return question, lambda a: levels[_level(a)]
    if inspect.isclass(ret) and issubclass(ret, Enum):
        # The member name is the label, its value (a string) describes it to the model.
        criteria = {m.name: m.value if isinstance(m.value, str) else None for m in ret}
        question = {"type": "choice", "instructions": doc, "criteria": criteria}
        return question, lambda a: ret[a["choice"]]
    raise TypeError(f"{fn.__name__}: the return annotation must be bool, Literal[...], an Enum "
                    f"or an IntEnum, got {ret!r}")


def _level(answer):
    """Most probable level of a `score` answer (its `score` field is an average, not a level)."""
    probs = answer.get("probabilities")
    if probs:
        return int(max(probs, key=lambda k: float(probs[k])))
    return round(float(answer["score"]))


def decide(fn=None, *, min_confidence=None, router=None, **predict_kwargs):
    """Turn a typed, documented function into a Laya decision.

    `min_confidence`: below it, raise `Undecided` instead of returning a guess.
    `router`: any object with Laya's `predict(state, questions, **kw)` (default: `laya.Router()`).
    Other keyword arguments go to `predict` (for example `model="multilingual"`, `max_len=8192`).
    """
    if fn is None:  # used as @decide(...)
        return lambda f: decide(f, min_confidence=min_confidence, router=router, **predict_kwargs)

    question, decode = _question(fn)
    signature = inspect.signature(fn)
    if min_confidence is not None:
        predict_kwargs["min_confidence"] = min_confidence

    def state(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        params = bound.arguments
        # One argument: the text itself. Several: Laya reads a dict of named fields.
        return next(iter(params.values())) if len(params) == 1 else dict(params)

    def answer_of(result):
        answer = result["answers"]["answer"]
        if answer.get("low_confidence"):
            raise Undecided(fn.__name__, answer)
        return decode(answer)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = (router or _default_router()).predict(state(*args, **kwargs), {"answer": question},
                                                       **predict_kwargs)
        return answer_of(result)

    def batch(items, batch_size=None):
        """Answer many inputs in shared forward passes, in input order.

        For a one-argument function each item is that argument; otherwise each item is a dict
        of keyword arguments or a tuple of positional ones.
        """
        if len(signature.parameters) == 1:
            states = [state(item) for item in items]
        else:
            states = [state(**i) if isinstance(i, dict) else state(*i) for i in items]
        call = {k: v for k, v in predict_kwargs.items() if k == "min_confidence"}
        per_request = {k: v for k, v in predict_kwargs.items() if k != "min_confidence"}
        requests = [{"state": s, "questions": {"answer": question}, **per_request} for s in states]
        results = (router or _default_router()).predict_batch(requests, batch_size=batch_size, **call)
        return [answer_of(r) for r in results]

    wrapper.question = question
    wrapper.batch = batch
    return wrapper
