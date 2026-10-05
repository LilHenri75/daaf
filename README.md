# daaf — decision-as-a-function

Write a typed Python function with a docstring. [Laya](https://github.com/NandhaKishorM/laya) answers it, locally, in any of 100+ languages, in tens of milliseconds.

```python
from enum import Enum, IntEnum
from typing import Literal
from daaf import decide

@decide
def is_scam(message: str) -> bool:
    """Is this message a scam trying to get money or credentials?"""

class Dept(Enum):
    billing = "invoices, payments, refunds"   # the value describes the option to the model
    support = "bugs, outages, login problems"
    sales = "pricing, buying a product"

@decide
def department(message: str) -> Dept:
    """Which department should handle this message?"""

is_scam("Votre colis est bloqué, payez 1,99 € ici")  # True
department("I was billed twice")                     # Dept.billing
```

The function body is never run: the docstring is the question, the return type is the set of possible answers.

## Install

```bash
pip install daaf
```

The first call downloads the Laya model (a few hundred MB) and loads it, which takes a few seconds. Later calls are fast.

## Return types

| Annotation | Result |
|---|---|
| `bool` | `True` / `False` |
| `Literal["a", "b", ...]` | one of the values (strings, ints…) |
| `Enum` subclass | one member; string values describe each option, which helps the model |
| `IntEnum` subclass | an ordered level (Laya's `score` type); member names are read as the level labels |

```python
class Urgency(IntEnum):
    not_urgent = 0
    soon = 1
    blocking = 2

@decide
def urgency(message: str) -> Urgency:
    """How urgent is this message?"""

urgency("Production is down!")  # Urgency.blocking
```

## Arguments

- One argument: its value is the text Laya reads.
- Several arguments: Laya reads them as named fields, for example `def urgent(subject: str, body: str) -> bool`.

## Many inputs at once

```python
is_scam.batch(["message 1", "message 2", ...])         # one-argument function
urgent.batch([("subject", "body"), {"subject": "s", "body": "b"}])  # several arguments
```

Inputs share forward passes, which is much faster than a loop. Results come back in input order. `batch_size=` caps how many go through the model at once.

## Options

```python
@decide(min_confidence=0.8)      # raise daaf.Undecided below 80% instead of guessing
def is_scam(message: str) -> bool:
    """Is this message a scam trying to get money or credentials?"""

@decide(model="multilingual", max_len=8192)  # anything else goes to laya's Router.predict
def is_contract(document: str) -> bool:
    """Is this document a contract?"""

@decide(router=my_router)        # your own Router, or any object with Laya's .predict / .predict_batch
def is_spam(text: str) -> bool:
    """Is this message spam?"""
```

`Undecided` carries Laya's raw answer in `.answer`, so you can hand the case to a person or a larger model. In `.batch(...)`, the first undecided input raises it.

`is_scam.question` shows the exact question sent to Laya.

## Accuracy

daaf adds no intelligence: answers are as good as Laya's zero-shot checkpoints, which can make mistakes, especially on subtle questions. Test on your own examples, and use `min_confidence` where an error costs something. Clear docstrings and `Enum` descriptions help.

## License

Apache-2.0, like Laya.

## Tests

```bash
python test_daaf.py   # no model needed, a fake router stands in for Laya
```
