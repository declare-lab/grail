import regex as re
from math_verify import parse, verify

_BOXED_PATTERN = re.compile(r"\\boxed\{((?:[^{}]+|\{(?1)\})*)\}")


def is_conversational(messages):
    if isinstance(messages, list):
        message = messages[0]
        # Each message must a list of dictionaries with keys "role" and "content"
        if isinstance(message, dict) and "role" in message and "content" in message:
            return True
    return False


def verify_ans(gold, answer):
    if answer is None:
        return False
    try:
        gold = parse(f"${gold}$")
        answer = parse(f"${answer}$")
        return verify(gold, answer, timeout_seconds=5)  # default timeout is 5s
    except BaseException as e:
        return False


def check_correctness(text: str, gold: str) -> float:
    matches = list(_BOXED_PATTERN.finditer(text))
    if not matches:
        return 0.0

    last_boxed = matches[-1]
    answer = last_boxed.group(1).strip()

    if not verify_ans(gold, answer):
        return 0.0

    return 1.0
