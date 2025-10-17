from typing import Any, TypedDict, Literal

ErrorTypes = Literal[
    "ok",
    "bad_question_clarity",
    "bad_options_clarity",
    "no_correct_answer",
    "multiple_correct_answers",
    "wrong_groundtruth",
    "expert"
]


class MMLUReduxDoc(TypedDict):
    question: str
    choices: list[str]
    answer: int
    error_type: ErrorTypes
    source: str
    correct_answer: str | None
    potential_reason: str


def process_choice(choice: str) -> int | None:
    "Check if the choice is valid and convert to int."
    if choice.isdigit():
        return int(choice)

    if choice == "A":
        return 0

    if choice == "B":
        return 1

    if choice == "C":
        return 2

    if choice == "D":
        return 3

    return None


def process_answer(doc: MMLUReduxDoc) -> int:
    error_type = doc["error_type"]
    answer = doc["answer"]

    if error_type == "ok" or error_type == "expert":
        return answer

    if error_type == "multiple_correct_answers":
        # TODO: Check if answer is one of the correct ones
        return answer  # Keep original answer

    if error_type == "no_correct_answer":
        return 0 # Move correct answer to 0

    # These should be removed from the dataset
    if error_type == "bad_question_clarity" or error_type == "bad_options_clarity":
        return answer  # Keep original answer

    correct_answer = doc["correct_answer"]
    if isinstance(correct_answer, str) and '(' in correct_answer:
        correct_answer = correct_answer[:correct_answer.index('(')].strip()

    choice = process_choice(correct_answer) if correct_answer else None

    if not correct_answer:
        return answer

    if error_type == "wrong_groundtruth":
        if choice is not None:
            return choice

        raise ValueError(f"Invalid correct answer: {correct_answer}")

    if choice is not None:
        return choice

    return 0

def safe_process_choice(doc: MMLUReduxDoc) -> list[str]:
    choices = [choice.strip() for choice in doc["choices"]]
    error_type = doc["error_type"]

    if error_type == "no_correct_answer" and doc["correct_answer"]:
        choices[0] = doc["correct_answer"].strip()

    return choices
