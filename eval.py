# Evaluation: prove the RAG assistant actually works, with a number
# instead of just a feeling.
#
# For each question in eval_questions.json, we:
#   1. Run it through our real answer() function (same as ask.py/app.py).
#   2. Ask Gemini to act as a judge: does the generated answer match
#      what we expected, in substance? This is called "LLM-as-judge" --
#      using a model to grade another model's output, since checking
#      thousands of free-text answers by hand doesn't scale.
# Then we print a per-question pass/fail and a final score like "6/8".

import json
import time

from llm import ask as ask_llm
from rag import answer

EVAL_FILE = "eval_questions.json"

# The free tier only allows 5 Gemini requests per minute. Each question
# makes 2 calls (one to answer, one to judge), so pausing briefly between
# every call keeps us under that limit instead of tripping it.
SECONDS_BETWEEN_CALLS = 13

JUDGE_SYSTEM_INSTRUCTION = (
    "You are grading whether a generated answer matches an expected answer "
    "for a question-answering system. Respond with exactly one word: YES if "
    "the generated answer is factually consistent with the expected answer "
    "(even if worded differently), or NO if it is missing key information, "
    "contradicts the expected answer, or fails to honestly say 'I don't "
    "know' when the expected answer says the system should not know. "
    "Respond with only YES or NO -- nothing else."
)


def judge(question, expected_answer, generated_answer):
    """Ask Gemini whether generated_answer matches expected_answer."""
    prompt = (
        f"Question: {question}\n\n"
        f"Expected answer: {expected_answer}\n\n"
        f"Generated answer: {generated_answer}\n\n"
        "Does the generated answer correctly match the expected answer? "
        "Respond YES or NO only."
    )
    # Temperature 0.0: we want the judge to be as consistent as possible,
    # not creative.
    verdict = ask_llm(prompt, system_instruction=JUDGE_SYSTEM_INSTRUCTION, temperature=0.0)
    return verdict.strip().upper().startswith("YES")


def main():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    correct = 0
    for i, case in enumerate(test_cases, start=1):
        question = case["question"]
        expected_answer = case["expected_answer"]

        generated_answer, sources = answer(question)
        time.sleep(SECONDS_BETWEEN_CALLS)

        passed = judge(question, expected_answer, generated_answer)
        time.sleep(SECONDS_BETWEEN_CALLS)

        status = "PASS" if passed else "FAIL"
        print(f"[{i}/{len(test_cases)}] {status} -- {question}")
        if not passed:
            print(f"    expected:  {expected_answer}")
            print(f"    generated: {generated_answer}")
            print(f"    sources:   {sources}")

        if passed:
            correct += 1

    print(f"\nScore: {correct}/{len(test_cases)} correct")


if __name__ == "__main__":
    main()
