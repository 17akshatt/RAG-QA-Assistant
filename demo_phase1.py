# Demo: the SAME question, answered with different settings.
# This shows what "system_instruction" and "temperature" actually change.

from llm import ask

question = "What is the best programming language?"

print("=== 1) No system instruction (model just answers normally) ===")
print(ask(question))
print()

print("=== 2) System instruction changes the model's ROLE (pirate) ===")
print(ask(question, system_instruction="You are a pirate. Answer in pirate speak."))
print()

print("=== 3) System instruction changes the model's STYLE (strict, one sentence) ===")
print(ask(
    question,
    system_instruction="You are a concise, no-nonsense professor. Answer in exactly one sentence.",
    temperature=0.0,  # low temperature = focused, consistent answers
))
print()

print("=== 4) Same question twice at HIGH temperature (watch it vary) ===")
print("Run 1:", ask(question, temperature=1.5))
print()
print("Run 2:", ask(question, temperature=1.5))
