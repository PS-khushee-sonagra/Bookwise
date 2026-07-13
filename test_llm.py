from app.llm import LLMGenerator

llm = LLMGenerator()

answer = llm.generate(
    """
What is Artificial Intelligence?
Answer in one sentence.
"""
)

print()

print(answer)