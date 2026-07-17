import os
import sys

# Add project root to sys.path to allow importing app
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

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
