"""
LLM interface for BookWise.

Responsibilities:
- Initialize the Gemini model.
- Generate answers from prompts.
"""

import logging

from google import genai

from app.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)


class LLMGenerator:
    """
    Gemini client for BookWise.
    """

    def __init__(
    self,
    model_name=GEMINI_MODEL,
) -> None:
        if not GEMINI_API_KEY:
            raise ValueError(
                "Gemini API key is missing."
            )

        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        self.model_name = model_name

        logger.info(
            "Gemini client initialized."
        )

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate an answer from Gemini.

        Args:
            prompt:
                Complete prompt string.

        Returns:
            Generated answer.
        """

        if not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        try:

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

            if response.text:
                return response.text.strip()

            return "No response generated."

        except Exception as e:

            logger.exception(
                "Gemini generation failed."
            )

            raise RuntimeError(
                f"Gemini Error: {e}"
            )