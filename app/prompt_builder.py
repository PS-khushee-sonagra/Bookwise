"""
Prompt Builder for BookWise

Responsibilities:
- Build grounded prompts for the LLM.
- Format retrieved context.
- Instruct the model to answer only from the document.
"""

from typing import Dict, List


class PromptBuilder:
    """
    Builds prompts for Gemini.
    """

    SYSTEM_PROMPT = """
You are BookWise AI, an intelligent document question-answering assistant.

Your task is to answer questions comprehensively and accurately using ONLY the provided document context.

Instructions:
1. Base your answer strictly on the Context section. Synthesize information from multiple pages or sections of the Context if required to form a complete and coherent answer.
2. Do not introduce outside knowledge or make assumptions not supported by the Context.
3. If the answer cannot be found in the Context, reply exactly:
   "I don't know based on the uploaded document."
4. Keep the answer accurate, structured, and easy to read. Use bullet points or numbered lists where appropriate.
5. Do NOT include any source pages, page numbers, citations, or a "Source Pages" section in your answer. The application will handle citation generation separately.
"""

    def build_prompt(
        self,
        question: str,
        retrieved_chunks: List[Dict],
    ) -> str:
        """
        Build the final prompt for Gemini.

        Args:
            question:
                User question.

            retrieved_chunks:
                Retrieved document chunks.

        Returns:
            Complete prompt string.
        """

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if not retrieved_chunks:
            context = "No relevant context was retrieved from the document."

        else:

            context_parts = []

            for chunk in retrieved_chunks:
                metadata = chunk.get('metadata', {})
                source = metadata.get('source', 'Unknown')
                page_number = chunk.get('page_number', 'Unknown')

                context_parts.append(
                    f"""Document: {source} | Page: {page_number}

{chunk['text']}
"""
                )

            context = "\n" + ("-" * 60) + "\n"
            context += ("\n" + ("-" * 60) + "\n").join(context_parts)

        prompt = f"""
{self.SYSTEM_PROMPT}

===========================================================
CONTEXT
===========================================================

{context}

===========================================================
QUESTION
===========================================================

{question}

===========================================================
ANSWER
===========================================================

Provide your answer below.
"""

        return prompt