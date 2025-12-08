from typing import List, Dict
from lib.message.prompts import (
    DRAFT_SYSTEM_INSTRUCTION,
    DRAFT_INPUT_TEMPLATE,
    REFINE_SYSTEM_INSTRUCTION,
    REFINE_INPUT_TEMPLATE
)


class PromptGenerator:
    """
    Handles the creation of prompts for the Draft-and-Refine summarization methodology.
    """

    def _format_sentences(self, sentences: List[str]) -> str:
        """
        Formats a list of sentences into a bulleted string representation.

        Args:
            sentences (List[str]): A list of extracted sentences.

        Returns:
            str: A formatted string where each sentence is on a new line with an arrow.
        """
        if not sentences:
            return "No specific sentences provided."

        # Clean and format sentences
        cleaned = [s.strip() for s in sentences if s.strip()]
        return "-> " + "\n-> ".join(cleaned)

    def generate_draft_prompt(self, title: str, abstract: str) -> Dict[str, str]:
        """
        Generates the prompt for the first stage: creating the initial draft.

        Args:
            title (str): The title of the paper.
            abstract (str): The abstract of the paper.

        Returns:
            Dict[str, str]: A dictionary containing the 'instruction' (system prompt)
                            and the formatted 'content' (user prompt).
        """
        formatted_content = DRAFT_INPUT_TEMPLATE.format(
            title=title.strip(),
            abstract=abstract.strip()
        ).strip()

        return {
            "instruction": DRAFT_SYSTEM_INSTRUCTION.strip(),
            "content": formatted_content
        }

    def generate_refine_prompt(self, title: str, current_draft: str, selected_sentences: List[str]) -> Dict[str, str]:
        """
        Generates the prompt for the second stage: refining the draft with selected sentences.

        Args:
            title (str): The title of the paper (for context).
            current_draft (str): The output generated in the draft stage.
            selected_sentences (List[str]): Key sentences extracted from the full text.

        Returns:
            Dict[str, str]: A dictionary containing the 'instruction' (system prompt)
                            and the formatted 'content' (user prompt).
        """
        formatted_sentences = self._format_sentences(selected_sentences)

        formatted_content = REFINE_INPUT_TEMPLATE.format(
            title=title.strip(),
            draft=current_draft.strip(),
            selected_sentences=formatted_sentences
        ).strip()

        return {
            "instruction": REFINE_SYSTEM_INSTRUCTION.strip(),
            "content": formatted_content
        }