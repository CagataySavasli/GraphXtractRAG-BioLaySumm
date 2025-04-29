from lib.prompt_factories.prompt.abstract_prompt_factory import AbstractPromptFactory


class LaySummaryPromptFactory(AbstractPromptFactory):
    def __init__(self, selected_sentences=None):
        super().__init__(selected_sentences=selected_sentences)

    def get_prompt(self):
        prompt = (
                self.get_instruction()
                + "For the given infos, generate the outputs.\n\n"
                + self.info()
        )
        return prompt

    def get_instruction(self):
        return (
            "# TASK:\n"
            "Craft a succinct and straightforward lay summary aimed at an audience without a specialized background "
            "in the subject. Leverage the key sentences provided to ensure your summary encapsulates the essence and "
            "findings of the scientific research. Here is a structured breakdown to guide your summary:\n\n"

            "**Key Findings from Selected Sentences**: Integrate the core discoveries or conclusions highlighted in "
            "the selected key sentences.\n\n"

            "Your goal is to concisely relay the scientific insights in terms that a layperson can understand, "
            "ensuring the summary is educational yet engaging. Maintain a pragraph with length of 16-24 sentences and "
            "approximetly 400 words, and focus strictly on the material provided without inferring additional data or "
            "voicing personal interpretations. Strive for clarity by avoiding or simplifying scientific jargon to "
            "keep the text accessible and relatable to the general public.\n"
            "---\n"

            "# FORMAT:\n"
            "Follow the following format:\n\n"

            "## INPUT:\n"
            "selected_key_sentences: Key sentences selected from different sections of the paper\n"
            "## OUTPUT:\n"
            "lay_summary: A concise, clear summary of the research paper suitable for a general audience, adhering to "
            "specified criteria (length, clarity, focus, technical jargon avoidance)\n\n"

            "---\n"
        )

    def few_info(self, selected_sentences):
        raise NotImplementedError("ZeroShotPromptFactory does not support few_info")

    def sentence_list_to_string(self, sentence_list):
        sentence_string = ""
        for sentence in sentence_list:
            sentence_string += "- " + sentence + "\n"
        return sentence_string

    def info(self):
        return (
            "## INPUT:\n"
            f"selected_key_sentences: \n"
            f"{self.sentence_list_to_string(self.selected_sentences)}\n" #['rag_sentences']
            "## OUTPUT:\n"
            "Use this JSON schema:\n"
            "{'lay_summary': str} \n"
            "Return: dict"
        )
