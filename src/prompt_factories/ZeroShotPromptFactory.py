from src.prompt_factories.AbstractPromptFactory import AbstractPromptFactory


class ZeroShotPromptFactory(AbstractPromptFactory):
    def __init__(self, row=None):
        super().__init__(row=row)

    def get_prompt(self):
        prompt = (
            self.pre_intro()
            + "For the given inputs, generate the outputs.\n\n"
            + self.info(self.row)
        )
        return prompt

    def pre_intro(self):
        return (
            "# TASK:\n"
            "Craft a succinct and straightforward lay summary aimed at an audience without a specialized background in the subject. Leverage the title, abstract, and key sentences provided to ensure your summary encapsulates the essence and findings of the scientific research. Here is a structured breakdown to guide your summary:\n\n"

            "1. **Title Information**: Extract the main topic or issue addressed in the study from the title.\n"
            "2. **Abstract Details**: From the abstract, distill the primary purpose and approach of the research.\n"
            "3. **Key Findings from Selected Sentences**: Integrate the core discoveries or conclusions highlighted in the selected key sentences.\n\n"

            "Your goal is to concisely relay the scientific insights in terms that a layperson can understand, ensuring the summary is educational yet engaging. Maintain a length of 4-6 sentences, and focus strictly on the material provided without inferring additional data or voicing personal interpretations. Strive for clarity by avoiding or simplifying scientific jargon to keep the text accessible and relatable to the general public.\n"
            "---\n"

            "# FORMAT:\n"
            "Follow the following format:\n\n"

            "## INPUT:\n"
            "title: Title of the research paper\n"
            "abstract: Abstract of the research paper, providing a brief summary\n"
            "selected_key_sentences: Key sentences selected from different sections of the paper\n"
            "## OUTPUT:\n"
            "lay_summary: A concise, clear summary of the research paper suitable for a general audience, adhering to specified criteria (length, clarity, focus, technical jargon avoidance)\n\n"

            "---\n"
        )

    def few_info(self, row):
        raise NotImplementedError("ZeroShotPromptFactory does not support few_info")

    def info(self, row):
        return (
            "## INPUT:\n"
            f"title: {row['title']}\n"
            f"abstract: [{' '.join(map(str, row['abstract']))}]\n"
            f"selected_key_sentences: {str(row['rag_sentences'])}\n"
            "## OUTPUT:\n"
            f"lay_summary: \n\n"
        )