from lib.utility.CaseBuilder import CaseBuilder
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import inflect
import re
import torch

class DatasetGenerator:
    def __init__(self):
        """
        Initialize the DatasetGenerator with the given casebuilder.
        """
        self.casebuilder = CaseBuilder()
        self.data = None
        self.idx = 0

        # Initialize the inflect engine
        self.inflect_engine = inflect.engine()

        # Load header clusters and precompute a filter dictionary for clustering headings
        self.header_clusters = pd.read_json('src/dataset/header_clusters.json')
        self._header_cluster_length = len(self.header_clusters)
        self._filter_dict = self._build_filter_dict()

        # Initialize tokenizer and model from the specified BERT model name
        self.tokenizer = AutoTokenizer.from_pretrained(self.casebuilder.bert_model_name)
        self.model = AutoModel.from_pretrained(self.casebuilder.bert_model_name)
        self.model.eval()

    def _build_filter_dict(self) -> dict:
        """
        Build a dictionary mapping each token in the 'filter' lists to its cluster value.
        If a token appears in multiple rows, the first occurrence is used.
        """
        filter_dict = {}
        for _, row in self.header_clusters.iterrows():
            for token in row['filter']:
                if token not in filter_dict:
                    filter_dict[token] = row['cluster_value']
        return filter_dict

    def set_data(self, data: pd.DataFrame) -> None:
        """Set the dataset to be processed."""
        print("Setting data")
        self.data = data

    def feature_selection(self) -> None:
        """Select only the relevant columns from the data."""
        self.data = self.data[['headings', 'title', 'abstract', 'keywords', 'sections', 'summary']].copy()

    def fix_structure_columns(self) -> None:
        """
        Restructure the 'sections' and 'headings' columns by incorporating the title and abstract.
        Also computes heading clusters and joins list elements into strings where needed.
        """
        def restructure_row(row):
            # Combine title and abstract with existing sections and headings:
            # - For sections, wrap title in a list, then add abstract and the original sections.
            # - For headings, prepend 'Title' and 'Abstract' to the original headings.
            row['sections'] = [[row['title']]] + [row['abstract']] + row['sections']
            row['headings'] = ['Title', 'Abstract'] + row['headings']
            return row

        self.data = self.data.apply(restructure_row, axis=1)
        # Compute heading clusters and assign them as sentence clusters.
        self.data['heading_clusters'] = self.data.apply(self._get_sentence_clusters, axis=1)
        # Convert list columns into strings where applicable.
        self.data['abstract'] = self.data['abstract'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        self.data['keywords'] = self.data['keywords'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        self.data['summary'] = self.data['summary'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)

    def get_embedding(self, texts, batch_size: int = 256):
        """
        Generate embeddings for a list of texts using batching.
        If a memory error occurs, recursively reduce the batch size.
        """
        embeddings = []
        try:
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    tokens = self.tokenizer(batch, return_tensors='pt', padding=True,
                                            truncation=True, max_length=512)
                    outputs = self.model(**tokens)
                    batch_embeddings = outputs.pooler_output.cpu().numpy()
                    embeddings.extend(batch_embeddings)
        except Exception as e:
            if batch_size > 2:
                return self.get_embedding(texts, batch_size=batch_size // 2)
            else:
                raise RuntimeError(f"Error during embedding generation: {e}")
        return embeddings

    def get_embeddings_sections(self, sections):
        """
        Generate embeddings for each section in a list of sections while showing a progress bar.
        """
        self.idx += 1
        progress_ratio = self.idx / len(self.data)
        max_rate = 50
        filled_length = int(max_rate * progress_ratio)
        percent = round(progress_ratio * 100, 2)
        print(f"\r\t{self.idx}/{len(self.data)} - Loading: {percent}% |{'#' * filled_length}{' ' * (max_rate - filled_length)}|", end="")

        return [self.get_embedding(section) for section in sections]

    def preprocess(self) -> None:
        """Run the complete preprocessing pipeline on the dataset."""
        print("Preprocessing data")
        self.feature_selection()
        self.fix_structure_columns()

        print("-> Get Embeddings")
        print("\t -> Title")
        self.data['title_embedding'] = self.get_embedding(self.data['title'].tolist())

        print("\t -> Abstract")
        self.data['abstract_embedding'] = self.get_embedding(self.data['abstract'].tolist())

        print("\t -> Keywords")
        self.data['keywords_embedding'] = self.get_embedding(self.data['keywords'].tolist())

        print("\t -> Sections")
        self.data['sections_embedding'] = self.data['sections'].apply(self.get_embeddings_sections)

        print("\n*** Preprocessing done!!! ***")

    def get_data(self) -> pd.DataFrame:
        """Return the processed dataset."""
        return self.data

    def convert_plural_to_singular(self, text: str) -> str:
        """
        Convert plural words in the input text to their singular form.
        """
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        converted_tokens = []
        for token in tokens:
            if token.isalpha():
                singular = self.inflect_engine.singular_noun(token)
                converted_tokens.append(singular if singular else token)
            else:
                converted_tokens.append(token)
        result = " ".join(converted_tokens)
        # Remove extra spaces before punctuation marks.
        result = re.sub(r'\s([.,!?;:])', r'\1', result)
        return result

    def fix_whitespaces(self, text: str) -> str:
        """
        Normalize whitespace in the text by replacing multiple spaces with a single space and trimming edges.
        """
        text = text.replace('\ufeff', '')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def clean_text(self, text: str) -> str:
        """
        Clean text by normalizing whitespace, converting to lowercase,
        and converting plural words to singular.
        """
        text = self.fix_whitespaces(text)
        text = text.lower()
        return self.convert_plural_to_singular(text)

    def cluster_headings(self, headings):
        """
        Cluster the list of headings by cleaning each one and mapping it to a cluster value.
        """
        cleaned_headings = [self.clean_text(heading) for heading in headings]
        clusters = []
        for heading in cleaned_headings:
            # Retrieve the cluster value using the precomputed filter dictionary;
            # use default value if not found.
            cluster_value = self._filter_dict.get(heading, self._header_cluster_length)
            clusters.append(cluster_value)
        # Normalize the cluster values.
        return [val / self._header_cluster_length for val in clusters]

    def _get_sentence_clusters(self, row):
        """
        For a given row, assign each sentence in every section the corresponding heading cluster.
        """
        heading_clusters = self.cluster_headings(row['headings'])
        sentence_clusters = []
        for i, section in enumerate(row['sections']):
            sentence_clusters.extend([heading_clusters[i]] * len(section))
        return sentence_clusters
