from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import string
import nltk
import torch


class Preprocessor():
    def __init__(self):
        # NLTK kontrolü
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        # --- GPU/Cihaz Ayarı ---
        if torch.cuda.is_available():
            self.device = 'cuda'
            print(f"✅ Model GPU (NVIDIA CUDA) üzerinde çalışacak.")
        elif torch.backends.mps.is_available():
            self.device = 'mps'
            print("✅ Model GPU (Apple Silicon MPS) üzerinde çalışacak.")
        else:
            self.device = 'cpu'
            print("⚠️ GPU bulunamadı. Model CPU üzerinde çalışacak.")

        # self.device = 'cpu'
        # print("⚠️ GPU bulunamadı. Model CPU üzerinde çalışacak.")
        print(f"Preprocessor Device: {self.device}")

        # Modeli yükle
        self.model = SentenceTransformer('NeuML/pubmedbert-base-embeddings', device=self.device)#sentence-transformers/allenai-specter

    def get_embedding_one_sentence(self, sentence: str) -> list:
        embedding = self.model.encode(
            [sentence],
            convert_to_numpy=True,
            device=self.device,
            show_progress_bar=False
        )
        # Parquet hatasını önlemek için .tolist() kullanıyoruz
        return embedding[0].tolist()

    def get_embedding_one_section(self, sections: list[str]) -> list:
        """Tek bir makalenin cümlelerini alır."""
        if not sections:
            return []

        embeddings = self.model.encode(
            sections,
            batch_size=1,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=self.device
        )
        # Parquet hatasını önlemek için .tolist() kullanıyoruz
        return embeddings.tolist()

    def get_embedding_multi_sections(self, sections: list) -> list:
        """
        Hem .apply() (tek makale) hem de Batch (çok makale) için çalışır.
        Dönüş değeri her zaman Python Listesidir (Parquet uyumlu).
        """
        if not sections:
            return []

        # DURUM 1: .apply() ile tek bir makale (List[str]) gelmişse
        if isinstance(sections[0], str):
            return self.get_embedding_one_section(sections)

        # DURUM 2: Batch (List[List[str]]) gelmişse (Flattening Mantığı)
        all_sections_flat = [sec for sublist in sections for sec in sublist]
        lengths = [len(x) for x in sections]

        if not all_sections_flat:
            return []

        # Modeli çalıştır
        all_embeddings = self.model.encode(
            all_sections_flat,
            batch_size=1,
            convert_to_numpy=True,  # Tensor yerine Numpy alıyoruz
            show_progress_bar=False,
            device=self.device
        )

        # Tekrar eski yapısına (Nested List) dönüştür
        nested_embeddings = []
        cursor = 0
        for length in lengths:
            # Slice alıp listeye çeviriyoruz (.tolist() KRİTİK)
            segment = all_embeddings[cursor: cursor + length]
            nested_embeddings.append(segment.tolist())
            cursor += length

        return nested_embeddings

    def split_into_sentences(self, text: str) -> list[str]:
        if not text or not isinstance(text, str):
            return []
        try:
            return [s.strip() for s in nltk.sent_tokenize(text)]
        except:
            return text.split('. ')

    def section_headings_fixer(self, section_headings: list[str]) -> list[str]:
        if not isinstance(section_headings, list) or len(section_headings) == 0:
            return []

        # (Orijinal mantığınız korunmuştur)
        merged_list = []
        i = 0
        n = len(section_headings)
        punctuation_set = set(string.punctuation)

        while i < n:
            current_phrase = section_headings[i]
            if current_phrase is None: current_phrase = ""

            while (i + 1 < n):
                next_word = section_headings[i + 1]
                if next_word is None: next_word = ""

                clean_current = current_phrase.strip()
                clean_next = next_word.strip()

                if not clean_next:
                    i += 1
                    continue

                first_char_next = clean_next[0] if clean_next else ""
                should_merge = False

                # Merge logic
                if first_char_next.islower() and clean_current != "Introduction":
                    should_merge = True
                elif first_char_next.isdigit() or first_char_next in punctuation_set:
                    should_merge = True
                elif clean_current.lower().endswith((' and', ' of')):
                    should_merge = True
                elif clean_current and clean_current[-1] in punctuation_set:
                    should_merge = True
                elif clean_current.isdigit():
                    should_merge = True
                elif clean_current == 'Main' and clean_next.lower().startswith('text'):
                    should_merge = True
                elif clean_current.startswith('Experiment') and clean_next.startswith('–'):
                    should_merge = True
                elif current_phrase.endswith("–"):
                    should_merge = True

                if should_merge:
                    current_phrase += " " + next_word
                    i += 1
                else:
                    break

            merged_list.append(current_phrase)
            i += 1

        return merged_list

    def filter_mismatched_sections(self, df: pd.DataFrame) -> pd.DataFrame:
        def check_lengths(row):
            try:
                s = row.get('sections', [])
                h = row.get('section_headings', [])
                return len(s) == len(h)
            except:
                return False

        keep_mask = df.apply(check_lengths, axis=1)
        return df.loc[keep_mask].reset_index(drop=True)