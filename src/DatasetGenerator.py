from transformers import AutoTokenizer, AutoModel
import torch
class DatasetGenerator:
    def __init__(self, casebuilder):
        self.casebuilder = casebuilder
        self.data = None
        
        self.tokenizer = AutoTokenizer.from_pretrained(casebuilder.bert_model_name)
        self.model = AutoModel.from_pretrained(casebuilder.bert_model_name)
        self.model.eval()

        self.idx = 0

    def set_data(self, data):
        self.data = data

    def feature_selection(self):
        self.data = self.data[['title', 'abstract', 'keywords', 'sections', 'summary']].copy()

    def fix_structure_columns(self):
        self.data['abstract'] = self.data['abstract'].apply(lambda x: ' '.join(x))
        self.data['keywords'] = self.data['keywords'].apply(lambda x: ', '.join(x))
        self.data['summary'] = self.data['summary'].apply(lambda x: ' '.join(x))

    def get_embedding(self, texts, batch_size=256):
        embeddings = []
        try:
            with torch.no_grad():
                for index in range(0, len(texts), batch_size):
                    batch = texts[index:index + batch_size]
                    tokens = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
                    outputs = self.model(**tokens)
                    batch_embeddings = outputs.pooler_output.cpu().numpy()
                    embeddings.extend(batch_embeddings)
        except Exception as e:
            if batch_size > 2:
                return self.get_embedding(texts, batch_size//2)
            else:
                assert False, f"Error: {e}"
        return embeddings

    def get_embeddings_sections(self, sections):
        self.idx += 1
        max_rate = 50
        rate = max_rate * self.idx / len(self.data)
        int_rate = int(rate)
        print(f"\r\t\t  {self.idx}/{len(self.data)} - Loading: {round(rate*2, 2)}% |{"#"*int_rate}{" "*(max_rate-int_rate)}|", end="")

        return [self.get_embedding(section) for section in sections]

    def preprocess(self):
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

        print("*** Preprocessing done!!! ***")

    def get_data(self):
        return self.data

