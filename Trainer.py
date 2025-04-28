#%%
from lib.gyms.GeminiGYM import GeminiGYM
from lib.utility.case_builder import CaseBuilder
from lib.utility.result_calculator import ResultCalculator
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import json
import sys
#%%
dataset_name = sys.argv[1] #"elife"
rag_strategy = sys.argv[2] #"similarityRAG"
rag_n = int(sys.argv[3]) # 30
print("RAG Strategy: ", rag_strategy)
print("RAG N: ", rag_n)
#%%
case_builder = CaseBuilder(dataset_name=dataset_name,
                           rag_strategy=rag_strategy,
                           rag_n=rag_n)
result_calculator = ResultCalculator()
#%%
print("Dataset Name: ", case_builder.dataset_name)
df_train = pd.read_json(f'dataset/processed/{case_builder.dataset_name}/train.json').reset_index(drop=True)
df_test = pd.read_json(f'dataset/processed/{case_builder.dataset_name}/test.json').reset_index(drop=True)

print("Train Shape: ", df_train.shape)
print("Test Shape: ", df_test.shape)
#%%
def get_similarity(row):
    nodes = [x for y in row['sections_embedding'] for x in y]

    # Tüm cosine similarity değerlerini hesapla
    similarities = cosine_similarity(nodes)

    # Ortalama cosine similarity hesaplama
    avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])

    return avg_similarity, similarities

df_train[['avg_similarity', 'similarities']] = df_train.apply(get_similarity, axis=1, result_type='expand')
df_test[['avg_similarity', 'similarities']] = df_test.apply(get_similarity, axis=1, result_type='expand')

#%%
gemini_gym = GeminiGYM()
gemini_gym.set_train_data(df_train.copy())
print("GeminiGYM Created")
#%%
display_name = f"{case_builder.rag_strategy}_{case_builder.rag_n}_{case_builder.dataset_name}"
gemini_gym.fine_tune(display_name=display_name,
                     epoch_count=20)
print("Fine Tuning Completed")
#%%
gemini_gym.set_test_data(df_test.copy())
test_results = gemini_gym.evaluate()
print("Evaluation Completed")
#%%
predicted = test_results['pred']
actual = test_results['true']
#%%
score_dict = result_calculator.evaluate(predicted, actual)
print("Score Calculated")
#%%
generated_text_df = pd.DataFrame({'generated': predicted, 'actual': actual})
generated_text_df.to_csv(f'outputs/generated_texts/{display_name}_genrated_text.csv', index=False)
score_df = pd.DataFrame([score_dict])
score_df.to_csv(f'outputs/scores/{display_name}_score.csv', index=False)
print("Results Saved")
#%%
result_calculator.display_rank(score_dict)
#%%
