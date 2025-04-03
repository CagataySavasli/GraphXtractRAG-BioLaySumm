#%%
from lib.gyms.GeminiGYM import GeminiGYM
from lib.utility.CaseBuilder import CaseBuilder
from lib.utility.ResultCalculator import ResultCalculator
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
#%%
rag_strategy = sys.argv[1] #"similarityRAG"
print("RAG Strategy: ", rag_strategy)
#%%
case_builder = CaseBuilder(rag_strategy=rag_strategy)
result_calculator = ResultCalculator()
#%%
print("Dataset Name: ", case_builder.dataset_name)
df_train = pd.read_json(f'dataset/processed/{case_builder.dataset_name}/train.json').reset_index(drop=True)
df_test = pd.read_json(f'dataset/processed/{case_builder.dataset_name}/test.json').reset_index(drop=True)
print("Train Shape: ", df_train.shape)
print("Test Shape: ", df_test.shape)
#%%
gemini_gym = GeminiGYM()
gemini_gym.set_train_data(df_train.copy())
print("GeminiGYM Created")
#%%
display_name = f"{case_builder.rag_strategy}_{case_builder.dataset_name}"
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
generated_text_df.to_csv(f'outputs/generated_texts/{case_builder.rag_strategy}_{case_builder.dataset_name}_genrated_text.csv', index=False)
score_df = pd.DataFrame([score_dict])
score_df.to_csv(f'outputs/scores/{case_builder.rag_strategy}_{case_builder.dataset_name}_score.csv', index=False)
print("Results Saved")
#%%
result_calculator.display_rank(score_dict)
#%%
