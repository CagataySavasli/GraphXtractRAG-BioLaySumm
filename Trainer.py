#%%
from lib.gyms.GeminiGYM import GeminiGYM
from lib.utility.CaseBuilder import CaseBuilder
from lib.utility.ResultCalculator import ResultCalculator
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
#%%
rag_strategy = sys.argv[1] #"similarityRAG"
#%%
case_builder = CaseBuilder(rag_strategy=rag_strategy)
result_calculator = ResultCalculator()
#%%
# Load dataset and split
df = pd.read_json(f'dataset/processed/{case_builder.dataset_name}/val.json')
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
#%%
gemini_gym = GeminiGYM()
gemini_gym.set_train_data(df_train.copy())
#%%
gemini_gym.fine_tune(display_name=rag_strategy,
                     epoch_count=20,
                     batch_size=8,
                     learning_rate=5e-4)
#%%
gemini_gym.set_test_data(df_test.copy())
test_results = gemini_gym.evaluate()
#%%
predicted = test_results['pred']
actual = test_results['true']
#%%
score_dict = result_calculator.evaluate(predicted, actual)
#%%
generated_text_df = pd.DataFrame({'generated': predicted, 'actual': actual})
generated_text_df.to_csv(f'outputs/generated_texts/{case_builder.rag_strategy}_{case_builder.dataset_name}_genrated_text.csv', index=False)
score_df = pd.DataFrame([score_dict])
score_df.to_csv(f'outputs/scores/{case_builder.rag_strategy}_{case_builder.dataset_name}_score.csv', index=False)
#%%
result_calculator.display_rank(score_dict)
#%%
