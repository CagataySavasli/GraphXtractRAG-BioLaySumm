#%%
from lib.utility.CaseBuilder import CaseBuilder
from lib.utility.ResultCalculator import ResultCalculator
from lib.gyms.SelectorPipelineGYM import SelectorPipelineGYM
import pandas as pd
import sys
#%%
dataset_name = sys.argv[1] #"elife"
rag_n = int(sys.argv[2]) # 30
print("Dataset Name: ", dataset_name)
print("RAG N: ", rag_n)
#%%
#%%
case_builder = CaseBuilder(
    dataset_name=dataset_name,
    rag_strategy="graphxtract",
    rag_n=rag_n,
)
result_calculator = ResultCalculator()
selector_strategy = "MIX"
#%%
print("Dataset Name: ", case_builder.dataset_name)
df_train = pd.read_json(f'dataset/processed/{case_builder.dataset_name}/train.json').reset_index(drop=True)
df_test = pd.read_json(f'dataset/processed/{case_builder.dataset_name}/test.json').reset_index(drop=True)

print("Train Shape: ", df_train.shape)
print("Test Shape: ", df_test.shape)
#%%
selector_gym = SelectorPipelineGYM(selector_strategy, df_train, df_test)

selector_gym.train(20)
selector_gym.save_selector()
selector_gym.plot_training_loss()

predicted_summaries, referance_summaries = selector_gym.test()
score_dict = result_calculator.evaluate(predicted_summaries, referance_summaries)
print("Score Calculated")
#%%
display_name = f"Selector_{case_builder.dataset_name}_{case_builder.rag_n}_{selector_strategy}"
generated_text_df = pd.DataFrame({'generated': predicted_summaries, 'actual': referance_summaries})
generated_text_df.to_csv(f'outputs/generated_texts/{display_name}_genrated_text.csv', index=False)
score_df = pd.DataFrame([score_dict])
score_df.to_csv(f'outputs/scores/{display_name}_score.csv', index=False)
print("Results Saved")
#%%
result_calculator.display_rank(score_dict)
#%%
