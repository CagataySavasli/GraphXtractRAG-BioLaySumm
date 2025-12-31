from datasets import load_dataset, load_from_disk

HF_DATA_PATHS = {
    "elife" : "BioLaySumm/BioLaySumm2025-eLife",
    "plos" : "BioLaySumm/BioLaySumm2025-PLOS"
}

class DataLoaderReloader():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        self.local_save_path = f"./dataset/raw/{dataset_name}/biolaysumm_{dataset_name}_local_dataset"
        self.hf_data_path = HF_DATA_PATHS[dataset_name]

    def download(self):
        ds = load_dataset(self.hf_data_path)
        ds.save_to_disk(self.local_save_path)

    def reload(self):
        ds = load_from_disk(self.local_save_path)
        return ds
