from datasets import load_dataset


class Dataset:
    """
    Dataset class to load the chunked dataset 
    """
    def __init__(self, dataset_path: str, split: str) -> None:
        self.dataset_path = dataset_path
        self.split = split

    def get_dataset(self) -> list:
        dataset = load_dataset(path="json", data_files=self.dataset_path, split=self.split)
        dataset = dataset.map(lambda x: {
            "id": f'{x["document_id"]}-{x["chunk_id"]}',
            "text": x["chunk"],
            "metadata": {
                "laptop_id": x["laptop_id"],
                "text": x["chunk"],
            }
        })
        # drop uneeded columns
        dataset = dataset.remove_columns(["chunk_id","chunk","document_id"])
        return dataset
