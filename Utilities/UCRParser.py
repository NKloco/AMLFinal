"""
This is a utility class made to convert the UCR datasets into a class that contains the train, test and easy access to values and 
"""

import os
from torch.utils.data import Dataset

UCR_DATASETS_PATH = "UCRDatasets"

class UCRDataset(Dataset):
    def __init__(self, data):
        self._data = data
        self._len = len(data)

    def __getitem__(self, idx):
        sample = self._data[idx]
        return sample.data, sample.label

    def __len__(self):
        return self._len

class Sample(object):
    def __init__(self, sample_text):
        data = sample_text.split(",")
        self._label = int(data[0])
        self._sample_data = [float(sample) for sample in data[1:]]
    
    @property
    def data(self):
        return self._sample_data
    
    @property
    def label(self):
        return self._label

def get_single_dataset(path):
    with open(path, "r") as data_file:
        data = data_file.readlines()
    
    data_samples = [Sample(sample) for sample in data]
    dataset = UCRDataset(data_samples)

    return dataset

def read_dataset(dataset_name):
    print("Loading the {} dataset...".format(dataset_name))
    test_file_name = "{}_TEST".format(dataset_name)
    test_path = os.path.join(UCR_DATASETS_PATH, dataset_name, test_file_name)

    train_file_name = "{}_TRAIN".format(dataset_name)
    train_path = os.path.join(UCR_DATASETS_PATH, dataset_name, train_file_name)

    test_dataset = get_single_dataset(test_path)
    train_dataset = get_single_dataset(train_path)

    print("The dataset {} was loaded.".format(dataset_name))
    return train_dataset, test_dataset
    

def main():
    print("Testing the UCRParser utility module...")
    train, test = read_dataset("ArrowHead")
    print("The dataset has {} training samples and {} test samples.".format(len(train), len(test)))
    print("The first sample from the training set is {}".format(train[0]))
    print("UCRParser module tested successfully!")

if __name__ == "__main__":
    main()