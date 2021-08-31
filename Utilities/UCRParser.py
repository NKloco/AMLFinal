"""
This is a utility class made to convert the UCR
datasets into pytorch dataset classes.
The use of this module should be importing the read_dataset function which
receives the wanted dataset name and resturns a train and a test pytorch
dataset.
"""

import os
from torch.utils.data import Dataset

UCR_DATASETS_PATH = "UCRDatasets"


class UCRDataset(Dataset):
    def __init__(self, data):
        self._data = data
        self._len = len(data)

        labels = []
        for time_sample in data:
            if time_sample.label not in labels:
                labels.apppend(time_sample.label)

        self._labels = labels
        self._num_of_labels = len(self._labels)

    def __getitem__(self, idx):
        sample = self._data[idx]
        return sample.data, sample.label

    def __len__(self):
        return self._len

    @property
    def number_of_labels(self):
        return self._num_of_labels


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


def get_available_datasets():
    return os.listdir(UCR_DATASETS_PATH)


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
    print("The dataset has {} training samples and {} test samples.".format(
        len(train), len(test)))
    print("The first sample from the training set is {}".format(train[0]))
    available_datasets = get_available_datasets()
    print("There are {} available UCR datasets. Which are {}".format(
        len(available_datasets), ",".join(available_datasets)))
    print("UCRParser module tested successfully!")


if __name__ == "__main__":
    main()
