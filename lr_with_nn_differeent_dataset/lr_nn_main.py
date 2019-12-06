import pickle
from load_file import unpickle
import numpy as np


if __name__ == "__main__":
    print("Logistic Regression with a NN")

    # Loading the dataset
    dict_datase = unpickle("datasets/data_batch_1")
    for x in dict_datase:
        print(x)
    # print(len(dict_datase))
    print(dict_datase[b'data'][0].size)