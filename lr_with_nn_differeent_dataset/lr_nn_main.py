import pickle
from load_file import unpickle
import numpy as np


if __name__ == "__main__":
    print("Logistic Regression with a NN")

    # Loading the dataset
    dict_datase = unpickle("datasets/data_batch_1")