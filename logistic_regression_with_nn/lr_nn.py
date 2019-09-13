###### OBJECTIVES ######
# 1. Construct the Coursera Logistic Regression with a NN
#       - Add task by task
#       - Learn the funnctions structures
#       - Explain the relatiom with the theory
#       - Check outputs on the fly
########################

# numpy is the fundamental package for scientific computing with Python.
# h5py is a common package to interact with a dataset that is stored on an H5 file.
# matplotlib is a famous library to plot graphs in Python.
# PIL and scipy are used here to test your model with your own picture at the end.
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset





# RUN
if __name__ == "__main__":
    print("Logistic Regression with a NN")