# http://www.cs.toronto.edu/~kriz/cifar.html

# data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
# labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    data = np.array(dict[b'data'])
    labels = np.array(dict[b'labels'])

    ratio = int(data.shape[0] * 0.8) # is 8000
    train_set_x_orig = data[:ratio] # shape is (8000, 3072)

    train_set_x_orig = np.reshape(train_set_x_orig,(ratio, 3, 32, 32)) # the shape is (8000, 3, 32, 32)
    train_set_x_orig = np.transpose(train_set_x_orig, (0, 2, 3, 1)) # the shape is (8000, 32, 32, 3) --> *which is what we need*

    train_set_y_orig = labels[:ratio] # first 8000 labels
    train_set_y_orig[train_set_y_orig > 1] = 1

    test_ratio = int(data.shape[0] * 0.2)  # is 2000
    test_set_x_orig = data[ratio:]  # shape is (2000, 3072)

    test_set_x_orig = np.reshape(test_set_x_orig, (test_ratio, 3, 32, 32))  # the shape is (2000, 3, 32, 32)
    test_set_x_orig = np.transpose(test_set_x_orig,(0, 2, 3, 1))  # the shape is (2000, 32, 32, 3) --> *which is what we need*

    test_set_y_orig = labels[ratio:] # last 2000 labels
    test_set_y_orig[test_set_y_orig > 1] = 1

    classes = np.array(b'airplane', b'non-airplane')

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes






    # for x in dict:
        # print(x)
    # print(len(dict))
    # print(dict[b'batch_label'])

    # return dict