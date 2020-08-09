import numpy as np
import idx2numpy as idx2np
import matplotlib.pyplot as plt



def get_tests():
    tests = idx2np.convert_from_file("train-images.idx3-ubyte")
    labels = idx2np.convert_from_file("train-labels.idx1-ubyte")
    indices = np.arange(np.shape(tests)[0])
    np.random.shuffle(indices)
    tests = tests[indices]
    labels = labels[indices]
    return tests, labels


tests, labels = get_tests()


def display_image(img,label):
    print("label:", label)
    plt.imshow(img, cmap="gray", interpolation="nearest")
    plt.show()

