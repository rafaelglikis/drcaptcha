import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
import scipy.misc as smp
from captcha.models import Digit


def load_data(mat_file_path, width=28, height=28, max_=None):
    """
    Load data in from .mat file as specified by the paper.

    :param mat_file_path: path to the .mat, should be in sample/
    :param width: specified width
    :param height: specified height
    :param max_: the max number of samples to load
    :return: A tuple of training and test data, and the mapping for class code to ascii value, in the following format:
        ((training_images, training_labels), (testing_images, testing_labels), mapping)
    """

    # Load convoluted list structure form loadmat
    mat = loadmat(mat_file_path)

    # Load char mapping
    mapping = {kv[0]: kv[1:][0] for kv in mat['dataset'][0][0][2]}

    # Load training data
    if max_ is None:
        max_ = len(mat['dataset'][0][0][0][0][0][0])
    training_images = mat['dataset'][0][0][0][0][0][0][:max_].reshape(max_, height, width, 1)
    training_labels = mat['dataset'][0][0][0][0][0][1][:max_]

    # Load testing data
    if max_ is None:
        max_ = len(mat['dataset'][0][0][1][0][0][0])
    else:
        max_ = int(max_ / 6)
    testing_images = mat['dataset'][0][0][1][0][0][0][:max_].reshape(max_, height, width, 1)
    testing_labels = mat['dataset'][0][0][1][0][0][1][:max_]

    # Reshape training data to be valid
    for i in tqdm(range(len(training_images)), desc="Training Data"):
        training_images[i] = rotate(training_images[i])

    # Reshape testing data to be valid
    for i in tqdm(range(len(testing_images)), desc="Test Data"):
        testing_images[i] = rotate(testing_images[i])

    # Convert type to float32
    training_images = training_images.astype('float32')
    testing_images = testing_images.astype('float32')

    # Normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255
    nb_classes = len(mapping)

    return (training_images, training_labels), (testing_images, testing_labels), mapping, nb_classes


def rotate(img):
    """
    Used to rotate images (for some reason they are transposed on read-in)
    :param img: image in matrix format
    :return: rotated img
    """
    flipped = np.fliplr(img)
    return np.rot90(flipped)


def display(img, threshold=0.5):
    """
    Displays the value
    :param img: in matrix format
    :param threshold: -
    :return:
    """
    render = ''
    for row in img:
        for col in row:
            if col > threshold:
                render += '#'
            else:
                render += '.'
        render += '\n'
    return render


def create_image(image_data, path):
    """
    Creates an image from image data.
    :param image_data: image as a matrix
    :param path: path to image
    :return:
    """
    image_data = image_data.squeeze()
    img = smp.toimage(image_data)
    img.save(path)


def valid_char(char):
    """
    Checks if the current character is valid.
    :param char: -
    :return: True or False
    """
    ascii_val = int(ord(char))
    return (65 <= ascii_val <= 90) or (97 <= ascii_val <= 122) or (48 <= ascii_val <= 57)


def get_merge_char(char):
    """
    Gets the merge representation of the character as described in the paper.
    :param char: -
    :return:
    """
    if not valid_char(char):
        raise Exception("Not a valid character.")

    merge_letters = ['c', 'i', 'j', 'k', 'l', 'm', 'o', 'p', 's', 'u', 'v', 'w', 'x', 'y', 'z']
    if char in merge_letters:
        return chr(ord(char) - 32)

    return char


def get_mapping():
    """
    :return: The value->ascii_value mapping for the emnist-balanced dataset.
    """
    return {
        0: 48, 1: 49, 2: 50, 3: 51, 4: 52, 5: 53, 6: 54, 7: 55, 8: 56, 9: 57, 10: 65,
        11: 66, 12: 67, 13: 68, 14: 69, 15: 70, 16: 71, 17: 72, 18: 73, 19: 74, 20: 75,
        21: 76, 22: 77, 23: 78, 24: 79, 25: 80, 26: 81, 27: 82, 28: 83, 29: 84, 30: 85,
        31: 86, 32: 87, 33: 88, 34: 89, 35: 90, 36: 97, 37: 98, 38: 100, 39: 101, 40: 102,
        41: 103, 42: 104, 43: 110, 44: 113, 45: 114, 46: 116
    }


def load_data_from_db(count=None):
    """
    Loads labeled data from database and splits to training and test set.
    :param count: number of images to load
    :return:  A tuple of training and test data, in the following format:
        (training_images, training_labels, testing_images, testing_labels)
    """

    if Digit.objects.all().filter(dataset="train").exclude(value__isnull=True).count() < 47:
        raise Exception("Not enough data!")
    if count is None:
        train_digits = Digit.objects.all().filter(dataset="train").exclude(value__isnull=True)
    else:
        train_digits = Digit.objects.all().filter(dataset="train").exclude(value__isnull=True)[:count]

    if Digit.objects.all().filter(dataset="test").exclude(value__isnull=True).count() < 47:
        raise Exception("Not enough data!")
    if count is None:
        test_digits = Digit.objects.all().filter(dataset="test").exclude(value__isnull=True)
    else:
        test_digits = Digit.objects.all().filter(dataset="test").exclude(value__isnull=True)[:count]

    rows, cols = 28, 28
    x_train = np.array([[i.bytes] for i in train_digits])
    y_train = [i.value for i in train_digits]
    x_test = np.array([[i.bytes] for i in test_digits])
    y_test = [i.value for i in test_digits]

    x_train = x_train.reshape(x_train.shape[0], rows, cols, 1)
    x_test = x_test.reshape(x_test.shape[0], rows, cols, 1)

    return x_train, y_train, x_test, y_test


def load_test_data_from_db(count=None):
    """
    Loads labeled data from database.
    :param count: number of images to load
    :return: A tuple of data, in the following format:
        (images, labels)
    """
    if Digit.objects.all().filter(dataset="test").exclude(value__isnull=True).count() < 47:
        raise Exception("Not enough data!")
    if count is None:
        digits = Digit.objects.all().filter(dataset="test").exclude(value__isnull=True)
    else:
        digits = Digit.objects.all().filter(dataset="test").exclude(value__isnull=True)[:count]

    rows, cols = 28, 28
    data = np.array([[i.bytes] for i in digits])
    labels = [i.value for i in digits]

    data = data.reshape(data.shape[0], rows, cols, 1)

    return data, labels
