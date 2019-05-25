from captcha.models import Captcha
from captcha.models import Digit
import uuid
import PIL
from PIL import Image
import numpy as np
from captcha.models import Digit
from captcha.models import Captcha
from captcha.services.ml.emnist import get_merge_char

BLANK_CHAR = '_'


def get_captcha_value(digits):
    """
    Given the digits returns the captcha value. If the digit is unlabeled then its value becomes "_"
    :param digits: list of Digit objects
    :return: Captcha value
    """
    value = ""
    for digit in digits:
        if digit.value is None:
            value += BLANK_CHAR
        else:
            value += str(digit.ascii_value)
    return value


def create_image(path, images):
    """
    Creates an image by concatenating images
    :param path: path of image
    :param images: a list of images
    :return:
    """
    imgs = [PIL.Image.open(i) for i in images]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))
    # Save image
    imgs_comb = PIL.Image.fromarray(imgs_comb)
    imgs_comb.save(path)


def generate_captcha():
    """
    Generates a captcha
    :return: the generated captcha
    """
    digit1 = Digit.get_random_digit()
    digit2 = Digit.get_random_digit()
    digit3 = Digit.get_random_digit()
    digit4 = Digit.get_random_digit()
    digit5 = Digit.get_random_digit()
    digit6 = Digit.get_random_digit()

    images = [
        str(digit1.image),
        str(digit2.image),
        str(digit3.image),
        str(digit4.image),
        str(digit5.image),
        str(digit6.image)
    ]

    value = get_captcha_value([digit1, digit2, digit3, digit4, digit5, digit6])
    path = "static/captcha/" + str(value) + str(uuid.uuid1()) + ".png"
    create_image(path, images)

    captcha = Captcha(
        value=value,
        image=path,
        digit1=digit1,
        digit2=digit2,
        digit3=digit3,
        digit4=digit4,
        digit5=digit5,
        digit6=digit6
    )

    captcha.save()

    return captcha


def check_captcha_value(captcha, value):
    """
    Checks the captcha value
    :param captcha:
    :param value:
    :return:
    """
    def values_match(char1, char2):
        """
        Checks if the two characters match by checking both merged values and real.
        :param char1: -
        :param char2: -
        :return: True or False
        """
        print(char1)
        print(char2)
        print(get_merge_char(char2))
        return char1 == char2 or char1 == get_merge_char(char2) or char2 == get_merge_char(char1)

    blank_indexes = [index for index, c in enumerate(captcha.value) if c == BLANK_CHAR]

    # Checks length
    if len(value) != 6:
        return False

    # Check Non blank values only
    for index, char in enumerate(captcha.value):
        if index not in blank_indexes:
            if not values_match(char, value[index]):
                return False

    # Update blank values
    for index, char in enumerate(captcha.value):
        if index in blank_indexes:
            try:
                new_value = get_merge_char(value[index])
                update_digit(captcha, index, new_value)
            except Exception as e:
                str(e)
                return False

    # If there are blank values update captcha
    if blank_indexes:
        captcha.value = value
        captcha.save()

    return True


def update_digit(captcha, index, value):
    """
    Updates captcha's digit's value
    :param captcha: Captcha model entity
    :param index: index in value text
    :param value: string
    :return:
    """
    def update_digit_value(digit: Digit, v):
        digit.ascii_value = v
        digit.value = ord(v)
        digit.save()

    if index == 0:
        update_digit_value(captcha.digit1, value)
    elif index == 1:
        update_digit_value(captcha.digit2, value)
    elif index == 2:
        update_digit_value(captcha.digit3, value)
    elif index == 3:
        update_digit_value(captcha.digit4, value)
    elif index == 4:
        update_digit_value(captcha.digit5, value)
    elif index == 5:
        update_digit_value(captcha.digit6, value)



