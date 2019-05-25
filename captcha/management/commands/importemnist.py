from django.core.management.base import BaseCommand
import uuid
from captcha.services.ml.emnist import *
from tqdm import tqdm


class Command(BaseCommand):
    help = "Imports emnist dataset into the database"

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.NOTICE('Loading mat file . . .'))
        (x_train, y_train), (x_test, y_test), mapping, nb_classes = load_data("dataset/emnist-balanced.mat")

        self.stdout.write(self.style.NOTICE('Generating Images . . .'))
        digits = []
        for img_data, img_label in zip(tqdm(x_train, desc="Training Data"), y_train):
            path = "static/digits/" + str(img_label) + str(uuid.uuid1()) + ".png"
            create_image(img_data, path)
            digit = Digit(value=img_label[0], ascii_value=chr(mapping[img_label[0]]), dataset='train', bytes=img_data, image=path)
            digits.append(digit)

        Digit.objects.bulk_create(digits)

        digits = []
        for img_data, img_label in zip(tqdm(x_test, desc="Test Data"), y_test):
            path = "static/digits/" + str(img_label) + str(uuid.uuid1()) + ".png"
            create_image(img_data, path)
            digit = Digit(value=img_label[0], ascii_value=chr(mapping[img_label[0]]), dataset='test', bytes=img_data, image=path)
            digits.append(digit)

        Digit.objects.bulk_create(digits)

        self.stdout.write(self.style.SUCCESS('EMNIST dataset imported'))


