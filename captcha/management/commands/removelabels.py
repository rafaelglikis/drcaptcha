from django.core.management.base import BaseCommand
from tqdm import tqdm

from captcha.models import Digit


class Command(BaseCommand):
    help = "Removes labels from alphanumeric images from the database"

    def add_arguments(self, parser):
        parser.add_argument('count', type=int)

    def handle(self, *args, **options):
        digits = Digit.get_random_digits(options['count'])

        for digit in tqdm(digits):
            digit.value = None
            digit.ascii_value = None
            digit.save()

        self.stdout.write(self.style.SUCCESS('Succesfully unlabeled {} digit(s)'.format(len(digits))))
