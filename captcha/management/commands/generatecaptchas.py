from tqdm import tqdm
from django.core.management.base import BaseCommand
from captcha.services.captcha import *


class Command(BaseCommand):
    help = "Generates captchas by concatenating alphanumeric images from the database"

    def add_arguments(self, parser):
        parser.add_argument('count', type=int)

    def handle(self, *args, **options):
        for _ in tqdm(range(0, options['count'])):
            generate_captcha()
        self.stdout.write(self.style.SUCCESS('Succesfully created {} captcha(s)'.format(options['count'])))
