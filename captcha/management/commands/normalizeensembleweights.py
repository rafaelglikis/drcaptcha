from django.core.management.base import BaseCommand
from captcha.models import Ensemble


class Command(BaseCommand):
    help = "Normalizes weights for an ensemble"

    def add_arguments(self, parser):
        parser.add_argument('primary-key', type=int)

    def handle(self, *args, **options):
        ensemble = Ensemble.objects.get(id=options['primary-key'])
        ensemble.normalize_weights()
        ensemble.save()
        self.stdout.write(self.style.SUCCESS('Accuracy: {}'.format(ensemble.accuracy)))




