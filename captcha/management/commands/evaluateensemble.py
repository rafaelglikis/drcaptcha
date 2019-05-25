from django.core.management.base import BaseCommand
from captcha.services.ml.emnist import load_data_from_db
from captcha.services.ml.models import EnsembleModel
from captcha.models import Ensemble


class Command(BaseCommand):
    help = "Evaluates ensemble models."

    def add_arguments(self, parser):
        parser.add_argument('primary-key', type=int)

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Loading EMNIST dataset'))
        _, _, x_test, y_test = load_data_from_db()
        ensemble = Ensemble.objects.get(id=options['primary-key'])
        ensemble_model = EnsembleModel(ensemble)
        ensemble.accuracy = ensemble_model.evaluate(x_test, y_test)
        ensemble.save()
        self.stdout.write(self.style.SUCCESS('Accuracy: {}'.format(ensemble.accuracy)))

