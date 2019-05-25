from django.core.management.base import BaseCommand
from captcha.services.ml.emnist import load_data_from_db
from captcha.services.ml.models import EnsembleModel, NNModel, DNNModel, CNNModel1, CNNModel2
from captcha.models import Ensemble, NeuralNetworkGroup


class Command(BaseCommand):
    help = "Trains the neural networks"

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Loading EMNIST dataset'))
        x_train, y_train, x_test, y_test = load_data_from_db()
        ensemble = Ensemble()
        ensemble.accuracy = 0
        ensemble.save()
        ensemble = Ensemble.objects.get(id=ensemble.id)

        self.stdout.write(self.style.SUCCESS('Training Neural Network Model'))
        model = NNModel()
        model.set_training_data(x_train, y_train)
        model.set_test_data(x_test, y_test)
        model.build()
        model.train(epochs=9, batch_size=256)
        NeuralNetworkGroup.objects.create(ensemble=ensemble, neural_network=model.save())

        self.stdout.write(self.style.SUCCESS('Training Deep Neural Network Model'))
        model = DNNModel()
        model.set_training_data(x_train, y_train)
        model.set_test_data(x_test, y_test)
        model.build()
        model.train(epochs=6, batch_size=256)
        NeuralNetworkGroup.objects.create(ensemble=ensemble, neural_network=model.save())

        self.stdout.write(self.style.SUCCESS('Training Convolutional Neural Network Model 1'))
        model = CNNModel1()
        model.set_training_data(x_train, y_train)
        model.set_test_data(x_test, y_test)
        model.build()
        model.train(epochs=8, batch_size=256)
        NeuralNetworkGroup.objects.create(ensemble=ensemble, neural_network=model.save())

        self.stdout.write(self.style.SUCCESS('Training Convolutional Neural Network Model 2'))
        model = CNNModel2()
        model.set_training_data(x_train, y_train)
        model.set_test_data(x_test, y_test)
        model.build()
        model.train(epochs=8, batch_size=256)
        NeuralNetworkGroup.objects.create(ensemble=ensemble, neural_network=model.save())

        self.stdout.write(self.style.SUCCESS('Saving Ensemble Model'))

        ensemble.normalize_weights()
        ensemble.save()

        ensemble_model = EnsembleModel(ensemble)

        self.stdout.write(self.style.SUCCESS('Evaluating Ensemble Model'))
        ensemble.accuracy = ensemble_model.evaluate(x_test, y_test)
        ensemble.save()



