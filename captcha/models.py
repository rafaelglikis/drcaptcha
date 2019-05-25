from django.db import models
from django.db.models import Max
from picklefield.fields import PickledObjectField
import random


class Digit(models.Model):
    ascii_value = models.CharField(max_length=5, blank=True, null=True)
    value = models.SmallIntegerField(blank=True, null=True)
    dataset = models.CharField(max_length=15)
    bytes = PickledObjectField(default=None)
    image = models.ImageField(null=True, blank=True)

    def __str__(self):
        return str(self.ascii_value)

    @classmethod
    def get_random_digit(cls):
        max_id = Digit.objects.all().aggregate(max_id=Max("id"))['max_id']
        while True:
            pk = random.randint(1, max_id)
            digit = Digit.objects.filter(pk=pk).first()
            if digit:
                return digit

    @classmethod
    def get_random_digits(cls, count):
        digits = []
        for i in range(0, count):
            digits.append(Digit.get_random_digit())
        return digits


class Captcha(models.Model):
    value = models.CharField(max_length=7)
    image = models.ImageField(null=True, blank=True)
    digit1 = models.ForeignKey(Digit, null=True, related_name='digit1', on_delete=models.CASCADE)
    digit2 = models.ForeignKey(Digit, null=True, related_name='digit2', on_delete=models.CASCADE)
    digit3 = models.ForeignKey(Digit, null=True, related_name='digit3', on_delete=models.CASCADE)
    digit4 = models.ForeignKey(Digit, null=True, related_name='digit4', on_delete=models.CASCADE)
    digit5 = models.ForeignKey(Digit, null=True, related_name='digit5', on_delete=models.CASCADE)
    digit6 = models.ForeignKey(Digit, null=True, related_name='digit6', on_delete=models.CASCADE)

    @classmethod
    def get_random(cls):
        max_id = Captcha.objects.all().aggregate(max_id=Max("id"))['max_id']
        while True:
            pk = random.randint(1, max_id)
            captcha = Captcha.objects.filter(pk=pk).first()
            if captcha:
                return captcha

    def __str__(self):
        return str(self.value)


class NeuralNetwork(models.Model):
    path = models.CharField(max_length=63)
    create_date = models.DateTimeField(auto_now_add=True, blank=True)

    epochs = models.IntegerField(default=1)
    batch_size = models.IntegerField(default=1)
    accuracy = models.FloatField()
    training_set_size = models.IntegerField(default=0)
    test_set_size = models.IntegerField(default=0)

    model_summary_image = models.ImageField(null=True, blank=True)
    loss_graph = models.ImageField(null=True, blank=True)
    accuracy_graph = models.ImageField(null=True, blank=True)
    confusion_matrix = models.ImageField(null=True, blank=True)


class Ensemble(models.Model):
    accuracy = models.FloatField(null=True)
    neural_networks = models.ManyToManyField(
        NeuralNetwork,
        through="NeuralNetworkGroup"
    )

    def normalize_weights(self):
        total_weights = sum([NeuralNetworkGroup.objects.get(id=model.id).weight for model in self.neuralnetworkgroup_set.all()])
        for model in self.neuralnetworkgroup_set.all():
            model = NeuralNetworkGroup.objects.get(id=model.id)
            model.weight = model.weight / total_weights
            model.save()


class NeuralNetworkGroup(models.Model):
    class Meta:
        unique_together = ['neural_network', 'ensemble']
    neural_network = models.ForeignKey(NeuralNetwork, on_delete=models.CASCADE)
    ensemble = models.ForeignKey(Ensemble, on_delete=models.CASCADE)
    weight = models.FloatField(default=1)

    def __str__(self):
        return str(str(str(self.ensemble) + str(self.neural_network)))

