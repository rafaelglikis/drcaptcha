import numpy as np
import tensorflow as tf
from captcha.services.ml.plot import *
from captcha.models import NeuralNetwork, Ensemble
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.python.keras.layers \
    import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense


class EnsembleModel:
    class ModelGroup:
        def __init__(self, model, weight):
            self.model = model
            self.weight = weight

    def __init__(self, ensemble: Ensemble):
        self.ensemble = ensemble
        self.models = []

        for model in ensemble.neuralnetworkgroup_set.all():
            weight = model.weight
            model = tf.keras.models.load_model(model.neural_network.path, custom_objects=None, compile=True)
            self.models.append(self.ModelGroup(model, weight))

    def evaluate(self, x_test, y_test):
        predictions = []
        for model in self.models:
            predictions.append(model.model.predict(x_test) * model.weight)

        pred = [sum(prediction) for prediction in zip(*predictions)]
        accuracy = accuracy_score(y_test, np.argmax(pred, axis=1))
        print("Ensemble model accuracy: {}".format(accuracy))
        return accuracy

    def predict(self, data):
        predictions = []
        for model in self.models:
            predictions.append(model.model.predict(data) * model.weight)
        pred = [sum(prediction) for prediction in zip(*predictions)]
        return pred


class Model:
    def __init__(self, num_classes=47):
        self.num_classes = num_classes
        self.model = keras.Sequential()
        self.epochs = 0
        self.batch_size = 0
        self.history = []
        self.score = []
        self.training_set_size = 0
        self.test_set_size = 0
        self.x_train = np.array([])
        self.y_train = np.array([])
        self.x_test = np.array([])
        self.y_test = np.array([])
        self.input_shape = ()
        self.confusion_matrix = []

    def set_training_data(self, x_train, y_train):
        print(x_train.shape[0], 'train samples')
        self.input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
        self.x_train = x_train
        self.y_train = keras.utils.to_categorical(y_train, self.num_classes)

    def set_test_data(self, x_test, y_test):
        print(x_test.shape[0], 'test samples')
        self.x_test = x_test
        self.y_test = keras.utils.to_categorical(y_test, self.num_classes)

    def train(self, epochs=1, batch_size=128):
        self.epochs = epochs
        self.batch_size = batch_size
        self.training_set_size = self.x_train.shape[0]
        self.test_set_size = self.x_test.shape[0]

        self.history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            # validation_split=0.1,
            # validation_data=(x_valid, y_valid),
            shuffle=True
        )
        self.score = self.model.evaluate(self.x_test, self.y_test)
        print("Accuracy: ", self.score[1])

        self.confusion_matrix = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(self.predict(self.x_test), axis=1))

    def predict(self, data):
        return self.model.predict(data)

    def save(self):
        path = "ml_models/" + str(uuid.uuid1()) + ".h5"
        self.model.save(path)

        nn = NeuralNetwork()
        nn.training_set_size = self.training_set_size
        nn.test_set_size = self.test_set_size
        nn.model_summary_image = plot_model(self.model)
        nn.epochs = self.epochs
        nn.batch_size = self.batch_size
        nn.accuracy_graph = plot_accuracy(self.history)
        nn.loss_graph = plot_loss(self.history)
        nn.accuracy = self.score[1]
        nn.confusion_matrix = plot_confusion_matrix(self.confusion_matrix)
        nn.path = path
        nn.save()

        return nn


class NNModel(Model):
    def build(self, num_classes=47):
        self.model.add(Flatten(input_shape=self.input_shape))
        self.model.add(Dense(units=10000, activation='relu'))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(units=num_classes, activation='softmax'))

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adadelta',
            metrics=['accuracy'])

        print(self.model.summary())


class DNNModel(Model):
    def build(self, num_classes=47):
        self.model.add(Flatten(input_shape=self.input_shape))
        self.model.add(Dense(units=4096,activation='relu'))
        self.model.add(Dropout(rate=0.1))
        self.model.add(Dense(units=1024, activation='relu'))
        self.model.add(Dropout(rate=0.1))
        self.model.add(Dense(units=512, activation='relu'))
        self.model.add(Dropout(rate=0.1))
        self.model.add(Dense(units=num_classes, activation='softmax'))

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adadelta',
            metrics=['accuracy'])

        print(self.model.summary())


class CNNModel1(Model):
    def build(self, num_classes=47):
        self.model.add(Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=self.input_shape))
        self.model.add(Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(rate=0.25))
        self.model.add(Dense(units=num_classes, activation='softmax'))

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='Adagrad',
            metrics=['accuracy'])

        print(self.model.summary())


class CNNModel2(Model):
    def build(self, num_classes=47):
        self.model.add(Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding='valid',
            input_shape=self.input_shape,
            activation='relu'))
        self.model.add(Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.25))
        self.model.add(Flatten())

        self.model.add(Dense(units=512, activation='relu'))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(units=num_classes, activation='softmax'))

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adadelta',
            metrics=['accuracy'])

        print(self.model.summary())

