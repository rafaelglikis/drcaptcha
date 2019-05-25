import uuid
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from captcha.services.ml.emnist import get_mapping


def plot_accuracy(history):
    """
    Creates an accuracy graph that plots the accuracy of a model in each epoch
    :param history: the history of the model as returned by keras model fit function
    :return: path to the graph image
    """
    import matplotlib.pyplot as plt
    path = "static/graphs/" + "acc-plot-" + str(uuid.uuid1()) + ".png"
    plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(path)
    plt.close()
    return path


def plot_loss(history):
    """
    Creates a loss graph that plots the loss function value of a model in each epoch
    :param history: the history of the model as returned by keras model fit function
    :return: path to the graph image
    """
    import matplotlib.pyplot as plt
    path = "static/graphs/" + "loss-plot-" + str(uuid.uuid1()) + ".png"
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(path)
    plt.close()
    return path


def plot_model(model):
    """
    Creates a graphical representation of a keras model
    :param model: keras model
    :return: path to the image
    """
    path = "static/graphs/" + "model-plot-" + str(uuid.uuid1()) + ".png"
    keras.utils.plot_model(
        model,
        to_file=path,
        show_shapes=True,
        show_layer_names=True
    )
    return path


def plot_confusion_matrix(cm):
    """
    Creates a heatmap for a confusion matrix
    :param cm: confusion matrix
    :return: path to the image
    """
    path = "static/graphs/" + "confusion-matrix-" + str(uuid.uuid1()) + ".png"
    labels = [chr(v) for k, v in get_mapping().items()]
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    sns.heatmap(cm_df, xticklabels=True, yticklabels=True, cmap='RdYlGn')
    fig = plt.gcf()
    fig.set_size_inches(12, 12)
    # plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(path)
    plt.close()
    return path
