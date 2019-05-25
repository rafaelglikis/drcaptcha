import base64
import io
import re
from tensorflow.keras.preprocessing import image as im_prep
from PIL import Image
import tensorflow as tf
import numpy as np
from captcha.models import Ensemble
from captcha.services.ml import emnist
from captcha.services.ml.models import EnsembleModel


def preprocess(image):
    """
    Preprocess image as arrived from request.
    :param image: -
    :return: Preprocessed image.
    """
    image_string = re.search(r'base64,(.*)', image).group(1)
    image_bytes = io.BytesIO(base64.b64decode(image_string))
    image = Image.open(image_bytes).resize((28, 28), Image.ANTIALIAS)
    image.save('temp.png')

    img = im_prep.load_img(
        path="temp.png",
        grayscale=True,
        target_size=(28, 28, 1)
    )
    img = im_prep.img_to_array(img)
    return img.reshape(1, 28, 28, 1)


def predict(img):
    """
    Given a preprocessed image predicts her value using the top rated ensemble in the database
    :param img: preprocessed image
    :return: prediction(s)
    """
    ensemble_model = EnsembleModel(Ensemble.objects.order_by('-accuracy')[0])
    print("Loaded ensemble model: {}".format(ensemble_model.ensemble.id))

    probabilities = ensemble_model.predict(img)[0]
    tf.keras.backend.clear_session()

    mapping = emnist.get_mapping()
    for i, probability in enumerate(probabilities):
        print(" - " + chr(mapping[i]), probability)

    # prediction = np.asscalar(np.argmax(probabilities))
    predictions = [np.asscalar(prediction) for prediction in probabilities.argsort()[-3:][::-1]]
    predictions = [chr(mapping[np.int64(prediction)])  for prediction in predictions if probabilities[prediction] > 0.1]

    predictions = " or ".join(predictions)
    print("Predicted value: {}".format(predictions))
    return predictions
