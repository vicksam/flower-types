import json
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow_hub import KerasLayer

IMG_SHAPE = 224

# Load and pre-process image from path
def prepare_image(image_path):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    # Add extra dimension expected by the model
     # Change shape from (224, 224, 3) to (1, 224, 224, 3)
    image = np.expand_dims(image, axis = 0)
    return image

# Resize image to 224x224 and normalize the color range
def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (IMG_SHAPE, IMG_SHAPE))
    image /= 255
    return image

# Load keras model from path
def load_model(model_path):
    return tf.keras.models.load_model(
        model_path,
        # model uses KerasLayer which is a part of hub.KerasLayer
        custom_objects = {'KerasLayer' : KerasLayer}
    )

# Load class names from a json file by indexes of top k predictions
def load_class_names(category_names_path, top_k_indexes):
    with open(category_names_path, 'r') as f:
        class_dictionary = json.load(f)
        class_names = [class_dictionary[str(i + 1)] for i in top_k_indexes]
        return class_names

def print_results(class_names, probabilities):
    pd.options.display.float_format = '{:.2%}'.format
    df = pd.DataFrame(
        probabilities,
        columns = ['Probability'],
        index = class_names
    )
    print(df)
