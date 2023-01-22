import tensorflow as tf
import numpy as np
from PIL import Image
import base64, io, os

class_names = ["apple", "apricot", "avocado", "banana", "bell pepper", "black berry", "blueberry", "cantaloupe", "cherry", "coconut", "coffee", "desert fig", "eggplant", "fig", "grape", "grapefruit", "kiwi", "lemon", "lime", "lychee", "mango", "orange", "pear", "pineapple", "pomegranate", "pumpkin", "raspberry", "strawberry", "watermelon"] 

def load_model(model_name):
    return tf.keras.models.load_model(f"static/{model_name}.h5")

model = load_model("fig")

def predict(image, model):
    image = image.resize((180, 180)).convert("RGB")
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array, verbose=False)
    return tf.nn.softmax(predictions[0])

def load_image(b64):
    bin_image = base64.b64decode(b64)
    return Image.open(io.BytesIO(bin_image))

def detect(image):
    predictions = predict(image, model).numpy().tolist()
    confidences = {class_name : str(prediction) for class_name, prediction in zip(class_names, predictions)}
    return {
        "confidences": confidences,
        "fruit": class_names[np.argmax(predictions)],
        "value": np.max(predictions)
    }
