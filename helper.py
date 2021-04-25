"""HELPER MODULE"""
import tensorflow as tf
import tensorflow_hub as hub

BIRD_MODEL_URL = 'https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1'


def load_model():
    """Load the model"""
    return hub.KerasLayer(BIRD_MODEL_URL)


def convert_image_to_tensor(image):
    """converts loaded image to tensor"""
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, 0)
    return image_tensor
