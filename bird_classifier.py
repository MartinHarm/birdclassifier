"""Bird species classifier"""
import logging
import os
import time
import urllib.request
from urllib.error import HTTPError, URLError

import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_hub as hub

log = logging.getLogger()

BIRD_MODEL_URL = 'https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1'
BIRD_LABELS_URL = 'https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv'

IMAGE_URLS = [
    'https://upload.wikimedia.org/wikipedia/commons/c/c8/Phalacrocorax_varius_-Waikawa%2C_Marlborough%2C_New_Zealand-8.jpg',
    'https://quiz.natureid.no/bird/db_media/eBook/679edc606d9a363f775dabf0497d31de8c3d7060.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/8/81/Eumomota_superciliosa.jpg',
    'https://i.pinimg.com/originals/f3/fb/92/f3fb92afce5ddff09a7370d90d021225.jpg',
    'https://cdn.britannica.com/77/189277-004-0A3BC3D4.jpg'
]


def load_model():
    """Loads Keraslayer model for birds_v1.

    Returns:
        KerasLayer.

    """
    return hub.KerasLayer(BIRD_MODEL_URL)


def load_image(image_url):
    """Retrieves and scales image to be used as tensor.

    Args:
        image_url: URL for the image to be processed.

    Throws:
        HTTPError: when server connection times out
        URLError: when url malformed

    Returns:
        image if successfully loaded and [] if had exceptions.

    """
    try:
        image_get_response = urllib.request.urlopen(image_url)
        image_array = np.asarray(bytearray(image_get_response.read()), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255
    except HTTPError as e:
        log.error('Server couldn\'t fulfill the request. For url:' + image_url)
        return []
    except URLError as e:
        log.error('We failed to reach the server. For url:' + image_url)
        return []
    else:
        return image


def load_and_cleanup_labels():
    """Retrieves and cleans up bird species labels.

    Returns:
        Bird species

    """
    bird_labels_raw = urllib.request.urlopen(BIRD_LABELS_URL).readlines()
    bird_labels_lines = [line.decode('utf-8').replace('\n', '') for line in bird_labels_raw]
    bird_labels_lines.pop(0)  # remove header (id, name)
    birds = {}
    for bird_line in bird_labels_lines:
        bird_id = int(bird_line.split(',')[0])
        bird_name = bird_line.split(',')[1]
        birds[bird_id] = {'name': bird_name}

    return birds


bird_model = load_model()
bird_labels = load_and_cleanup_labels()


def convert_image_to_tensor(image):
    """Converts image(bytes) to img_tensor.

    Args:
        img: Image to be converted.

    Returns:
        Image tensor

    """
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, 0)
    return image_tensor


def print_results(birds_names_with_results_ordered, index, img_url):
    """Prints out organized top three results for given image.

    Args:
        birds_names_with_results_ordered: List of bird names and scores(ordered).
        index: Image index.
        img_url: URL of the currently processed image

    """
    print('Run: %s' % int(index + 1), img_url)
    for i in range(1, 4):
        bird_name = birds_names_with_results_ordered[i * (-1)][1]['name']
        bird_score = birds_names_with_results_ordered[i * (-1)][1]['score']
        print(f"{i}.Match:  {bird_name} with score: {bird_score}")


def get_model_result(image_url):
    """Calls bird model tensor and sorts results by score.

    Args:
        image_url: URL of the image to be processed.

    Returns:
        Result of bird species sorted by their score.

    """
    image = load_image(image_url)
    if not len(image):
        return
    model_result = bird_model.call(convert_image_to_tensor(image)).numpy()
    for index, value in np.ndenumerate(model_result):
        bird_labels[index[1]]['score'] = value

    return sorted(bird_labels.items(), key=lambda x: x[1]['score'])


def main():
    """Application Main"""
    for index, image_url in enumerate(IMAGE_URLS):
        model_result = get_model_result(image_url)
        if model_result:
            print_results(model_result, index, image_url)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print('Time spent: %s' % (time.time() - start_time))
