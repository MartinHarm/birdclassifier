"""This module does blah blah."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from helper import load_model, convert_image_to_tensor
import time
import urllib.request
import cv2
import numpy as np

BIRD_LABELS_URL = 'https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv'

image_urls = [
    'https://upload.wikimedia.org/wikipedia/commons/c/c8/Phalacrocorax_varius_-Waikawa%2C_Marlborough%2C_New_Zealand-8.jpg',
    'https://quiz.natureid.no/bird/db_media/eBook/679edc606d9a363f775dabf0497d31de8c3d7060.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/8/81/Eumomota_superciliosa.jpg',
    'https://i.pinimg.com/originals/f3/fb/92/f3fb92afce5ddff09a7370d90d021225.jpg',
    'https://cdn.britannica.com/77/189277-004-0A3BC3D4.jpg'
]



def load_and_cleanup_labels():
    """Loads bird labels and normalizes bird the values"""
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


def order_birds_by_result_score(model_raw_output):
    """Orders bird model results to show ranked results"""
    for index, value in np.ndenumerate(model_raw_output):
        bird_index = index[1]
        bird_labels[bird_index]['score'] = value

    return sorted(bird_labels.items(), key=lambda x: x[1]['score'])


def get_top_n_result(top_index, birds_names_with_results_ordered):
    """returns number n result"""
    bird_name = birds_names_with_results_ordered[top_index * (-1)][1]['name']
    bird_score = birds_names_with_results_ordered[top_index * (-1)][1]['score']
    return bird_name, bird_score


def load_image(image_url):
    """Loads the current image and mainpulates it for tensor"""
    image_get_response = urllib.request.urlopen(image_url)
    image_array = np.asarray(bytearray(image_get_response.read()), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255
    return image


def print_results(birds_names_with_results_ordered, index):
    """Prints out the results"""
    print('Run: %s' % int(index + 1))
    for i in range(1, 4):
        bird_name, bird_score = get_top_n_result(i, birds_names_with_results_ordered)
        print(f"{i}.Match:  {bird_name} with score: {bird_score}")


def main():
    """Main method"""
    for index, image_url in enumerate(image_urls):
        model_result = \
            bird_model.call(convert_image_to_tensor(load_image(image_url))).numpy()
        print_results(order_birds_by_result_score(model_result), index)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print('Time spent: %s' % (time.time() - start_time))
