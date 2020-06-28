
import argparse
import os
import tempfile

import tensorflow as tf
import tensorflow.keras.backend as B
import tensorflow.keras as K
import numpy as np
from sklearn.metrics import classification_report
import yaml

from utils import read_json
from utils import read_video
from utils import center_crop
from utils import get_file_list
from utils import normalize


LABEL_MAP_FILE = 'data/label_map_100.json'

with open('configs/config.yml') as fp:
    config_data = yaml.load(fp, yaml.FullLoader)

def get_pridiction(model, data):
    '''Pridict labels using keras model.predict method.
    Args:
        model--> the network
        data--> the data (4D array)
    Returns:
        label-->predicted label of the data
    '''
    # Add batch dimension to the data
    data = np.expand_dims(data, axis=0)
    logits = model(data, training=False)

    prob = B.softmax(logits)
    label = B.argmax(prob)

    return label.numpy()

def evaluate(model_path, file_dir, meta_file):
    '''Evaluates a trained model on the test samples and outputs the 
    overall and average accuracy values.
    Args:
        model_path--> path to the keras saved model
        file_dir --> path to the video files
        meta_file --> meta file containing the test video file names and labels
    Returns:
        report--> returns a text report showing main classification metrics using
                scikit-learn classification report method
    '''

    loaded_model = K.models.load_model(
        model_path,
        custom_objects={
            'B': B
        })
    test_data, test_label = get_file_list(file_dir, meta_file, file_type='.mp4')
    test_prediction = []

    for x, y in zip(test_data, test_label):
        video = read_video(x)
        video = normalize(video)
        video, _ = center_crop(video, target_size=(
            config_data.get('HEIGHT'), config_data.get('WIDTH')))
        prob = get_pridiction(loaded_model, video)
        test_prediction.append(prob[0])
    
    label_map = read_json(LABEL_MAP_FILE)
    vocabulary = [label_map.get(str(i)) for i in range(config_data.get('NUM_CLASSES'))]
    report = classification_report(
        test_label, test_prediction,
        target_names=vocabulary, zero_division=1)
    
    return report

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--model_dir',
        type=str, help='Path to the .h5 model file')
    parser.add_argument('-s', '--video_dir', type=str,
        help='Directory where the videos are saved')
    parser.add_argument('-m', '--meta_file', type=str,
        help='Path to meta file')
    
    args = parser.parse_args()
    report = evaluate(args.model_dir, args.video_dir, args.meta_file)
    print(report)
        
    



    
