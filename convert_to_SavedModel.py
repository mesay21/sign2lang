
import argparse
import os
import tempfile

import tensorflow as tf
import tensorflow.keras.backend as B
import tensorflow.keras as K
import numpy as np

def convert_to_saved_model(model, path):
    '''Convert a tensorflow model to SavedModel format, which is suitable for 
    deploying.
    Args:
        model--> keras model
        path--> directory to save the converted model
    Returns:
        converts keras model saved in h5 format into SavedModel format
        to be used for serving.
    
    '''
    if os.path.isdir(path):
        print('\n Removing already existing model')
        os.remove(path)
    else:
        print('\n Creating the directory')
        os.makedirs(path)
    tf.saved_model.save(model, path)
    print('\n Finished generating ServeModel')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--model_dir',
        type=str, help='Path to the .h5 model file')
    parser.add_argument('-s', '--serve_dir', type=str,
        help='Directory to save the model converted for serving')
    parser.add_argument('-v', '--version', type=str,
        help='Model version number')
    
    args = parser.parse_args()
    ##TODO add data path
    model = K.models.load_model(
        args.model_dir,
        custom_objects={
            'B': B
        })

    save_path = os.path.join(args.serve_dir, args.version)
    convert_to_saved_model(model, save_path)


    
