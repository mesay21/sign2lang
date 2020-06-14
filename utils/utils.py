import json
import os
import random

import cv2
import numpy as np
import tensorflow as tf


def parse_example(example_proto, features):
    """Parse serialized tensors.
    Args:
        example_proto--> tf.train.Example protocol buffer message
        features--> a dictionary describing the features
    Returns:
        A dictionary mapping the feature keys to tensors
    
    """
    return tf.io.parse_single_example(example_proto, features)


def decode_image(encoded_image):
    '''Decode an image from a string.
    Args:
        encoded_image--> JPEG encoded string tensor
    returns-->
        image--> JPEG decoded image tensor
    '''
    img = tf.io.decode_jpeg(encoded_image)
    img.set_shape((256, 256, 3))
    return decode_image


def normalize(image):
    '''Normalize input image by subtracting mean and standard deviation. The mean
    and standard deviation values are taken from ImageNet data.

    Args:
        image--> Image tensor of shape 3D (H x W x C) 
    Returns:
        image--> Normalized image tensor
    '''

    imagenet_mean = tf.reshape(tf.constant([0.485, 0.456, 0.406]), (1, 1, -1))
    imagenet_std = tf.reshape(tf.constant([0.229, 0.224, 0.225]), (1, 1, -1))

    image = tf.cast(image, tf.float32) / 255.

    image = (image - imagenet_mean) / imagenet_std

    return image


def parse_video(example_proto):
    '''Parse a tfrecord file containing videos and return a normalized video array.
    Args:
        example_proto--> tf.train.Example protocol buffer message
    Retruns:
        video--> N-D tensor containing the video frames (#F x H x W x C)
    '''

    video_features = {
        'num_frames': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'frames': tf.io.VarLenFeature(tf.string),
    }

    parse_features = parse_example(example_proto, video_features)
    frames = parse_features['frames']
    video = tf.map_fn(lambda x: decode_image(x),
                      frames.values, dtype=tf.uint8)
    video = tf.map_fn(lambda x: normalize(x), video, dtype=tf.float32)
    
    return video
