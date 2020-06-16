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


def get_random_frames(frames, num_classes, label=None, num_frames=20):
    '''Given a numpy array for a video frame, sample frames 
        without loosing temporal sequence.
    Args:
        frames--> 4D array of frames
        num_classes--> number of classes (int)
        num_frames--> number of frames to select
        label--> class label of the video
    Retruns:
        sampled_frames--> 4D array frames
        label--> one-hot encoded label of the video
    '''
    if (len(frames) < num_frames):
        # If the number of frames is less than the required number of frames,
        # append frames with zero values.
        append_shape = [num_frames - len(frames)] + list(frames.shape[1:])
        zeros_array = np.zeros(shape=append_shape)
        frames = np.concatenate((frames, zeros_array), axis=0)
        assert len(frames) == num_frames

        return frames, tf.one_hot(label, num_classes, dtype=tf.float32)

    rate = len(frames)//num_frames
    index = np.arange(0, len(frames), rate)
    sampled_frames = np.stack([frames[i] for i in index[:num_frames]], axis=0)
    
    return sampled_frames, tf.one_hot(label, num_classes, dtype=tf.float32)


def random_flip(video):
    '''Horizontaly flip a video 
    Args:
        video--> 4D array of video
    Returns:
        horizontaly flipped video
    '''
    if random.random() < 0.5:
        return video[:, :, ::-1, :]

    return video


def random_crop(video, label=None, target_size=None):
    '''Crop rectangular part of a video.
    Args:
        video--> 4D array of video frames
        label--> label of the video
        target_size--> target size of the frames
    Retruns:
        cropped_video--> cropped version of the input video (4D array)     
    '''
    _, h, w, _ = video.shape
    th, tw = target_size
    video = random_flip(video)

    if (h == th) and (w == tw):

        return video, label

    offset_h = random.randint(0, h - th) if (h != th) else 0
    offset_w = random.randint(0, w - tw) if (w != tw) else 0

    cropped_video = video[:, offset_h:(
        offset_h + th), offset_w:(offset_w + tw), :]

    return cropped_video, label


def center_crop(video, label=None, target_size=None):
    '''Crop center part of a video array.
    Args:
        video--> 4D array of video frames
        label--> label of the video
        target_size--> target size of the frames
    Retruns:
        cropped_video--> center cropped input video (4D array)     
    '''
    _, h, w, _ = video.shape
    th, tw = target_size

    offset_h = int((h - th)/2)
    offset_w = int((w - tw)/2)

    cropped_video = video[:, offset_h:(
        offset_h + th), offset_w:(offset_w + tw), :]

    return cropped_video, label


def read_json(json_path):
    ''' Read a JSON file.
    Args:
        json_path-->path to JSON file.
        shuffle--->
    Returns:
        data--> Contents of the JSON file
    '''
    with open(json_path, 'r+') as fp:
        data = json.load(fp)

    return data
