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


