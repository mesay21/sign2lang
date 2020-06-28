import argparse
import os

import tensorflow as tf
import cv2

from utils import read_json
from utils import read_video

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def video_to_tfrecords(video_dir, meta_file, dest_dir, file_type):
    '''Convert video files into a tfrecord format for training and write them to a file.
    Args:
        video_dir--> directory where the videos are stored
        meta_file--> JSON file containing information about the video files.
                    This file has information such as file name, numerical and gloss label of
                    the file.
        dest_dir--> directory to save the output tfrecord files
        file_type--> extension of the video files (eg. '.mp4', '.avi')  
    
    Returns:
        None
    
    '''
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    video_files = read_json(meta_file)
    print('Started writting video files to tfrecord format')
    
    for instance in video_files:
        
        file_path = os.path.join(
            video_dir, '{}{}'.format(instance['video_id'], file_type))
        feature = {}
        file_name = os.path.join(dest_dir, '{}{}'.format(
            instance['video_id'], '.tfrecords'))
        
        writer = tf.io.TFRecordWriter(file_name)
        video = read_video(file_path)

        num_frames, height, width, channel = video.shape
        
        #convert frames to bytes
        frames = [tf.compat.as_bytes(cv2.imencode(".jpg", frame)[
                                     1].tobytes()) for frame in video]

        feature['frames'] = _bytes_feature(frames)
        feature['height'] = _int64_feature(height)
        feature['width'] = _int64_feature(width)
        feature['channels'] = _int64_feature(channel)
        feature['gloss'] = _bytes_feature(
            [tf.compat.as_bytes(instance['gloss'])])
        feature['label'] = _int64_feature(instance['label'])
        feature['num_frames'] = _int64_feature(num_frames)
        feature['video_id'] = _bytes_feature(
            [tf.compat.as_bytes(instance['video_id'])])
        feature['format'] = _bytes_feature([tf.compat.as_bytes(file_type)])

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        writer.write(example.SerializeToString())
        writer.close()
    
    print('Fineshed writting video files to tfrecord format')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-s', '--video_dir',
        type=str, help='Directory where the video are stored')

    parser.add_argument(
        '-d', '--dest_dir',
        type=str, help='Directory where the output files are stored')

    parser.add_argument(
        '-m', '--meta_file',
        type=str, help='Path to the JSON file containing the file information')

    parser.add_argument(
        '-t', '--file_type',
        type=str, default='.mp4', help='Video files extension (default: .mp4)')
    
    args = parser.parse_args()
    video_to_tfrecords(
        args.video_dir, args.meta_file,
        args.dest_dir, args.file_type)
    

