'''Training function for Sign video classification.
'''

import argparse
import os
import datetime as dt

import tensorflow as tf
from tensorflow import keras as K
import wandb
from wandb.keras import WandbCallback
import yaml

# from model import Model
from src.model import Model
from utils import *

with open('configs/config.yml') as fp:
    config_data = yaml.load(fp, yaml.FullLoader)

hyperparameter_defaults = dict(
    dropout=config_data.get('dropout'),
    num_frames=config_data.get('num_frames'),
    batch_size=config_data.get('batch_size'),
    learning_rate=config_data.get('learning_rate'),
    epochs=config_data.get('epochs'),
    depth=config_data.get('depth'),
    )

wandb.init(
    entity=config_data.get('entity'),
    project=config_data.get('project'),
    config=hyperparameter_defaults)

config = wandb.config

def input_pipeline(file_path, meta_file, training=False):
    '''Create training input pipline using tf.data API
    Args:
        file_path--> path containing the tfrecord files
        meta_file--> path to the JSON file containing file ID's and corresponding labels
        training--> Indicates if the pipeline is for training or valdation/test
    Returns:
        
    '''
    crop_size = [config_data.get('HEIGHT'), config_data.get('WIDTH')]
    #Get filenames and labels
    file_list, label_list = get_file_list(file_path, meta_file)

    label = tf.data.Dataset.from_tensor_slices(label_list)

    data = tf.data.TFRecordDataset(file_list)
    # Parse videos from tfrecord files
    data = data.map(parse_video, num_parallel_calls=8)
    dataset = tf.data.Dataset.zip((data, label))

    dataset = dataset.map(lambda x, y: tf.py_function(get_random_frames,
                                                      [x, y, config.num_frames], [tf.float32, tf.float32]))

    # If the pipeline is for training apply random cropping other wise perform center cropping
    if training:
        dataset = dataset.map(lambda x, y: tf.py_function(
            random_crop, [x, y, crop_size], [tf.float32, tf.float32]))
        dataset = dataset.batch(config.batch_size, drop_remainder=True)

    else:

        dataset = dataset.map(lambda x, y: tf.py_function(
            center_crop, [x, y, crop_size], [tf.float32, tf.float32]))
        dataset = dataset.batch(config.batch_size)
    
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

def train(log_dir, dataset_dir):
    '''Define and train a network.
    Args:
        log_dir--> path to save training outputs
        dataset_dir--> path to the dataset folder
    Returns:
        None    
    '''

    #Clear session
    tf.keras.backend.clear_session()

    #Read dataset
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'validation')

    train_meta = os.path.join(
        dataset_dir, 'wlasl_{}/train_{}.json'.format(
            config_data.get('NUM_CLASSES'), config_data.get('NUM_CLASSES')))
    val_meta = os.path.join(
        dataset_dir, 'wlasl_{}/val_{}.json'.format(
            config_data.get('NUM_CLASSES'),  config_data.get('NUM_CLASSES')))


    train = input_pipeline(train_dir, train_meta, training=True)
    validation = input_pipeline(val_dir, val_meta)

    checkpoint_dir = os.path.join(log_dir, 'checkpoint_{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))

    #Check if the directory exits and create if not
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model_file = os.path.join(log_dir,
        'Sign2Text-I3D-{}.h5'.format(
            dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))

    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    #Model input
    with strategy.scope():
        
        network = Model(
            num_channels=config_data.get('CHANNELS'),
            num_classes=config_data.get('NUM_CLASSES'),
            num_frames=config.num_frames,
            frame_height=config_data.get('HEIGHT'),
            frame_width=config_data.get('WIDTH'),
            dropout_prob=config.dropout)
        
        model = network.network(depth=config.depth)
        model.summary()
        
        optimizer = K.optimizers.Adam(config.learning_rate)

        model.compile(optimizer=optimizer,
                    loss=K.losses.CategoricalCrossentropy(from_logits=True), metrics=['categorical_accuracy'])

    model.summary()
    
    chekpoint_path = os.path.join(
        checkpoint_dir, "weights-{epoch:02d}-{val_loss:.2f}.hdf5")

    callbacks = [K.callbacks.TensorBoard(checkpoint_dir, write_images=True, profile_batch=0),
                 K.callbacks.ModelCheckpoint(chekpoint_path, save_weights_only=True,
                                             save_best_only=True, monitor='val_loss'),
                 K.callbacks.EarlyStopping(min_delta=1e-4, patience=20, restore_best_weights=True, verbose=1),
                 WandbCallback()]
#     Train the model
    model.fit(x=train, epochs=config.epochs, callbacks=callbacks,
                    validation_data=validation, verbose=1)

    model.reset_metrics()
    model.save(model_file)  # save model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log_dir', type=str,
                        help='Directory to save the trained model and checkpoints')

    parser.add_argument('-d', '--dataset_dir', type=str,
                        help='Directory where the dataset is saved')

    args = parser.parse_args()

    train(args.log_dir, args.dataset_dir)
