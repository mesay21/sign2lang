'''
This module contains a class that loads the Inflated Inception 3D (I3D) model and 
attaches a custom classification to the model. The I3D model is initalized from
weights pretrained on the ImageNet and Kinetics dataset.
'''

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers as L
import tensorflow.keras.backend as B

from .i3d_inception import Inception_Inflated3d
from .i3d_inception import conv3d_bn

class Model:
    ''' Load I3D model and attach custom output layer.    
    '''
    def __init__(
        self,
        num_classes,
        num_frames,
        frame_height,
        frame_width,
        num_channels,
        dropout_prob = 0.3
        ):

        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.num_frames = num_frames
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.num_channels = num_channels

    def _load_i3d_model(self, input_tensor, depth=0):
        '''create the i3d model with a given input shape.
        Args:
            input_tensor--> keras input tensor of shape
                (-1, HEIGHT, WIDTH, CHANNEL)
            depth--> the number of layers at the top to train.
                eg. 0 depth means all the I3D layers are frozen.
        Returns:
            I3D model       
        '''

        i3d_net = Inception_Inflated3d(
            include_top=False,
            weights='rgb_imagenet_and_kinetics',
            input_tensor=input_tensor)

        num_layers_to_freez = len(i3d_net.layers) - depth
        for layer in i3d_net.layers[:num_layers_to_freez]:
            layer.trainable = False
        
        return i3d_net

    def network(self, depth):
        ''' Attach a classifier layer to the i3d network.
        Args:
            input_tensor--> Keras Input tensor
            i3d_net--> I3D network
        Returns: 
            model--> a keras model      
        '''
        #network input
        x = K.Input(
            shape=(None, self.frame_height, self.frame_width, self.num_channels))
        
        # Load I3D RGB model
        
        i3d_net = self._load_i3d_model(x, depth)       
        features = i3d_net.layers[-1].output
        
        dropout = L.Dropout(rate=self.dropout_prob)(features)
        
        logits = conv3d_bn(
            dropout,
            self.num_classes, 1, 1, 1,
            use_activation_fn=False,
            use_bias=True, use_bn=False)
        
        logits = L.Reshape(target_shape=(-1, self.num_classes))(logits)
        
        logits = L.Lambda(lambda x: B.mean(x, axis=1, keepdims=False),
                output_shape=lambda s: (s[0], s[2]))(logits)    
        
        model = K.Model(inputs=x, outputs=logits, name='Custom-I3D-Model')

        return model

        
