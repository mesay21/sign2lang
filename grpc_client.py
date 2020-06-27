import os
import argparse
import json

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as B

import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from utils import read_json
from utils import center_crop
from utils import normalize
from utils import get_file_list
from utils import get_random_frames
from utils import read_video

LABEL_MAP_FILE = 'data/label_map_100.json'


def label_to_word(y):
    '''Reads a json file containing the mapping from integer label to word and
    returns the word corresponding to the input label.
    Args:
        y--> label
    Retruns
        word--> word corresponding to the label 
    '''
    data = read_json(LABEL_MAP_FILE)

    return data['{}'.format(y)]


def process_response(response):
    '''Get the response from the model, extract logits values and convert it to label.
    Args:
        response--> server response for the prediction request
    Returns:
        word--> word corresponding to the class
    '''
    output = {}
    logits = response.outputs['lambda'].float_val
    output['model_info'] = {
        'name': response.model_spec.name,
        'version': response.model_spec.version.value}
    prediction_prob = B.softmax(logits)
    label = B.argmax(prediction_prob).numpy()
    word = label_to_word(label)
    prob = prediction_prob[label]
    output['result'] = {
        'word': word,
        'confidence': prob.numpy()
    }
    return output

def grpc_request(data, server_address):
    ''' Send a gRPC request to a server and receive prediction results
    Args:
        data--> a 4D array of video frames
        server_address--> the address of the server where the model is
    Returns:
        outputs--> prediction outputs of the model
    '''

    channel = grpc.insecure_channel(server_address)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    
    prediction_request = predict_pb2.PredictRequest()
    prediction_request.model_spec.name = 'sign2text'
    prediction_request.model_spec.signature_name = 'serving_default'
    prediction_request.inputs['input_1'].CopyFrom(
        tf.make_tensor_proto(data, shape=data.shape))

    try:
        response = stub.Predict(prediction_request, 10) # wait 10 seconds
    except grpc.RpcError as e:
        print('gRPC error \nError code: {} \nDetails: {}'.format(e.code().name, e.details()))
        exit()

    return response

def pre_process_input_data(data, sample=None):
    '''Apply transformation fuctions (normalization, 
    sampling frames and center cropping) on a video array. 

    Args:
        data--> a 4D array of video frames
        sample--> sample a fixed number of frames from the video (int)
    
    Returns:
         data --> a tranformed 5D data array with batch size   
    '''
    
    data = normalize(np.array(data))
    
    if sample is not None:
        data, _ = get_random_frames(data, label=0, num_frames=sample)
    
    data, _ = center_crop(data, target_size=(224, 224))
    data = np.expand_dims(data, axis=0)

    return data.astype(np.float32)

def predict(data, server_address):
    '''Receives a 4D array of video data, splits into chunks of frames and sends
    each chunk to the model for prediction
    Args:
        data--> 4D array of video frames
        server_address--> address of the server where the model is served.
    Returns:
        text--> a list of words corresponding to the video input 
    '''
    
    caption = {}
    words = []
    confidence = []

    if len(data) < 48:
        data = pre_process_input_data(data)
        response = grpc_request(data, server_address)
        output = process_response(response)
        words.append(output['result']['word'])
        confidence.append(str(output['result']['confidence']))
    
    else:
        start, step, end = 0, 16, 48
        soft_max_th = 0.09
        while True:
            #Check if the number of frames left is less than 16 and drop them
            if (end - start) < 16:
                break
            partial_frames = data[start:end]
            partial_frames = pre_process_input_data(partial_frames)
            response = grpc_request(partial_frames, server_address)
            output = process_response(response)
            if output['result']['confidence'] > soft_max_th:
                words.append(output['result']['word'])
                confidence.append(str(output['result']['confidence']))
                start = start + 48
                end = min(len(data), end + 48)
            else:
                # Add more frames if confidence is less than the threshold
                end = min(len(data), end + step)
            #Stop when the all frames are processed
            if end >= len(data):
                if len(words) == 0 :
                    words.append('Unknown word')
                break
    
    caption['words'] = words
    caption['confidence'] = confidence
    caption['model_info'] = output['model_info']
    caption = json.dumps(caption)           

    return caption

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_path', type=str,
    help='Path to a video file')

    parser.add_argument('-i', '--ip_addr', type=str,
    help='IP address of the server')

    parser.add_argument('-p', '--port', type=int,
    help='Port address of the application')

    args = parser.parse_args()

    video = read_video(args.video_path)

    server_addr = '{}:{}'.format(args.ip_addr, args.port)

    response = predict(video, server_addr)
    print(response)




    
