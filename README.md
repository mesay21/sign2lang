# Sign2Text

## What is Sign2Text?
Sign2Text is a machine learning model that translates signed videos into text. 
* Sign2Text uses the Inflated 3D Inception architecture from [Deepmind](https://github.com/deepmind/kinetics-i3d) 	to classify signed videos into words. The model used for this work is trained on the [kinetics](https://	deepmind.com/research/open-source/kinetics) dataset.

* Transfer learning is applied to fine-tune the pre-trained model on the [WLASL](https://github.com/dxli94/WLASL) dataset. 

* The final product is a trained model deployed in a docker image using Tensorflow serving and an API
	to communicate with the image for inference.

## Features

* Training your own sign2text model
* Loading a trained model and performing inference on test samples
* Converting video files into TFrecord format to speedup training
* Client API to send gRPC request and receive inference
 
## Setup
Sign2Text requires Python 3 to run. 
1. Clone the repository
```sh
$ git clone https://github.com/mesay21/sign2text.git
```
2. I recommend to install [Anconda](https://www.anaconda.com/products/individual) for package managment and  
run the environment.yml file to create an environment named sign-to-text.  Run the following command:

```sh
$ conda env create -f environment.yml
```
* Activate the environment using 
```sh 
$ conda activate sign-to-text
```
* Install all the dependencies inside the environment
```sh
$ pip install -r requirements.txt
```
3. Install [Docker](https://docs.docker.com/get-docker/) to use the deployed model.
* Get the docker image containing the model from docker hub
```sh
$ docker pull mesayb/sign2text
```
* Start sign2text container and open a gRPC port with the following command
```sh
$ docker run -d -p <port-number>:8500 --name sign2text mesayb/sign2text
```
Now the docker named **sign2text** is runnng at IP address **0.0.0.0** and port **port-number**

4. Test the docker image by running a sample prediction request using the video files  
from **data/sample_videos** directory
```sh
$ python grpc_client.py --video_path data/sample_videos/<video-name> --ip_addr 0.0.0.0 --port <port-number>
```
### Training your own sign2text model
    
1. Download the WLASL dataset from the [repository](https://github.com/dxli94/WLASL).
2. The *train.py* file expects the training and validation video files to be in TFrecord format.  
    Convert the videos into TFrecord format using the *conver_to_tfrecord.py* file.  Run the following command to see 
    instructions on how to use the file.
```sh
$ python convert_to_tfrecords.py --help
```
3. The *train.py* module expects the dataset directory to be structured as follows
```bash  
|--- dataset  
    |--- train   
    |--- validation  
    |--- test  
    |--- wlasl_<num-classes>  
```
**dataset/train** : directory containing training tfrecord files  

**dataset/validation**: directory containing validation tfrecord files  

**dataset/test**: directory containing test video files  

**dataset/wlasl_num-classes**: directory containing JSON files related to the training, validation, and test files.  
    The JSON files contain information such as video file name, numrical and gloss label of the video file and other information.
    You can find these files in the **data/meta-data** directory. 
    *num-classes* the number of classes to which the files belong.  
    For example, **wlasl_100** refers too a directory containing JSON files corresponding to the top 100 classes from the original dataset.

4. I recommend to use [weights and biases](https://www.wandb.com/) to track the training. Create a free account, create a project, and add your
    account name and project name in the **configs/config.yml** file in the section "wandb init values".  
    You can also modify default hyperparameter configuration values in this file.

5. Run the *train.py* file to train your model
```sh
$ python train.py --log_dir <log-directory> --dataset_dir <dataset-directory>
```

**log-directory**: directory where training checkpoint and model are saved  
**dataset-directory**: directory where the dataset is saved (as structured above)

6. Generate performance report on the test samples using the test.py module.  You can use your trained model or there is
a saved model in the **data/weights** folder to run the test.  Run the following command to see usage instructions
```sh
$ python test.py --help
```

## To do

add activation visualization module  
add test


