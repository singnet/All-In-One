# All-In-One
All in one [paper](https://arxiv.org/abs/1611.00851) implementation
## Getting started
All in one convolutional network for face analysis presents a multipurpose algorithm for simultaneous face detection, face alignment, pose estimation, gender recognition, smile detection, age estimation and face recognition using a single convolutional neural network(CNN).
## Prerequisites
The project can be run by installing conda virtual environment with python=3.6 and installing dependencies using pip. Inside the projects directory run the following commands.
* *`conda create -n <environment_name> python=3.6`.* <br/>After creating the environment use the following commands to install dependacies.
* *`pip install keras`*
* *`pip install tensorflow`*
* *`pip install sklearn`*
* *`pip install pandas`*
* *`pip install opencv-python`*
* *`pip install dlib`*
## How to train the model
Three datasets are used for training the network.
  * *[AFLW dataset](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/) provides a large-scale collection of annotated face images gathered from the web, exhibiting a large variety in appearance (e.g., pose, expression, ethnicity, age, gender) as well as general imaging and environmental conditions.*
  * *[IMDB-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) is the largest publicly available dataset of face images with gender and age labels for training.*
  * *[CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations.*
  * *[Adience dataset](https://talhassner.github.io/home/projects/Adience/Adience-data.html) attempts to capture all the variations in appearance, noise, pose, lighting and more, that can be expected of images taken without careful preparation or posing.*

The network architecture can be found in this [paper](https://arxiv.org/abs/1611.00851). The model is built with deep convolutional layers in keras and is found in nets/model.py.

## Training
  The model can be trained with age, gender,detection, visibility,pose, landmarks,identity, smile, and eye_glasses labels by using the following commands inside the project's directory. \
The following code snippet is bash command to train the network in aflw dataset for face detection
```
python -m train --dataset aflw --images_path /path-to-dataset-images/ \
    --label detection --batch_size 100 --steps 500  --ol output-of-large-model --os output-of-small-model --epochs 10;
```
### Options
* *--images_path - Path to dataset images*
* *--dataset - Type of dataset to train the model. This could be imdb, wiki, celeba,yale,ck+,aflw. The layers that are going to be trained also depends on this choice.*
* *--label - This option specifies which for which type of classification/prediction to train the model. The choices are age, gender,detection, visibility,pose, landmarks,identity, smile, and eye_glasses.*
* *--epochs.*
* *--batch_size.*
* *--resume - To start training from previous checkpoint if available.*
* *--steps - Steps per epoch.*
* *--ol - Output filename to save large model(model with all layers)*
* *--os - Output filename to save small model(model with layers trained with current training)*
* *--load_model -*
* *--freeze - If true freezes shared layers of the model*
## How to run demo
## To do lists
#### Previous results recorded after training the model.
* *Gender estimation(~89% accuracy)*
* *Face detection(~90% accuracy)*
* *Smile detection(~91% accuracy)*
* *Age prediction(~4% accuaracy)*
#### Tasks remaining
* *Use CASIA and MORPH dataset for further training the model on age, detection and gender labels.*
* *Implement pose estimation, Landmark detection and Face recognition.* 

