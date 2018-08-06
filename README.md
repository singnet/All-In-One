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
  * *1.[AFLW dataset](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/) provides a large-scale collection of annotated face images gathered from the web, exhibiting a large variety in appearance (e.g., pose, expression, ethnicity, age, gender) as well as general imaging and environmental conditions.*
  * *2.[IMDB-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) is the largest publicly available dataset of face images with gender and age labels for training.*
  * *3.[CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations.*
  * *4.[Adience dataset](https://talhassner.github.io/home/projects/Adience/Adience-data.html) attempts to capture all the variations in appearance, noise, pose, lighting and more, that can be expected of images taken without careful preparation or posing.*

The network architecture can be found in this [paper](https://arxiv.org/abs/1611.00851). The network is built with deep convolutional layers in keras and is found in nets/model.py.

## Training
  The model can be trained with detection_probablity, kpoints_visibility, key_points, pose, smile, gender_probablity, age_estimation, face_reco for face recognition, is_young, eye_glasses, mouse_slightly_open labels by using the following commands inside the project's directory.

* *`python -m train --dataset aflw --images_path path_to_dataset --label detection --batch_size 100 --steps 500 --ol detection_large1 --os detection_small1 --epochs 10`*

* *`python -m train --dataset wiki --images_path path_to_dataset --label age --batch_size 100 --steps 500 --lr 1e-4  --ol detection_age_large1 --os detection_age_small1 --epochs 10`*

* *`python -m train --dataset wiki --images_path path_to_dataset --label gender --batch_size 100 --steps 500 --lr 1e-5  --ol detection_age_gender_large1 --os detection_age_gender_small1 --load_model path_to_model.json --epochs 10`*

* *`python -m train --dataset celeba --images_path path_to_dataset --label smile --batch_size 100 --steps 500 --lr 1e-5  --ol detection_age_gender_smile_large1 --os detection_age_gender_smile_small1 --load_model path_to_model.json --epochs 10`*

* *`python -m train --dataset celeba --images_path path_to_dataset --label smile --batch_size 100 --steps 500 --lr 1e-5  --ol detection_age_gender_smile_large1 --os detection_age_gender_smile_small1 --load_model path_to_model.json --epochs 10`*

* *`python -m train --dataset wiki --images_path path_to_dataset --label age --batch_size 100 --steps 500 --lr 1e-4 --ol detection_age_large1 --os detection_age_small --epochs 10`*

* *`python -m train --dataset wiki --images_path path_to_dataset --label age --batch_size 100 --steps 500 --lr 1e-4 --ol detection_age_large1 --os detection_age_small --epochs 10`*

* *`python -m train --dataset adience --images_path path_to_dataset --label pose --batch_size 100 --steps 500 --lr 1e-4 --ol detection_age_large1 --os detection_age_small --epochs 10`*

* *`python -m train --dataset adience --images_path path_to_dataset --label detection --batch_size 100 --steps 500 --lr 1e-4 --ol detection_age_large1 --os detection_age_small --epochs 10`*
## To do lists
* *Use CASIA and MORPH dataset for further training the model on age, detection and gender labels.*
