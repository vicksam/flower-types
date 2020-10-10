# Flower Image Classifier

Have you ever had a problem figuring out the name of the flower? I have and I had a hard time remembering their names.
Fortunately, it is no longer a problem. I created a neural network that does just that. It tells the flower type that is on a picture.

### Overview

I build and train a deep learning model using a pre-trained neural network. Then I create a simple command line application so the model can be used on any computer.

### Methods

* Deep learning

### Technologies

* Python
* Matplotlib
* NumPy
* Pandas
* TensorFlow

## How to use

### Model

To take a look at the notebook, just click on `flower_types.ipynb` and it should open automatically.

Alternatively you can download the project and open the file `flower_types.html` in your browser to see the results.

### Application

##### Requirements

To use the command line application, you need to have TensorFlow installed.

###### Creating an environment with Anaconda

For the ease of use I've included an environment file named `environment.yml`.

It can be used with Anaconda to easily create an environment that I used on any computer.

To create it, open terminal in the directory where you have downloaded the project and type:
```
conda env create -f environment.yml
```

Then you just need to activate it before using the application:
```
conda activate tf_env
```

#### Usage

Open terminal in the directory with the project and type:
```
python predict.py test-images/orange_dahlia.jpg flower_types.h5 --top_k 5 --category_names label_map.json
```
The result will be displayed as:
```
                  Probability
orange dahlia          24.67%
english marigold       20.53%
blanket flower          9.19%
gazania                 5.62%
osteospermum            5.20%

```
You can play with arguments like change `test-images/orange_dahlia.jpg` to `test-images/wild_pansy.jpg` to get prediction for other flower image.

##### Arguments
```
Required:
image_path            input image path
model_path            classifier model path

Optional:
--top_k TOP_K         k value for top k most likely classes to be displayed
--category_names CATEGORY_NAMES
                      path to json file with label-class dictionary

```

## Dataset

The dataset for the project contains images of 102 flower species commonly occuring in the UK. It comes from [Oxford](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

#### Size

For each out of 102 flower types there are between 40 and 258 images. In total there are 8189 images.

#### Sample images

<img src="assets/Flowers.png" width="500">
