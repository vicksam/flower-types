# Flower Image Classifier

In this project I build an image classifier to recognize flower types on images.

### Overview

I build and train the classifier using deep learning. Then I export the model to a simple command line application.

### Methods

* Deep learning

### Technologies

* Python
* Matplotlib
* NumPy
* Pandas
* TensorFlow

## How to use

#### Model

To take a look at the model, open `image_classifier.html` in a browser.

#### Application

Open terminal in the directory with the project and type:
```
python predict.py test-images/orange_dahlia.jpg image_classifier.h5 --top_k 5 --category_names label_map.json
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


##### Requirements
For the application to work, you need to have TensorFlow installed.

## Dataset

The dataset for the project contains images of 102 flower species commonly occuring in the UK. It comes from [Oxford](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

#### Size

For each out of 102 flower types there are between 40 and 258 images.

#### Sample images

<img src="assets/Flowers.png" width="500">
