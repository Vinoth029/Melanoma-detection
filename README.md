# Melanoma Detection
Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis. CNN based model needs to be build to accurately detect melanoma.

## Project Pipeline
* [Data Reading](#data-reading)
* [Dataset Creation & Visuvalisation](#dataset-creation-and-visualisation)
* [Base Model](#base-model)
* [Model Building](#model-building)
* [Class Imbalance](#class-imbalance)
* [Final Model](#final-model)
* [Conclusions](#conclusions)
* [Libraries Used](#libraries-used)


### Data Reading

The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.

> The data set contains the following diseases:
- Actinic keratosis
- Basal cell carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented benign keratosis
- Seborrheic keratosis
- Squamous cell carcinoma
- Vascular lesion
 
### Dataset Creation and Visualisation 

- Train & Validation dataset has been created from the train directory with a batch size of 32 and images are resized to 180*180 using tensorflow's `image_dataset_from_directory`.

- A code has been created to visualize one instance of all the nine classes present in the dataset.

![](/images/skin_diseases.png)

### Base Model

A base model has been created initially to detect the 9 classes present in the dataset. While building the model, samples are rescalled to normalize pixel values between (0,1).

> Base model hyper parameters
  - Epocs - 20
  - Optimizer - 'adam'
  - Loss function - Sparse Categorical Crossentropy
 
Base model is clearly overfitting with `training accuracy of 0.8521 and validation accuracy of 0.4609` and it is expected. The Validation loss has been ossilating inbetween 0.45 to 0.55 

![](/images/base_model_accuracy.png)

### Augmentation Strategy

In base model, Its very clear that model is overfitting. In order to address the overfitting issue, the augumentation layer from tensorflow is used.

> Data Augmentation
    In Augumentaion layer, Random flip, Random rotation and Random zoom were introduced to let the model to be more generalized.

### Model Building

The Model has been rebuild with the same hyper parameters with augumentation layer on the top. Apart from the augumentation layer, the dropout layers are also used to address the overfitting problem. 

Overfitting problem has been rectified however the model accuracy is low interms of both training and validation. 

`loss: 1.3155 - accuracy: 0.5435 - val_loss: 1.3338 - val_accuracy: 0.5213`

This is the clear sign of model's underfitting which might have caused due to class imbalance.

![](/images/model_accuracy.png)

### Class Imbalance

To address the underfitting issue, class imbalance has been examined.

>Class Distribution

![](/images/sample_size.png)

From the above plot, its been very abvious that class imbance plays a role here. To address the class imbalance issue, a library called `Augmentor` is used. Augmentor library has been used to create more samples using the same augumentation strategy however this time 500 samples has been created and stored locally. 

> Distribution with updated samples

![](/images/sample_size_after_augumentation.png)

### Final Model

As the class imbalance has been rectified, A final model has been build with the updated samples. This time few changes has been done in the model architecture. 
- Batch Normalization layer added
- Dropout ratio increased
- Augumentation layer removed
- Epohs incresed to 30

![](/images/final_model_accuracy.png)

Model performance has been impoved. Now its not under fitting however overfitting is happening. After 30 epochs the model accuracy and validation accuracy is as follows.

`loss: 0.3098 - accuracy: 0.8869 - val_loss: 1.5546 - val_accuracy: 0.7350`


## Conclusions

Accuracy has got increased after including the augumented samples using Augmentor library.

- Model performance is somewhat good, but still it is overfitting.
- Hyperparameter tuning is required.
- Need to add more convolution layers to the model with appropriate measures for overfitting.
- Need to add more samples or create samples by augumentation methods.
- Need to build the model for more Epocs.


## Libraries Used

``` python

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from glob import glob
import seaborn as sns
import pandas as pd
import numpy as np
import PIL
import cv2
import os

```
