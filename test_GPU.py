## Author: @panthadeep_b
## Created on: 19.Feb.26; 20.47 IST
## Build a 3 layer ANN using Numpy
## Execute RMSprop for training
##Building a simple 3-layer Artificial Neural Network (ANN)

import sys
import os
import numpy as np
import pandas as pd
import math
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt



print(f"TensorFlow version is: {tf.__version__}");
print(f"Torch version is: {torch.__version__}");
print("-----------------")

if torch.accelerator.is_available():
   print("Accelerator is available");
   dev_info = torch.accelerator.current_accelerator().type;
   print(f"Using device {dev_info}"); 
else:
   print("Using CPU");

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")


physical_devices = tf.config.list_physical_devices('GPU')
print(f"Physical devices: {physical_devices}");

if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"TensorFlow is using GPU: {physical_devices[0]}")
else:
    print("No GPU devices found. TensorFlow will run on CPU.")



##Perform matrix multiplication
arr1 = [[1,2,3]];
A1 = np.array(arr1);
print(f"arr1: {A1}");

arr2 = [[4,5,6]];
A2 = np.array(arr2);
print(f"arr2: {A2}");

A1_T = A1.T;
print(f"Transpose of arr1:\n {A1_T}");

x = tf.constant(A1_T);
y = tf.constant(A2);

z = tf.matmul(x,y);
print(f"The matrix mul is:\n {z}");






   





