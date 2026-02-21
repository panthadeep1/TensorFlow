## Author: @panthadeep_b
## Created on: 1.Jan.26; 23.40 IST
## Introduction to Tensor Flow
## WAP to Create your first Graph and Running it in a session

import torch
import tensorflow as tf
import math
print("Tensor-Flow version:", tf.__version__)
print("Torch version:", torch.__version__)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

x = tf.Variable(3, name="x");
y = tf.Variable(4, name="y");

f = pow(x,y);

print(f);
