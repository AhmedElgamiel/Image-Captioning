# Image-Captioning

[Deep Learning Nanodegree]

## Project Overview
Uses a CNN Encoder and a RNN Decoder to generate captions for input images. The Project has been reviewed by Udacity and graded Meets Specifications.

## Network Architecture

1. CNN encoder based on the ResNet architecture by Google, which encodes the images from COCO data set by Microsoft into the embedded feature vectors.
  ![encoder](https://github.com/AhmedElgamiel/Image-Captioning/blob/main/encoder.png)
2. Decoder, which is a sequential neural network consisting of LSTM units, which translates the feature vector into a sequence of tokens
  ![Decoder](https://github.com/AhmedElgamiel/Image-Captioning/blob/main/decoder.png)
  
## Implementation
After craeating the structure of the model , it is trained for 3 hours on GPU to achieve average loss of about 2% , then I tested it on some image , amd here are some results :

## Results
These are some of the outputs give by the network using the COCO dataset:
![Results](https://github.com/AhmedElgamiel/Image-Captioning/blob/main/example1.png)

