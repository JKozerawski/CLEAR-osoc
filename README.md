# CLEAR: Cumulative LEARnining One Shot One Class Image Recognition

This repository contains the code (Caffe) for "[CLEAR: Cumulative LEARnining One Shot One Class Image Recognition](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/3412.pdf)" paper (CVPR 2018) by [Jedrzej Kozerawski](https://github.com/JKozerawski/) and [Matthew Turk](http://www.cs.ucsb.edu/~mturk/).

### Citation
```
@inproceedings{kozerawski2018clear,
  title={CLEAR: Cumulative LEARning for One-Shot One-Class Image Recognition},
  author={Kozerawski, Jedrzej and Turk, Matthew},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3446--3455},
  year={2018}
}
```

## Introduction
This work addresses the novel problem of one-shot one-class classification. The goal is to estimate a classification decision boundary for a novel class based on a single image example. Our method exploits transfer learning to model the transformation from a representation of the input, extracted by a Convolutional Neural Network, to a classification decision boundary. We use a deep neural network to learn this transformation from a large labelled dataset of images and their associated class decision boundaries generated from ImageNet, and then apply the learned decision boundary to classify subsequent query images. We tested our approach on several benchmark datasets and significantly outperformed the baseline methods.

<img align="center" src="https://github.com/JKozerawski/CLEAR-osoc/blob/master/clear_images/train_pipeline.png">

<img align="center" src="https://github.com/JKozerawski/CLEAR-osoc/blob/master/clear_images/test_pipeline.png">

## Usage

### Dependencies
- Caffe
- Numpy
- ILSVRC2012 (for training)

### Train
TO DO

### Pretrained model

Tar package with Caffe model files is available [here](https://drive.google.com/file/d/1KRPXw5clTRveG27ro-JEe-GHbzfA28Kp/view?usp=sharing)

### Test
In the test.py file we use feature extraction script (using Inception V1 network) to do preprocessing of images.
If you want to use our test.py file - please download the Inception V1 .caffemodel and deploy files available [here](https://drive.google.com/file/d/1WctmdPPkMCu7XFuAFixruG_a55grGiFP/view?usp=sharing). Make sure to point to the directory where the model is with the --inception_model_dir flag when using the test.py

To test the network just run our example:
```
python test.py --model_dir pathToTheCLEARNetwork --inception_model_dir pathToTheInceptionNetwork --image imagePath --imageR pathToTheRightImage
```
