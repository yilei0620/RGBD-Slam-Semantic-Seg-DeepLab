# Slam combined with semantic segmentation method to remove movable objects

We implement common slam techniques to reconstruct the RGB-d mapping. Since the method can't deal with movable objects, we are trying to use semantic segmentation method to remove those objects during mapping (person, cat, car, bus ...). The semantic segmentation library used is [DeepLab-V2](https://bitbucket.org/aquariusjay/deeplab-public-ver2) [VGG-16](http://liangchiehchen.com/projects/DeepLabv2_vgg.html) based model. 

## Result
The result of common technique of slam:
![alt tag](https://github.com/yilei0620/Slam-Semantic-Seg-DeepLab/blob/master/slam_deepLab/result_slam.png)

The result of method combined with deeplab:
![alt tag](https://github.com/yilei0620/Slam-Semantic-Seg-DeepLab/blob/master/slam_deepLab/result_dp.png)

The result of technique using deep learning method is heavily based on the accuracy of machine learning model. For example, here we can see that we can't remove all configurations of the person in the scene but we can remove his last configuration. This is because the DeepLab can recognize this person in the last one while for previous key frames, it can't because the patch of this person is too small.
![alt tag](https://github.com/yilei0620/Slam-Semantic-Seg-DeepLab/blob/master/slam_deepLab/comparison.png)

## Installation

The package is based on [OpenCV 3.3](http://opencv.org/opencv-3-3.html), [g2o](https://github.com/RainerKuemmerle/g2o) and [DeepLab-V2](https://bitbucket.org/aquariusjay/deeplab-public-ver2). The installation process can be found on those links. The DeepLab package supports CPU only devices (I correct the error when compiling CPU only mode and also change the number of outputs of `MemoryDataLayer` in order to run `CRF layer`).

Compilation of DeepLab is the same as compiling process as [Caffe](http://caffe.berkeleyvision.org/). Please compile the deeplab package provided in the repository.

After install all of above packages, please copy `g2o` and `DeepLab` cmake file in to directory`slam_deeplab/cmake_modules/`. Then, compile the package by standard cmake process:

`mkdir build
cd build
cmake ..
make`


## Experiment
RGB data should be stored in `./data/rgb_png/` and named as `1.png,2.png`. Depth data should be stored in `./data/depth_png/` and named as the same way of RGB images.

`parameters.txt` is for setting parameters for both two methods. Detail please see the file.

`bin/slam` is for common method.
`bin/slamDP` is for learning based method.