# Slam combined with semantic segmentation method to remove movable objects

We implement common slam techniques to reconstruct the RGB-d mapping. Since the method can't deal with movable objects, we are trying to use semantic segmentation method to remove those objects during mapping (person, cat, car, bus ...). The semantic segmentation library used is [DeepLab-V2](https://bitbucket.org/aquariusjay/deeplab-public-ver2) [VGG-16](http://liangchiehchen.com/projects/DeepLabv2_vgg.html) based model. 

## Result


The result of common technique of slam:
![alt tag](https://github.com/yilei0620/RGBD-Slam-Semantic-Seg-DeepLab/blob/master/slam_deepLab/result_slam.png)

The result of method combined with deeplab:
![alt tag](https://github.com/yilei0620/RGBD-Slam-Semantic-Seg-DeepLab/blob/master/slam_deepLab/result_dp.png)

## Failed Result
The result of technique using deep learning method is heavily based on the accuracy of machine learning model. For example, here we can see that we can't remove all configurations of the person in the scene but we can remove his last configuration. This is because the DeepLab can't recognize this person when just a part of him that can be seen. In the future, we should try some better semantic segmentation model such as [DeepLab-V3](https://arxiv.org/abs/1706.05587) or [PsPNet](https://arxiv.org/abs/1612.01105).
![alt tag](https://github.com/yilei0620/RGBD-Slam-Semantic-Seg-DeepLab/blob/master/slam_deepLab/failed_exp.png)

We also can try real time semantic segmentation model such as [ICNet](https://arxiv.org/abs/1704.08545) to help with filtering movable objects' features in order to improve PnP RANSAC accuracy and efficiency.


## Installation

The package is based on [OpenCV 3.3](http://opencv.org/opencv-3-3.html), [Ceres](http://ceres-solver.org/index.html) and [DeepLab-V2](https://bitbucket.org/aquariusjay/deeplab-public-ver2). The installation process can be found on those links. The DeepLab package supports CPU only devices (I correct the error when compiling CPU only mode and also change the number of outputs of `MemoryDataLayer` in order to run `CRF layer`).

Compilation of DeepLab is the same as compiling process as [Caffe](http://caffe.berkeleyvision.org/). Please compile the deeplab package provided in the repository.

After install all of above packages, please copy `Ceres` and `DeepLab` cmake file in to directory`slam_deeplab/cmake_modules/`. Then, compile the package by standard cmake process:

`mkdir build
cd build
cmake ..
make`

The package was tested on Ubuntu 14.04 and 16.04.
When compiling in the Ubuntu 16.04, please edit `src/Cmakelist.txt`. Details please see the file.


## Experiment
RGB data should be stored in `./data/rgb_png/` and named as `1.png,2.png`. Depth data should be stored in `./data/depth_png/` and named as the same way of RGB images.

`parameters.txt` is for setting parameters for both two methods. Detail please see the file.

`bin/slam` is for common method.
`bin/slamDP` is for learning based method.
