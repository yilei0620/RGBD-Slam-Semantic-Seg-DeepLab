#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <caffe/caffe.hpp>
#include <boost/algorithm/string.hpp>
#include <google/protobuf/text_format.h>
#include <caffe/blob.hpp>
#include <caffe/common.hpp>
#include <caffe/net.hpp>
#include <caffe/proto/caffe.pb.h>
#include <caffe/util/io.hpp>
#include <caffe/layers/memory_data_layer.hpp>

#include <iostream>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>


using namespace std;

#define NetTy float 


pair<cv::Mat,vector<int>>  imgpreprocess (cv::Mat img);

template <typename Dtype>
caffe::Net<Dtype>* loadNet(std::string param_file, std::string pretrained_param_file, caffe::Phase phase)
{
    caffe::Net<Dtype>* net(new caffe::Net<Dtype>(param_file, phase));
    net->CopyTrainedLayersFrom(pretrained_param_file);

    return net;
}

pair<cv::Mat,vector<int>>  imgpreprocess (cv::Mat img){
  // We need to resize the input image to size 513 * 513 . The resized image shouldn't distort the image
  cv::Mat tmp, complement, newimg;
  vector<cv::Mat> channels;
  int height = img.rows, width = img.cols, newW,newH;
  cout<<"Image sizes are "<<width<<","<<height<<endl;
    if (width > height){
      newW = 513;
      newH = height*513/width;
      cv::resize(img,tmp,cv::Size(newW, newH));
      complement =  cv::Mat::zeros(cv::Size(513,513 - height*513/width),CV_8UC3);
      newimg = mergeRows(tmp,complement);
    }
    else{
      newW = width*513/height;
      newH = 513;
      cv::resize(img,tmp,cv::Size(newW,newH));
      complement =  cv::Mat::zeros(cv::Size(513 - width*513/height,513 ),CV_8UC3);
      newimg = mergeCols(tmp,complement);
    }
    return pair<cv::Mat, vector<int> > (newimg,vector<int> {newW, newH});
}


int main(int argc, char** argv) {
	string model = "test.prototxt";
    	string weights = "./model/train2_iter_20000.caffemodel";
    	vector<int> compression_params;
  	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  	compression_params.push_back(3);

    	cv::Mat img = cv::imread("./data/rgb_png/56.png",1);
      int Orig_H = img.rows, Orig_W = img.cols;
      pair<cv::Mat,vector<int>> processimg = imgpreprocess(img); // We need to resize the input image with size 513 * 513 
      cv::imwrite("./data/resize_img.png", processimg.first,  compression_params);
      int aft_H = processimg.second[1], aft_W = processimg.second[0];

      std::vector<cv::Mat> dv = { processimg.first }; // We still need vectors as input
      std::vector<int> label = { 0 };
      std::vector<pair<int, int >  > dim = { pair<int, int> (aft_H,   aft_W) };

    caffe::Net<NetTy>* _net = loadNet<NetTy>(model, weights, caffe::TEST); // load structure and weights
  
    cout<<"loading weights completed!"<<endl;
    caffe::MemoryDataLayer<NetTy> *m_layer_ = (caffe::MemoryDataLayer<NetTy> *)_net->layers()[0].get();
    m_layer_->AddMatVector(dv, label, dim);

    int end_ind = _net->layers().size();
    std::vector<caffe::Blob<NetTy>*> input_vec;
    _net->Forward(input_vec);
    boost::shared_ptr<caffe::Blob<NetTy>> layerData = _net->blob_by_name("crf_interp_argmax");
      cv::Mat lab = cv::Mat::zeros(cv::Size(513,513),CV_32F);
  const NetTy* pstart = layerData->cpu_data(); // layerData->cpu_data() returns a pointer to an array
  float tmp;
  
      for (int i = 0; i < 513; i++){
           for (int j = 0; j< 513; j++){
             tmp = *pstart;
             if (fabs(tmp - 5) <= 1E-6 | fabs(tmp - 9) <= 1E-6 | fabs(tmp- 11 ) <=1E-6 | fabs(tmp - 16) <=1E-6 | fabs(tmp - 18) <= 1E-6 | fabs(tmp - 20) <= 1E-6 | fabs(tmp) <= 1E-6){
               lab.at<float>(i,j) = 0;
             }
             else{
              lab.at<float>(i,j) = 100;
             }
             pstart++;
           }
      }

   lab = lab(cv::Rect(0,0, aft_W, aft_H));
      cv::resize(lab,lab,cv::Size(Orig_W,Orig_H));
   cv::imwrite("./data/result.png", lab,  compression_params);
  	return 0;
}