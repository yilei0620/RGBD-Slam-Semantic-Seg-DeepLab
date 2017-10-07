#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include <pcl/filters/passthrough.h>


#include "slamBase.h"
#include "ceres/ceres.h"
#include "ceresbase.h"

#include <caffe/caffe.hpp>
#include <boost/algorithm/string.hpp>
#include <google/protobuf/text_format.h>
#include <caffe/blob.hpp>
#include <caffe/common.hpp>
#include <caffe/net.hpp>
#include <caffe/proto/caffe.pb.h>
#include <caffe/util/io.hpp>
#include <caffe/layers/memory_data_layer.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#define NetTy float 

// Input an index，read corresponding frame
FRAME readFrame( int index, ParameterReader& pd );
// estimate the motion between 2 frames
double normofTransform( cv::Mat rvec, cv::Mat tvec );

// Judge the relation between 2 frams
enum CHECK_RESULT {NOT_MATCHED=0, TOO_FAR_AWAY, TOO_CLOSE, KEYFRAME}; 
// Judge the relation between 2 frams
CHECK_RESULT checkKeyframes( FRAME& f1, FRAME& f2, ceres::examples::MapOfPoses* poses,
            ceres::examples::VectorOfConstraints* constraints , bool is_loops=false );
//  check nearby loops
void checkNearbyLoops( vector<FRAME>& frames, FRAME& currFrame,  ceres::examples::MapOfPoses* poses,
            ceres::examples::VectorOfConstraints* constraints );
// check random loops
void checkRandomLoops( vector<FRAME>& frames, FRAME& currFrame, ceres::examples::MapOfPoses* poses,
            ceres::examples::VectorOfConstraints* constraints );
PointCloud::Ptr image2PointClouddeeplab( cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera, cv::Mat& Mask );

cv::Mat computerMask(caffe::Net<NetTy>* _net, pair<cv::Mat,vector<int > >&  processedImg,  int Orig_H, int Orig_W);

template <typename Dtype>
caffe::Net<Dtype>* loadNet(std::string param_file, std::string pretrained_param_file, caffe::Phase phase)
{
    caffe::Net<Dtype>* net(new caffe::Net<Dtype>(param_file, phase));
    net->CopyTrainedLayersFrom(pretrained_param_file);

    return net;
}


int main( int argc, char** argv )
{

    ParameterReader pd;
    int startIndex  =   atoi( pd.getData( "start_index" ).c_str() );
    int endIndex    =   atoi( pd.getData( "end_index"   ).c_str() );

    vector<FRAME > keyframes;
    // initialize smt
    cout<<"Initializing ..."<<endl;

    int currIndex = startIndex; // currIndex
    FRAME currFrame = readFrame( currIndex, pd ); 
    string detector = pd.getData( "detector" );

    CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    computeKeyPointsAndDesp( currFrame, detector );
    
    /******************************* 
    // ceres initialization
    *******************************/
   ceres::examples::MapOfPoses poses;
   ceres::examples::VectorOfConstraints constraints;


    // initialize optimizer
    //  Add 1st vertex to optimizer
   AddVertex(currIndex, Eigen::Vector3d(0,0,0), Eigen::Quaterniond(1, 0 , 0, 0),  &poses);
    keyframes.push_back( currFrame );

    double keyframe_threshold = atof( pd.getData("keyframe_threshold").c_str() );
    bool check_loop_closure = pd.getData("check_loop_closure")==string("yes");

    for ( currIndex=startIndex+1; currIndex<endIndex; currIndex++ )
    {
        cout<<"Reading files "<<currIndex<<endl;
        FRAME currFrame = readFrame( currIndex,pd ); 
        computeKeyPointsAndDesp( currFrame, detector );
        // compare currFrame and last key Frame
        CHECK_RESULT result = checkKeyframes( keyframes.back(), currFrame, &poses, &constraints  ); 

        switch (result){
        case NOT_MATCHED:
            cout<<RED"Not enough inliers."<<endl;
            break;
        case TOO_FAR_AWAY:
            cout<<RED"Too far away, may be an error."<<endl;
            break;

        case TOO_CLOSE:
            cout<<RESET"Too close, not a keyframe"<<endl;
            break;

        case KEYFRAME:
            cout<<GREEN"Frame ID "<<currIndex<<GREEN" is a new keyframe "<<endl;
            // if it is a key frame, check loops
            if (check_loop_closure)
                {
                    checkNearbyLoops( keyframes, currFrame, &poses, &constraints  );
                    checkRandomLoops( keyframes, currFrame, &poses, &constraints  );
                }
                keyframes.push_back( currFrame );
                break;

        default:
                break;
            }
        
    }

    // global optimization

   CHECK(ceres::examples::OutputPoses("./data/poses_original.txt", poses))
      << "Error outputting to poses_original.txt";
   ceres::Problem problem;
  ceres::examples::BuildOptimizationProblem(constraints, &poses, &problem);
  CHECK(ceres::examples::SolveOptimizationProblem(&problem))
      << "The solve was not successful, exiting.";

  CHECK(ceres::examples::OutputPoses("./data/poses_optimized.txt", poses))
      << "Error outputting to poses_original.txt";


     // point cloud joint
    cout<<"saving the point cloud map..."<<endl;
    PointCloud::Ptr output ( new PointCloud() ); 
    PointCloud::Ptr tmp ( new PointCloud() );

    pcl::VoxelGrid<PointT> voxel; // grid filter, estimate resolution
    pcl::PassThrough<PointT> pass; // z-filter, ignore points whose distance are too large
    pass.setFilterFieldName("z");
    pass.setFilterLimits( 0.0, 6.0 ); //ignore 4 meters or further

    double gridsize = atof( pd.getData( "voxel_grid" ).c_str() ); //find the resolution
    voxel.setLeafSize( gridsize, gridsize, gridsize );

     string model = "./model/test.prototxt";
    string weights = "./model/train2_iter_20000.caffemodel";

  caffe::Net<NetTy>* _net = loadNet<NetTy>(model, weights, caffe::TEST); // load structure and weights
  cout<<"loading weights completed!"<<endl;

   int Orig_W = 0, Orig_H = 0;
  if (keyframes.size() > 0){
     Orig_W =  keyframes[0].rgb.cols;
      Orig_H = keyframes[0].rgb.rows;
  }
  cv::Mat Mask;
  pair<cv::Mat,vector<int>>  processedImg;

    Eigen::Matrix4d pose;
    for (size_t i=0; i<keyframes.size(); i++)
    {
        processedImg = imgpreprocess(keyframes[i].rgb);             
      Mask =  computerMask(_net, processedImg , Orig_H, Orig_W);
        pose = ReadOptimizedPose(keyframes[i].frameID,  poses );
        PointCloud::Ptr newCloud = image2PointClouddeeplab( keyframes[i].rgb, keyframes[i].depth, camera, Mask ); //transfer it to point cloud
        // filter
        voxel.setInputCloud( newCloud );
        voxel.filter( *tmp );
        pass.setInputCloud( tmp );
        pass.filter( *newCloud );
        // join point cloud
        pcl::transformPointCloud( *newCloud, *tmp, pose);
        *output += *tmp;
        tmp->clear();
        newCloud->clear();
    }

    voxel.setInputCloud( output );
    voxel.filter( *tmp );
    //store
    pcl::io::savePCDFile( "./result.pcd", *tmp );
    
    cout<<"Final map is saved."<<endl;
    return 0;
    
}

FRAME readFrame( int index, ParameterReader& pd )
{
    FRAME f;
    string rgbDir   =   pd.getData("rgb_dir");
    string depthDir =   pd.getData("depth_dir");
    
    string rgbExt   =   pd.getData("rgb_extension");
    string depthExt =   pd.getData("depth_extension");

    stringstream ss;
    ss<<rgbDir<<index<<rgbExt;
    string filename;
    ss>>filename;
    f.rgb = cv::imread( filename );

    ss.clear();
    filename.clear();
    ss<<depthDir<<index<<depthExt;
    ss>>filename;

    f.depth = cv::imread( filename, -1 );
    f.frameID = index;
    return f;
}

double normofTransform( cv::Mat rvec, cv::Mat tvec )
{
    return fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec)))+ fabs(cv::norm(tvec));
}

CHECK_RESULT checkKeyframes( FRAME& f1, FRAME& f2,  ceres::examples::MapOfPoses* poses,
            ceres::examples::VectorOfConstraints* constraints, bool is_loops)
{
    static ParameterReader pd;
    static int min_inliers = atoi( pd.getData("min_inliers").c_str() );
    static double max_norm = atof( pd.getData("max_norm").c_str() );
    static double keyframe_threshold = atof( pd.getData("keyframe_threshold").c_str() );
    static double max_norm_lp = atof( pd.getData("max_norm_lp").c_str() );
    static CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    // compare f1 and f2
    RESULT_OF_PNP result = estimateMotion( f1, f2, camera );
    if ( result.inliers < min_inliers )  //not enough inliers and give up this frame
        return NOT_MATCHED;
    // measure the motion
    double norm = normofTransform(result.rvec, result.tvec);
    if ( is_loops == false )
    {
        if ( norm >= max_norm )
            return TOO_FAR_AWAY;   // too far away, may be error
    }
    else
    {
        if ( norm >= max_norm_lp)
            return TOO_FAR_AWAY;
    }

    if ( norm <= keyframe_threshold )
        return TOO_CLOSE;   // too adjacent frame
    // set vertex
    if (is_loops == false)
    {
        AddVertex(f2.frameID, Eigen::Vector3d(0,0,0), Eigen::Quaterniond(1, 0 , 0, 0),  poses);
    }
    // set edge
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix< double, 6,6 >::Identity();
    // information matrix 6*6
    information(0,0) = information(1,1) = information(2,2) = 100;
    information(3,3) = information(4,4) = information(5,5) = 100;
    Eigen::Isometry3d T = cvMat2Eigen( result.rvec, result.tvec );
    T = T.inverse() * Eigen::Isometry3d::Identity() ;
    Eigen::Matrix3d rotation_matrix;
     rotation_matrix << T(0,0) , T(0,1) , T(0,2) ,
                                    T(1,0) , T(1,1) , T(1,2),
                                    T(2,0) , T(2,1) , T(2,2);
    Eigen::Quaterniond Q = Eigen::Quaterniond( rotation_matrix );
    AddEgde( f1.frameID,  f2.frameID  ,  Eigen::Vector3d( T(0,3),  T(1,3) ,   T(2,3) ) , Q,  information,  constraints );

    return KEYFRAME;
}

void checkNearbyLoops( vector<FRAME>& frames, FRAME& currFrame, ceres::examples::MapOfPoses* poses,
            ceres::examples::VectorOfConstraints* constraints)
{
    static ParameterReader pd;
    static int nearby_loops = atoi( pd.getData("nearby_loops").c_str() );
    
    //near loop checking
    if ( frames.size() <= nearby_loops )
    {
        // no enough keyframes, check everyone
        for (size_t i=0; i<frames.size(); i++)
        {
            checkKeyframes( frames[i], currFrame,  poses, constraints, true );
        }
    }
    else
    {
        // check the nearest ones
        for (size_t i = frames.size()-nearby_loops; i<frames.size(); i++)
        {
            checkKeyframes( frames[i], currFrame, poses, constraints , true );
        }
    }
}

void checkRandomLoops( vector<FRAME>& frames, FRAME& currFrame, ceres::examples::MapOfPoses* poses,
            ceres::examples::VectorOfConstraints* constraints )
{
    static ParameterReader pd;
    static int random_loops = atoi( pd.getData("random_loops").c_str() );
    srand( (unsigned int) time(NULL) );
    // random closure checking
    
    if ( frames.size() <= random_loops )
    {
        // no enough keyframes, check everyone
        for (size_t i=0; i<frames.size(); i++)
        {
            checkKeyframes( frames[i], currFrame, poses, constraints , true );
        }
    }
    else
    {
        // randomly check loops
        for (int i=0; i<random_loops; i++)
        {
            int index = rand()%frames.size();
            checkKeyframes( frames[index], currFrame, poses, constraints , true );
        }
    }
}

cv::Mat computerMask(caffe::Net<NetTy>* _net, pair<cv::Mat,vector<int > >&  processedImg,  int Orig_H, int Orig_W){
  int aft_H = processedImg.second[1], aft_W = processedImg.second[0];
  std::vector<cv::Mat> dv = { processedImg.first }; // We still need vectors as input
  std::vector<int> label = { 0 };
  std::vector<pair<int, int >  > dim = { pair<int, int> (aft_H,   aft_W) };
  
  caffe::MemoryDataLayer<NetTy> *m_layer_ = (caffe::MemoryDataLayer<NetTy> *)_net->layers()[0].get();
  m_layer_->AddMatVector(dv, label, dim);

  int end_ind = _net->layers().size();
  std::vector<caffe::Blob<NetTy>*> input_vec;
    _net->Forward(input_vec);
  boost::shared_ptr<caffe::Blob<NetTy>> layerData = _net->blob_by_name("crf_interp_argmax");
  const NetTy* pstart = layerData->cpu_data(); // layerData->cpu_data() returns a pointer to an array
  cv::Mat lab = cv::Mat::zeros(cv::Size(513,513),CV_32F);
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
    return lab;
}

PointCloud::Ptr image2PointClouddeeplab( cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera, cv::Mat& Mask ){
    PointCloud::Ptr cloud (new PointCloud);
  for (int r = 0; r < depth.rows; r++){
    for (int c = 0; c < depth.cols; c++){
      ushort d = depth.ptr<ushort > (r) [c];
      if (d == 0 | fabs(Mask.at<float>(r,c) - 0) > 1E-6) 
        continue;
      PointT p;

      p.z = double(d) /camera.scale;
      p.x = (c - camera.cx) * p.z / camera.fx;
                 p.y =  (r - camera.cy) * p.z / camera.fy;

                 p.b = rgb.ptr<uchar> (r) [c * 3];
                 p.g = rgb.ptr<uchar> (r) [c * 3  + 1];
                 p.r = rgb.ptr<uchar> (r) [c * 3  + 2];

                 cloud->points.push_back( p);

    }
  }
  cloud->height = 1;
  cloud->width = cloud->points.size();

  cout<<"point cloud size :" << cloud->points.size()<<endl;
  cloud->is_dense = false;
  return cloud;
}