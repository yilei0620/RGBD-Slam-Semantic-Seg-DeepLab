// #define CPU_ONLY = 1;


// #include <caffe/net.hpp>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>

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

#include <stdio.h>  
#include <stdlib.h>  
#include "slamBase.h"
#include <pcl/filters/passthrough.h>

using namespace std;

#define NetTy float 

vector<pair<int, int>> preporcessKeyFrame(vector<FRAME>& keyframes);

FRAME readFrame( int index, ParameterReader& pd );
// estimate the motion between 2 frames
double normofTransform( cv::Mat rvec, cv::Mat tvec );

typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3> >  SlamBlockSolver;
typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver ;

// Judge the relation between 2 frams
enum CHECK_RESULT {NOT_MATCHED=0, TOO_FAR_AWAY, TOO_CLOSE, KEYFRAME}; 
// Judge the relation between 2 frams
CHECK_RESULT checkKeyframes( FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops=false );
//  check nearby loops
void checkNearbyLoops( vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti );
// check random loops
void checkRandomLoops( vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti );

PointCloud::Ptr image2PointClouddeeplab( cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera, cv::Mat& Mask );
cv::Mat computerMask(boost::shared_ptr<caffe::Blob<NetTy>>& layerData, int H, int W, int Orig_H, int Orig_W);

template <typename Dtype>
caffe::Net<Dtype>* loadNet(std::string param_file, std::string pretrained_param_file, caffe::Phase phase)
{
    caffe::Net<Dtype>* net(new caffe::Net<Dtype>(param_file, phase));
    net->CopyTrainedLayersFrom(pretrained_param_file);

    return net;
}

vector<pair<int, int>> preporcessKeyFrame(vector<FRAME>& keyframes){
  FILE *stream;
  vector<pair<int, int>>  data_dim; // first is W, second is H
  vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(3);
  stream = fopen("./data/temp/temp.txt","w+");
  pair<cv::Mat,vector<int>>  processedImg;
  for (int i = 0; i < keyframes.size(); i++){
    fprintf(stream,"keyframe%d.png\n", i+1);
    processedImg = imgpreprocess(keyframes[i].rgb);
    cv::imwrite("./data/temp/keyframe"+ to_string(i+1) + ".png", processedImg.first, compression_params);
    data_dim.push_back(pair<int, int> (processedImg.second[1], processedImg.second[0]) );
  }
  fclose(stream);
  return data_dim;
}






int main(int argc, char** argv) {

  ParameterReader pd;
    int startIndex  =   atoi( pd.getData( "start_index" ).c_str() );
    int endIndex    =   atoi( pd.getData( "end_index"   ).c_str() );
    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(3);

    vector<FRAME > keyframes;
    // initialize smt
    cout<<"Initializing ..."<<endl;

    int currIndex = startIndex; // currIndex
    FRAME currFrame = readFrame( currIndex, pd ); 
    string detector = pd.getData( "detector" );

    CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    computeKeyPointsAndDesp( currFrame, detector );
    PointCloud::Ptr cloud = image2PointCloud( currFrame.rgb, currFrame.depth, camera );
    

    // whether show point cloud
    bool visualize = pd.getData("visualize_pointcloud")==string("yes");
    
    /******************************* 
    // g2o initialization
    *******************************/


    // initialize optimizer
    g2o::SparseOptimizer globalOptimizer; 
    auto linearSolver  = g2o::make_unique<SlamLinearSolver>();
    linearSolver->setBlockOrdering(false);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( g2o::make_unique<SlamBlockSolver>(std::move(linearSolver)) );


    globalOptimizer.setAlgorithm( solver ); 
    globalOptimizer.setVerbose( false );

    //  Add 1st vertex to optimizer
    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setId( currIndex );
    v->setEstimate( Eigen::Isometry3d::Identity() ); // suppose it is an identity matrix
    v->setFixed( true ); //fix the 1st vertex
    globalOptimizer.addVertex( v );

    keyframes.push_back( currFrame );

    double keyframe_threshold = atof( pd.getData("keyframe_threshold").c_str() );
    bool check_loop_closure = pd.getData("check_loop_closure")==string("yes");

    for ( currIndex=startIndex+1; currIndex<endIndex; currIndex++ )
    {
        cout<<"Reading files "<<currIndex<<endl;
        FRAME currFrame = readFrame( currIndex,pd ); 
        computeKeyPointsAndDesp( currFrame, detector );
        // compare currFrame and last key Frame
        CHECK_RESULT result = checkKeyframes( keyframes.back(), currFrame, globalOptimizer ); 

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
                    checkNearbyLoops( keyframes, currFrame, globalOptimizer );
                    checkRandomLoops( keyframes, currFrame, globalOptimizer );
                }
                keyframes.push_back( currFrame );
                break;

        default:
                break;
            }
        
 
    }

    // global optimization
    cout<<"optimizing pose graph, vertices: "<<globalOptimizer.vertices().size()<<endl;
    globalOptimizer.save("./data/result_before.g2o");
    globalOptimizer.initializeOptimization();
    globalOptimizer.optimize( 100 ); //可以指定优化步数
    globalOptimizer.save( "./data/result_after.g2o" );
    cout<<"Optimization done."<<endl;

    vector<pair<int, int>> data_dim = preporcessKeyFrame(keyframes);

     // point cloud joint
    cout<<"saving the point cloud map..."<<endl;
    PointCloud::Ptr output ( new PointCloud() ); 
    PointCloud::Ptr tmp ( new PointCloud() );

    pcl::VoxelGrid<PointT> voxel; // grid filter, estimate resolution
    pcl::PassThrough<PointT> pass; // z-filter, ignore points whose distance are too large
    pass.setFilterFieldName("z");
    pass.setFilterLimits( 0.0, 4.0 ); //ignore 4 meters or further

    double gridsize = atof( pd.getData( "voxel_grid" ).c_str() ); //find the resolution
    voxel.setLeafSize( gridsize, gridsize, gridsize );
 ////////////////////////////
/*
 Load deeplab model and initialize
  */
    /////////////////
    string model = "./model/test.prototxt";
    string weights = "./model/train2_iter_20000.caffemodel";

  caffe::Net<NetTy>* _net = loadNet<NetTy>(model, weights, caffe::TEST); // load structure and weights
  boost::shared_ptr<caffe::Blob<NetTy>> layerData;
  cout<<"loading weights completed!"<<endl;

   int Orig_W = 0, Orig_H = 0;
  if (keyframes.size() > 0){
     Orig_W =  keyframes[0].rgb.cols;
      Orig_H = keyframes[0].rgb.rows;
  }
  cv::Mat Mask;
    for (size_t i=0; i<keyframes.size(); i++)
    {  
        // estimate mask
       _net->ForwardPrefilled();                    // compute once                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
      layerData = _net->blob_by_name("crf_interp_argmax");
      Mask =  computerMask(layerData, data_dim[i].second, data_dim[i].first , Orig_H, Orig_W);
      cv::imwrite("./data/seg/keyframe"+ to_string(i+1) + ".png", Mask,  compression_params);
        // pick up a frame from g2o
        g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( keyframes[i].frameID ));
        Eigen::Isometry3d pose = vertex->estimate(); //the optimized position
        PointCloud::Ptr newCloud = image2PointClouddeeplab( keyframes[i].rgb, keyframes[i].depth, camera, Mask ); //transfer it to point cloud
        // filter
        voxel.setInputCloud( newCloud );
        voxel.filter( *tmp );
        pass.setInputCloud( tmp );
        pass.filter( *newCloud );
        // join point cloud
        pcl::transformPointCloud( *newCloud, *tmp, pose.matrix() );
        *output += *tmp;
        tmp->clear();
        newCloud->clear();
    }

    voxel.setInputCloud( output );
    voxel.filter( *tmp );
    //store
    pcl::io::savePCDFile( "./result.pcd", *tmp );
    globalOptimizer.clear();
    
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

CHECK_RESULT checkKeyframes( FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops)
{
    static ParameterReader pd;
    static int min_inliers = atoi( pd.getData("min_inliers").c_str() );
    static double max_norm = atof( pd.getData("max_norm").c_str() );
    static double keyframe_threshold = atof( pd.getData("keyframe_threshold").c_str() );
    static double max_norm_lp = atof( pd.getData("max_norm_lp").c_str() );
    static CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    // 比较f1 和 f2
    RESULT_OF_PNP result = estimateMotion( f1, f2, camera );
    if ( result.inliers < min_inliers ) //inliers不够，放弃该帧
        return NOT_MATCHED;
    // 计算运动范围是否太大
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
    // 向g2o中增加这个顶点与上一帧联系的边
    // 顶点部分
    // 顶点只需设定id即可
    if (is_loops == false)
    {
        g2o::VertexSE3 *v = new g2o::VertexSE3();
        v->setId( f2.frameID );
        v->setEstimate( Eigen::Isometry3d::Identity() );
        opti.addVertex(v);
    }
    // 边部分
    g2o::EdgeSE3* edge = new g2o::EdgeSE3();
    // 连接此边的两个顶点id
    edge->setVertex( 0, opti.vertex(f1.frameID ));
    edge->setVertex( 1, opti.vertex(f2.frameID ));
    edge->setRobustKernel( new g2o::RobustKernelHuber() );
    // 信息矩阵
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix< double, 6,6 >::Identity();
    // 信息矩阵是协方差矩阵的逆，表示我们对边的精度的预先估计
    // 因为pose为6D的，信息矩阵是6*6的阵，假设位置和角度的估计精度均为0.1且互相独立
    // 那么协方差则为对角为0.01的矩阵，信息阵则为100的矩阵
    information(0,0) = information(1,1) = information(2,2) = 100;
    information(3,3) = information(4,4) = information(5,5) = 100;
    // 也可以将角度设大一些，表示对角度的估计更加准确
    edge->setInformation( information );
    // 边的估计即是pnp求解之结果
    Eigen::Isometry3d T = cvMat2Eigen( result.rvec, result.tvec );
    // edge->setMeasurement( T );
    edge->setMeasurement( T.inverse() );
    // 将此边加入图中
    opti.addEdge(edge);
    return KEYFRAME;
}

void checkNearbyLoops( vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti )
{
    static ParameterReader pd;
    static int nearby_loops = atoi( pd.getData("nearby_loops").c_str() );
    
    // 就是把currFrame和 frames里末尾几个测一遍
    if ( frames.size() <= nearby_loops )
    {
        // no enough keyframes, check everyone
        for (size_t i=0; i<frames.size(); i++)
        {
            checkKeyframes( frames[i], currFrame, opti, true );
        }
    }
    else
    {
        // check the nearest ones
        for (size_t i = frames.size()-nearby_loops; i<frames.size(); i++)
        {
            checkKeyframes( frames[i], currFrame, opti, true );
        }
    }
}

void checkRandomLoops( vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti )
{
    static ParameterReader pd;
    static int random_loops = atoi( pd.getData("random_loops").c_str() );
    srand( (unsigned int) time(NULL) );
    // 随机取一些帧进行检测
    
    if ( frames.size() <= random_loops )
    {
        // no enough keyframes, check everyone
        for (size_t i=0; i<frames.size(); i++)
        {
            checkKeyframes( frames[i], currFrame, opti, true );
        }
    }
    else
    {
        // randomly check loops
        for (int i=0; i<random_loops; i++)
        {
            int index = rand()%frames.size();
            checkKeyframes( frames[index], currFrame, opti, true );
        }
    }
}

cv::Mat computerMask(boost::shared_ptr<caffe::Blob<NetTy>>& layerData, int H, int W, int Orig_H, int Orig_W){
  cv::Mat lab = cv::Mat::zeros(cv::Size(513,513),CV_32F);
  const NetTy* pstart = layerData->cpu_data(); // res5_6->cpu_data()返回的是多维数据（数组）
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
    cv::Mat Mask =  lab(cv::Rect(0,0,H,W));
    cv::resize(Mask,Mask,cv::Size(Orig_W,Orig_H));
    return Mask;
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
                 p.y = (r - camera.cy) * p.z / camera.fy;

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