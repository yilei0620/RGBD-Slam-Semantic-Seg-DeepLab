

#include<iostream>
using namespace std;

#include "slamBase.h"

#include <opencv2/core/eigen.hpp>

#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>

// Eigen !
#include <Eigen/Core>
#include <Eigen/Geometry>

int main( int argc, char** argv )
{
        ParameterReader pd;
    // 声明两个帧，FRAME结构请见include/slamBase.h
        FRAME frame1, frame2;
    
    //读取图像
        frame1.rgb = cv::imread( "./data/rgb_png/1.png" );
  //      imshow("img", frame1.rgb);
  //      cv::waitKey();
        frame1.depth = cv::imread( "./data/depth_png/1.png", -1);
   //     imshow("img", frame1.depth);
   //     cv::waitKey();
        frame2.rgb = cv::imread( "./data/rgb_png/5.png" );
   //     imshow("img", frame2.rgb);
   //     cv::waitKey();
        frame2.depth = cv::imread( "./data/depth_png/5.png", -1 );
    //    imshow("img", frame2.depth);
    //    cv::waitKey();

    // 提取特征并计算描述子
    cout<<"extracting features"<<endl;
    string detecter = pd.getData( "detector" );

    computeKeyPointsAndDesp( frame1, detecter );
    computeKeyPointsAndDesp( frame2, detecter );

    // 相机内参
    CAMERA_INTRINSIC_PARAMETERS camera;
    camera.fx = atof( pd.getData( "camera.fx" ).c_str());
    camera.fy = atof( pd.getData( "camera.fy" ).c_str());
    camera.cx = atof( pd.getData( "camera.cx" ).c_str());
    camera.cy = atof( pd.getData( "camera.cy" ).c_str());
    camera.scale = atof( pd.getData( "camera.scale" ).c_str() );

    cout<<"solving pnp"<<endl;
    // 求解pnp
    RESULT_OF_PNP result = estimateMotion( frame1, frame2, camera );

    cout<<result.rvec<<endl<<result.tvec<<endl;

    // 处理result
    // 将旋转向量转化为旋转矩阵
    cv::Mat R;
    cv::Rodrigues( result.rvec, R );
    Eigen::Matrix3d r;
    cv::cv2eigen(R, r);
  
    // 将平移向量和旋转矩阵转换成变换矩阵
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    Eigen::AngleAxisd angle(r);
    cout<<"translation"<<endl;
    Eigen::Translation<double,3> trans(result.tvec.at<double>(0,0), result.tvec.at<double>(0,1), result.tvec.at<double>(0,2));
    T = angle;
    T(0,3) = result.tvec.at<double>(0,0); 
    T(1,3) = result.tvec.at<double>(0,1); 
    T(2,3) = result.tvec.at<double>(0,2);

    // 转换点云
    cout<<"converting image to clouds"<<endl;
    PointCloud::Ptr cloud1 = image2PointCloud( frame1.rgb, frame1.depth, camera );
    PointCloud::Ptr cloud2 = image2PointCloud( frame2.rgb, frame2.depth, camera );

    // 合并点云
    cout<<"combining clouds"<<endl;
    PointCloud::Ptr output (new PointCloud());
    pcl::transformPointCloud( *cloud1, *output, T.matrix() );
    *output += *cloud2;
    pcl::io::savePCDFile("data/result.pcd", *output);
    cout<<"Final result saved."<<endl;

    pcl::visualization::CloudViewer viewer( "viewer" );
    viewer.showCloud( output );
    while( !viewer.wasStopped() )
    {
        
    }
    return 0;
}