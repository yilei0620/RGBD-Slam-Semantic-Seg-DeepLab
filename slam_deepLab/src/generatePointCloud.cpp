#include <iostream>
#include <string>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
 
// 定义点云类型
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud; 

 // 相机内参
const double camera_factor = 1000;
const double camera_cx = 325.5;
const double camera_cy = 253.5;
const double camera_fx = 518.0;
const double camera_fy = 519.0;


int main( int argc, char ** argv){
	cv::Mat rgb, depth;

	rgb = cv::imread("./data/rgb_png/1.png");
	depth = cv::imread("./data/depth_png/1.png",-1);

	PointCloud::Ptr cloud (new PointCloud);

	for (int r = 0; r < depth.rows; r++){
		for (int c = 0; c < depth.cols; c++){
			ushort d = depth.ptr<ushort > (r) [c];
			if (d == 0) 
				continue;
			PointT p;

			p.z = double(d) /camera_factor;
			p.x = (c - camera_cx) * p.z / camera_fx;
		             p.y = (r - camera_cy) * p.z / camera_fy;

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
	pcl::io::savePCDFile("./pointcloud.pcd", *cloud);

	cloud->points.clear();
	cout<<"point cloud is saved."<<endl;

	return 0;
}