

#include "slamBase.h"


PointCloud::Ptr image2PointCloud( cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera ){
	PointCloud::Ptr cloud (new PointCloud);

	for (int r = 0; r < depth.rows; r++){
		for (int c = 0; c < depth.cols; c++){
			ushort d = depth.ptr<ushort > (r) [c];
			if (d == 0) 
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

// point2dTo3d 
// input: 3d point (float) Point3f
cv::Point3f point2dTo3d( cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera ){
	cv::Point3f p; // 3D 点
    	p.z = double( point.z ) / camera.scale;
   	 p.x = ( point.x - camera.cx) * p.z / camera.fx;
    	p.y = ( point.y - camera.cy) * p.z / camera.fy;
    	return p;

}

// computeKeyPointsAndDesp: find key points and their corresponding descriptor
void computeKeyPointsAndDesp( FRAME& frame, string detector ){
	cv::Ptr<cv::Feature2D> _detector;
	if (detector.compare("SIFT") == 0 || detector.compare("sift") == 0 )
		_detector = cv::xfeatures2d::SIFT::create();
	else if (detector.compare("SURF") == 0 || detector.compare("surf") == 0 )
		_detector = cv::xfeatures2d::SURF::create();
	else if (detector.compare("ORB") == 0 || detector.compare("orb") == 0 )
		_detector = cv::ORB::create(1000);
	else {
		_detector = cv::xfeatures2d::SIFT::create();
	}


   	 _detector->detect( frame.rgb, frame.kp );
   	 _detector->compute( frame.rgb, frame.kp, frame.desp );
    	return;
}

// estimateMotion : measure the motion between 2 frames
// 
// output：rotation vector and translate vector
RESULT_OF_PNP estimateMotion( FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera )
{
	static ParameterReader pd;
	vector< cv::DMatch > matches;
    	cv::BFMatcher matcher;
    	matcher.match( frame1.desp, frame2.desp, matches );
   
    	cout<<"find total "<<matches.size()<<" matches."<<endl;
    	vector< cv::DMatch > goodMatches;
    	double minDis = 9999;
    	double good_match_threshold = atof( pd.getData( "good_match_threshold" ).c_str() );
    	for ( size_t i=0; i<matches.size(); i++ )
    	{
       	 if ( matches[i].distance < minDis )
            		minDis = matches[i].distance;
   	 }
   	if ( minDis < 10 ) 
        		minDis = 10;

    	for ( size_t i=0; i<matches.size(); i++ ){
       	 	if (matches[i].distance < good_match_threshold*minDis)
            			goodMatches.push_back( matches[i] );
    }

    	cout<<"good matches: "<<goodMatches.size()<<endl;
    	RESULT_OF_PNP result;
    	if (goodMatches.size() <= 6) {
    		result.inliers = 0;
    		return result;
    	}

    	vector<cv::Point3f> pts_obj;

   	 vector< cv::Point2f > pts_img;


	    for (size_t i=0; i<goodMatches.size(); i++)
	    {
	        // query for f1, train for f2
	        cv::Point2f p = frame1.kp[goodMatches[i].queryIdx].pt;
	        ushort d = frame1.depth.ptr<ushort>( int(p.y) )[ int(p.x) ];
	        if (d == 0)
	            continue;
	        pts_img.push_back( cv::Point2f( frame2.kp[goodMatches[i].trainIdx].pt ) );


	        cv::Point3f pt ( p.x, p.y, d );
	        cv::Point3f pd = point2dTo3d( pt, camera );
	        pts_obj.push_back( pd );
	    }

	    double camera_matrix_data[3][3] = {
	        {camera.fx, 0, camera.cx},
	        {0, camera.fy, camera.cy},
	        {0, 0, 1}
	    };

	    cout<<"solving pnp"<<endl;
	    // build camera intrisinc matrix
	    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
	    cv::Mat rvec, tvec, inliers;
	    // solve pnp
	    try{
	    	cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 0.99, inliers );
	    	result.rvec = rvec;
	    	result.tvec = tvec;
	    	result.inliers = inliers.rows;
	    }
	    catch (...){
	    	cout<<"PnPSlover Error, Bad matching, jump to next frame!"<<endl;
	    }
	    

	    
	    

	    return result;
}

Eigen::Isometry3d cvMat2Eigen( cv::Mat& rvec, cv::Mat& tvec ){
	cv::Mat R;

    	cv::Rodrigues( rvec, R );
    	Eigen::Matrix3d r;

    	for ( int i=0; i<3; i++ )
    		for ( int j=0; j<3; j++ ) 
            			r(i,j) = R.at<double>(i,j);
  
    	Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    	Eigen::AngleAxisd angle(r);
    	Eigen::Translation<double,3> trans(tvec.at<double>(0,0), tvec.at<double>(0,1), tvec.at<double>(0,2));
    	T = angle;
    	T(0,3) = tvec.at<double>(0,0); 
    	T(1,3) = tvec.at<double>(0,1); 
    	T(2,3) = tvec.at<double>(0,2);

    	return T;

}

// joinPointCloud 

PointCloud::Ptr joinPointCloud( PointCloud::Ptr original, FRAME& newFrame, Eigen::Isometry3d T, CAMERA_INTRINSIC_PARAMETERS& camera ) 
{
    PointCloud::Ptr newCloud = image2PointCloud( newFrame.rgb, newFrame.depth, camera );


    PointCloud::Ptr output (new PointCloud());
    pcl::transformPointCloud( *original, *output, T.matrix() );
    *newCloud += *output;


    static pcl::VoxelGrid<PointT> voxel;
    static ParameterReader pd;
    double gridsize = atof( pd.getData("voxel_grid").c_str() );
    voxel.setLeafSize( gridsize, gridsize, gridsize );
    voxel.setInputCloud( newCloud );
    PointCloud::Ptr tmp( new PointCloud() );
    voxel.filter( *tmp );
    return tmp;
}

cv::Mat mergeRows(cv::Mat A, cv::Mat B){
    int totalRows = A.rows + B.rows;

    cv::Mat mergedDescriptors(totalRows, A.cols, A.type());
    cv::Mat submat = mergedDescriptors.rowRange(0, A.rows);
    A.copyTo(submat);
    submat = mergedDescriptors.rowRange(A.rows, totalRows);
    B.copyTo(submat);
    return mergedDescriptors;
}

cv::Mat mergeCols(cv::Mat A, cv::Mat B){
    int totalCols = A.cols + B.cols;

    cv::Mat mergedDescriptors(totalCols, A.rows, A.type());
    cv::Mat submat = mergedDescriptors.colRange(0, A.cols);
    A.copyTo(submat);
    submat = mergedDescriptors.colRange(A.cols, totalCols);
    B.copyTo(submat);
    return mergedDescriptors;
}

pair<cv::Mat,vector<int>>  imgpreprocess (cv::Mat img){
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


