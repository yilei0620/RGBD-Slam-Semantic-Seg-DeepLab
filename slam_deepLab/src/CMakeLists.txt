# Add PCL Lib		
FIND_PACKAGE( PCL REQUIRED COMPONENTS common io visualization filters)

# Add OpenCV
FIND_PACKAGE( OpenCV REQUIRED )

# Add heads and libs
ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )

LIST( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
SET( G2O_ROOT /usr/local/include/g2o )
FIND_PACKAGE( G2O REQUIRED )

# CSparse
FIND_PACKAGE( CSparse REQUIRED )
INCLUDE_DIRECTORIES( ${G2O_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR} )

FIND_PACKAGE(Caffe REQUIRED)
INCLUDE_DIRECTORIES(${Caffe_INCLUDE_DIRS})
ADD_DEFINITIONS(${Caffe_DEFINITIONS})

ADD_LIBRARY( slambase slamBase.cpp )
TARGET_LINK_LIBRARIES( slambase
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )

ADD_EXECUTABLE( slam slam.cpp )
TARGET_LINK_LIBRARIES( slam
    slambase
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension ${CSPARSE_LIBRARY})

ADD_EXECUTABLE( slamDP slamDP.cpp )
TARGET_LINK_LIBRARIES( slamDP ${Caffe_LIBRARIES}
    ${OpenCV_LIBS}  slambase g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension ${CSPARSE_LIBRARY})


