# - Try to find MatIO
#
# The following variables are optionally searched for defaults
#  MATIO_ROOT_DIR:            Base directory where all MatIO components are found
#
# The following are set after configuration is done:
#  MATIO_FOUND
#  MATIO_INCLUDE_DIR
#  MATIO_LIBRARIES

include(FindPackageHandleStandardArgs)

set(MATIO_ROOT_DIR "" CACHE PATH "Folder contains MatIO")

find_path(MATIO_INCLUDE_DIR matio.h
    PATHS ${MATIO_ROOT_DIR})

find_library(MATIO_LIBRARY matio
    PATHS ${MATIO_ROOT_DIR}
    PATH_SUFFIXES
        lib)

find_package_handle_standard_args(MATIO DEFAULT_MSG
    MATIO_INCLUDE_DIR MATIO_LIBRARY)

MESSAGE(STATUS "MATIO include:" ${MATIO_INCLUDE_DIR})
MESSAGE(STATUS "MATIO lib:" ${MATIO_LIBRARY})

if(MATIO_FOUND)
    set(MATIO_INCLUDE_DIR ${MATIO_INCLUDE_DIR})
    set(MATIO_LIBRARIES ${MATIO_LIBRARY})
endif()
