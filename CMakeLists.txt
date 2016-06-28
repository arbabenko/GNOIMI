project (NearestSearch CXX C)
cmake_minimum_required(VERSION 2.6)

find_package(LAPACK)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -std=c++0x -m64")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -std=c++0x -m64")

SET(YAEL_DIR /home/arbabenko/libs/yael_v438)
SET(YAEL_LIB /home/arbabenko/libs/yael_v438/yael)

SET(BLAS_DIR /opt/OpenBLAS/include)
SET(BLAS_LIB /opt/OpenBLAS/lib)

#########################################################################################
INCLUDE_DIRECTORIES(${BLAS_DIR})
INCLUDE_DIRECTORIES(${YAEL_DIR})
LINK_DIRECTORIES(${YAEL_LIB})
LINK_DIRECTORIES(${BLAS_LIB})
ADD_EXECUTABLE (learnGNOIMI learn_GNOIMI.cpp)
TARGET_LINK_LIBRARIES (learnGNOIMI yael ${LAPACK_LIBRARIES})

#########################################################################################
INCLUDE_DIRECTORIES(${BLAS_DIR})
INCLUDE_DIRECTORIES(${YAEL_DIR})
LINK_DIRECTORIES(${YAEL_LIB})
LINK_DIRECTORIES(${BLAS_LIB})
ADD_EXECUTABLE (searchGNOIMI search_GNOIMI.cpp)
TARGET_LINK_LIBRARIES (searchGNOIMI yael ${LAPACK_LIBRARIES})
