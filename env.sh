# EXTERNAL_ROOT=$(pwd)/cmake-build-Release/external/install/Release

# # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${EXTERNAL_ROOT}/Ceres/lib:${EXTERNAL_ROOT}/glog/lib
# export LD_LIBRARY_PATH=${EXTERNAL_ROOT}/Ceres/lib:${EXTERNAL_ROOT}/glog/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=cmake-build-Release/external/install/Release/Ceres/lib:cmake-build-Release/external/install/Release/glog/lib:$LD_LIBRARY_PATH
