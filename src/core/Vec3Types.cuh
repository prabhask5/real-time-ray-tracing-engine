#pragma once

#ifdef USE_CUDA

class CudaVec3; // From CudaVec3.hpp.

using CudaColor = CudaVec3;
using CudaPoint3 = CudaVec3;

#endif // USE_CUDA