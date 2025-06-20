#pragma once

#ifdef USE_CUDA

struct CudaVec3; // From Vec3.cuh

using CudaColor = CudaVec3;
using CudaPoint3 = CudaVec3;

#endif // USE_CUDA