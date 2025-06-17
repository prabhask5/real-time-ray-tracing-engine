#include "Utility.cuh"

#ifdef USE_CUDA

__device__ double degrees_to_radians(double degrees) {
    return degrees * PI / 180.0;
}

__device__ double random_double(curandState* state) {
    return curand_uniform_double(state);
}

__device__ double random_double(curandState* state, double min, double max) {
    return min + (max - min) * random_double(state);
}

__device__ int random_int(curandState* state, int min, int max) {
    return static_cast<int>(random_double(state, min, max + 1));
}

#endif // USE_CUDA