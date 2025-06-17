#pragma once

#ifdef USE_CUDA

#include "../../core/Hittable.cuh"
#include "ONB.cuh"
#include "PDFTypes.cuh"
#include "Utility.cuh"
#include "Vec3.cuh"
#include "Vec3Utility.cuh"
#include <curand_kernel.h>

// Abstract base class for probability density functions.
class PDF {
public:
  __device__ virtual ~PDF() = default;

  __device__ virtual double value(const Vec3 &direction) const = 0;
  __device__ virtual Vec3 generate(curandState *state) const = 0;
};

// Uniform sampling over the unit sphere.
class SpherePDF : public PDF {
public:
  __device__ SpherePDF() = default;

  __device__ double value(const Vec3 &direction) const override {
    return 1.0 / (4.0 * PI);
  }

  __device__ Vec3 generate(curandState *state) const override {
    return random_unit_vector(state);
  }
};

// Cosine-weighted hemisphere sampling PDF.
class CosinePDF : public PDF {
public:
  __device__ CosinePDF(const Vec3 &w) : m_uvw(w) {}

  __device__ double value(const Vec3 &direction) const override {
    double cosine_theta = dot_product(unit_vector(direction), m_uvw.w());
    return fmax(0.0, cosine_theta / PI);
  }

  __device__ Vec3 generate(curandState *state) const override {
    return m_uvw.transform(random_cosine_direction(state));
  }

private:
  ONB m_uvw;
};

// PDF for sampling based on a hittable object.
class HittablePDF : public PDF {
public:
  __device__ HittablePDF(const Hittable &objects, const Point3 &origin)
      : m_objects(objects), m_origin(origin) {}

  __device__ double value(const Vec3 &direction) const override {
    return m_objects.pdf_value(m_origin, direction);
  }

  __device__ Vec3 generate(curandState *state) const override {
    return m_objects.random(m_origin, state);
  }

private:
  const Hittable &m_objects;
  Point3 m_origin;
};

// Mixture of two PDFs.
class MixturePDF : public PDF {
public:
  __device__ MixturePDF(const PDF *p0, const PDF *p1) {
    m_p[0] = p0;
    m_p[1] = p1;
  }

  __device__ double value(const Vec3 &direction) const override {
    return 0.5 * m_p[0]->value(direction) + 0.5 * m_p[1]->value(direction);
  }

  __device__ Vec3 generate(curandState *state) const override {
    if (random_double(state) < 0.5)
      return m_p[0]->generate(state);
    return m_p[1]->generate(state);
  }

private:
  const PDF *m_p[2];
};

#endif // USE_CUDA
