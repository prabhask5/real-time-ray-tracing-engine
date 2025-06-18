#pragma once

#ifdef USE_CUDA

#include "../../core/HitRecord.cuh"
#include "../../core/Ray.cuh"
#include "../../core/ScatterRecord.cuh"
#include "../../core/Vec3Types.hpp"
#include "../../utils/math/Interval.cuh"
#include "../../utils/math/PDF.cuh"
#include "../../utils/math/Random.cuh"
#include "../../utils/math/Vec3.hpp"
#include "../../utils/math/Vec3Utility.cuh"
#include "../textures/Texture.cuh"

enum CudaMaterialType {
  MATERIAL_LAMBERTIAN,
  MATERIAL_METAL,
  MATERIAL_DIELECTRIC,
  MATERIAL_DIFFUSE_LIGHT,
  MATERIAL_ISOTROPIC
};

struct CudaMetalMaterial {
  CudaColor albedo;
  double fuzz;

  __device__ bool scatter(const CudaRay &ray, const CudaHitRecord &rec,
                          CudaScatterRecord &srec) const {
    CudaVec3 reflected =
        cuda_reflect(cuda_unit_vector(ray.direction), rec.normal);
    CudaVec3 direction = reflected + fuzz * cuda_random_unit_vector();
    srec.attenuation = albedo;
    srec.skip_pdf_ray = CudaRay(rec.point, direction, ray.time());
    srec.pdf_ptr = nullptr;
    srec.skip_pdf = true;
    return true;
  }
};

struct CudaLambertianMaterial {
  CudaTexture texture;

  __device__ bool scatter(const CudaRay &ray, const CudaHitRecord &rec,
                          CudaScatterRecord &srec) const {
    srec.attenuation = texture.value(rec.u, rec.v, rec.point);
    srec.pdf_ptr = new CudaCosinePDF(rec.normal);
    srec.skip_pdf = false;
    return true;
  }

  __device__ double scattering_pdf(const CudaRay &ray, const CudaHitRecord &rec,
                                   const CudaRay &scattered) const {
    double cos_theta =
        cuda_dot_product(rec.normal, cuda_unit_vector(scattered.direction));
    return cos_theta < 0 ? 0 : cos_theta / CUDA_PI;
  }
};

struct CudaDielectricMaterial {
  double refraction_index;

  __device__ bool scatter(const CudaRay &ray, const CudaHitRecord &rec,
                          CudaScatterRecord &srec) const {
    srec.attenuation = CudaColor(1.0, 1.0, 1.0);
    srec.pdf_ptr = nullptr;
    srec.skip_pdf = true;

    double ri = rec.frontFace ? (1.0 / refraction_index) : refraction_index;
    Vec3 unit_dir = cuda_unit_vector(ray.direction);
    double cos_theta = fmin(cuda_dot_product(-unit_dir, rec.normal), 1.0);
    double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    bool cannot_refract = ri * sin_theta > 1.0;
    double r0 = (1 - ri) / (1 + ri);
    r0 = r0 * r0;
    double reflect_prob = r0 + (1 - r0) * pow((1 - cos_theta), 5);

    CudaVec3 direction;
    if (cannot_refract || reflect_prob > cuda_random_double())
      direction = cuda_reflect(unit_dir, rec.normal);
    else
      direction = cuda_refract(unit_dir, rec.normal, ri);

    srec.skip_pdf_ray = CudaRay(rec.point, direction);
    return true;
  }
};

struct CudaDiffuseLightMaterial {
  CudaTexture texture;

  __device__ Color emitted(const CudaRay &ray, const CudaHitRecord &rec,
                           double u, double v, const CudaPoint3 &p) const {
    if (!rec.frontFace)
      return CudaColor(0, 0, 0);
    return texture.value(u, v, p);
  }
};

struct CudaIsotropicMaterial {
  CudaTexture texture;

  __device__ bool scatter(const CudaRay &ray, const CudaHitRecord &rec,
                          CudaScatterRecord &srec) const {
    srec.attenuation = texture.value(rec.u, rec.v, rec.point);
    srec.pdf_ptr = new CudaSpherePDF();
    srec.skip_pdf = false;
    return true;
  }

  __device__ double scattering_pdf(const CudaRay &ray, const CudaHitRecord &rec,
                                   const CudaRay &scattered) const {
    return 1 / (4 * CUDA_PI);
  }
};

struct CudaMaterial {
  CudaMaterialType type;

  union {
    CudaMetalMaterial metal;
    CudaLambertianMaterial lambertian;
    CudaDielectricMaterial dielectric;
    CudaDiffuseLightMaterial diffuse;
    CudaIsotropicMaterial isotropic;
  };

  __device__ Color emitted(const CudaRay &ray, const CudaHitRecord &rec,
                           double u, double v, const CudaPoint3 &p) const {
    if (type == MATERIAL_DIFFUSE_LIGHT)
      return diffuse.emitted(ray, rec, u, v, p);
    return CudaColor(0, 0, 0);
  }

  __device__ bool scatter(const CudaRay &ray, const CudaHitRecord &rec,
                          CudaScatterRecord &srec) const {
    switch (type) {
    case MATERIAL_METAL:
      return metal.scatter(ray, rec, srec);
    case MATERIAL_LAMBERTIAN:
      return lambertian.scatter(ray, rec, srec);
    case MATERIAL_DIELECTRIC:
      return dielectric.scatter(ray, rec, srec);
    case MATERIAL_ISOTROPIC:
      return isotropic.scatter(ray, rec, srec);
    default:
      return false;
    }
  }

  __device__ double scattering_pdf(const CudaRay &ray, const CudaHitRecord &rec,
                                   const CudaRay &scattered) const {
    switch (type) {
    case MATERIAL_LAMBERTIAN:
      return lambertian.scattering_pdf(ray, rec, scattered);
    case MATERIAL_ISOTROPIC:
      return isotropic.scattering_pdf(ray, rec, scattered);
    default:
      return 0.0;
    }
  }
};

#endif // USE_CUDA
