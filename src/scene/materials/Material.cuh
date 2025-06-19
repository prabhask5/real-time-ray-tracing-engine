#pragma once

#ifdef USE_CUDA

#include "../../core/HitRecord.cuh"
#include "../../core/Ray.cuh"
#include "../../core/ScatterRecord.cuh"
#include "../../core/Vec3Types.cuh"
#include "../../utils/math/Interval.cuh"
#include "../../utils/math/PDF.cuh"
#include "../../utils/math/Utility.cuh"
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

  __device__ CudaMetalMaterial(CudaColor _albedo, double _fuzz)
      : albedo(_albedo), fuzz(_fuzz) {}

  __device__ bool scatter(const CudaRay &ray, const CudaHitRecord &rec,
                          CudaScatterRecord &srec, curandState *state) const {
    CudaVec3 reflected =
        cuda_reflect(cuda_unit_vector(ray.direction), rec.normal);
    CudaVec3 direction = reflected + fuzz * cuda_random_unit_vector(state);
    srec.attenuation = albedo;
    srec.skip_pdf_ray = CudaRay(rec.point, direction, ray.time);
    srec.pdf_data = nullptr;
    srec.skip_pdf = true;
    return true;
  }
};

struct CudaLambertianMaterial {
  CudaTexture texture;

  __device__ CudaLambertianMaterial(CudaTexture _texture) : texture(_texture) {}

  __device__ bool scatter(const CudaRay &ray, const CudaHitRecord &rec,
                          CudaScatterRecord &srec, curandState *state) const {
    srec.attenuation = texture.value(rec.u, rec.v, rec.point);
    srec.pdf_type = CudaMaterialType::CUDA_PDF_COSINE;
    srec.pdf_data = nullptr; // Cosine PDF is handled by type
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

  __device__ CudaDielectricMaterial(double _refraction_index)
      : refraction_index(_refraction_index) {}

  __device__ bool scatter(const CudaRay &ray, const CudaHitRecord &rec,
                          CudaScatterRecord &srec, curandState *state) const {
    srec.attenuation = CudaColor(1.0, 1.0, 1.0);
    srec.pdf_data = nullptr;
    srec.skip_pdf = true;

    double ri = rec.front_face ? (1.0 / refraction_index) : refraction_index;
    CudaVec3 unit_dir = cuda_unit_vector(ray.direction);
    double cos_theta = fmin(cuda_dot_product(-unit_dir, rec.normal), 1.0);
    double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    bool cannot_refract = ri * sin_theta > 1.0;
    double r0 = (1 - ri) / (1 + ri);
    r0 = r0 * r0;
    double reflect_prob = r0 + (1 - r0) * pow((1 - cos_theta), 5);

    CudaVec3 direction;
    if (cannot_refract || reflect_prob > cuda_random_double(state))
      direction = cuda_reflect(unit_dir, rec.normal);
    else
      direction = cuda_refract(unit_dir, rec.normal, ri);

    srec.skip_pdf_ray = CudaRay(rec.point, direction, ray.time);
    return true;
  }
};

struct CudaDiffuseLightMaterial {
  CudaTexture texture;

  __device__ CudaDiffuseLightMaterial(CudaTexture _texture)
      : texture(_texture) {}

  __device__ CudaColor emitted(const CudaRay &ray, const CudaHitRecord &rec,
                               double u, double v, const CudaPoint3 &p) const {
    if (!rec.front_face)
      return CudaColor(0, 0, 0);
    return texture.value(u, v, p);
  }
};

struct CudaIsotropicMaterial {
  CudaTexture texture;

  __device__ CudaIsotropicMaterial(CudaTexture _texture) : texture(_texture) {}

  __device__ bool scatter(const CudaRay &ray, const CudaHitRecord &rec,
                          CudaScatterRecord &srec, curandState *state) const {
    srec.attenuation = texture.value(rec.u, rec.v, rec.point);
    srec.pdf_type = CudaMaterialType::CUDA_PDF_SPHERE;
    srec.pdf_data = nullptr; // Sphere PDF is handled by type
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
  } data;

  __device__ CudaColor emitted(const CudaRay &ray, const CudaHitRecord &rec,
                               double u, double v, const CudaPoint3 &p) const {
    if (type == CudaMaterialType::MATERIAL_DIFFUSE_LIGHT)
      return data.diffuse.emitted(ray, rec, u, v, p);
    return CudaColor(0, 0, 0);
  }

  __device__ bool scatter(const CudaRay &ray, const CudaHitRecord &rec,
                          CudaScatterRecord &srec, curandState *state) const {
    switch (type) {
    case CudaMaterialType::MATERIAL_METAL:
      return data.metal.scatter(ray, rec, srec, state);
    case CudaMaterialType::MATERIAL_LAMBERTIAN:
      return data.lambertian.scatter(ray, rec, srec, state);
    case CudaMaterialType::MATERIAL_DIELECTRIC:
      return data.dielectric.scatter(ray, rec, srec, state);
    case CudaMaterialType::MATERIAL_ISOTROPIC:
      return data.isotropic.scatter(ray, rec, srec, state);
    default:
      return false;
    }
  }

  __device__ double scattering_pdf(const CudaRay &ray, const CudaHitRecord &rec,
                                   const CudaRay &scattered) const {
    switch (type) {
    case CudaMaterialType::MATERIAL_LAMBERTIAN:
      return data.lambertian.scattering_pdf(ray, rec, scattered);
    case CudaMaterialType::MATERIAL_ISOTROPIC:
      return data.isotropic.scattering_pdf(ray, rec, scattered);
    default:
      return 0.0;
    }
  }
};

// Helper constructor functions.

__host__ __device__ inline CudaMaterial
cuda_make_lambertian_material(CudaTexture texture) {
  CudaMaterial material;
  material.type = CudaMaterialType::MATERIAL_LAMBERTIAN;
  material.data.lambertian = CudaLambertianMaterial(texture);
  return material;
}

__host__ __device__ inline CudaMaterial
cuda_make_metal_material(CudaColor albedo, double fuzz) {
  CudaMaterial material;
  material.type = CudaMaterialType::MATERIAL_METAL;
  material.data.metal = CudaMetalMaterial(albedo, fuzz);
  return material;
}

__host__ __device__ inline CudaMaterial
cuda_make_dielectric_material(double refraction_index) {
  CudaMaterial material;
  material.type = CudaMaterialType::MATERIAL_DIELECTRIC;
  material.data.dielectric = CudaDielectricMaterial(refraction_index);
  return material;
}

__host__ __device__ inline CudaMaterial
cuda_make_diffuse_light_material(CudaTexture texture) {
  CudaMaterial material;
  material.type = CudaMaterialType::MATERIAL_DIFFUSE_LIGHT;
  material.data.diffuse = CudaDiffuseLightMaterial(texture);
  return material;
}

__host__ __device__ inline CudaMaterial
cuda_make_isotropic_material(CudaTexture texture) {
  CudaMaterial material;
  material.type = CudaMaterialType::MATERIAL_ISOTROPIC;
  material.data.isotropic = CudaIsotropicMaterial(texture);
  return material;
}

#endif // USE_CUDA
