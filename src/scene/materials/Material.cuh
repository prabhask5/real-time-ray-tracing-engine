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
#include "../../utils/memory/CudaSceneContext.cuh"
#include "../textures/Texture.cuh"
#include <iomanip>
#include <sstream>

enum class CudaMaterialType {
  MATERIAL_LAMBERTIAN,
  MATERIAL_METAL,
  MATERIAL_DIELECTRIC,
  MATERIAL_DIFFUSE_LIGHT,
  MATERIAL_ISOTROPIC
};

// POD struct for metallic (specular) surfaces like brushed aluminum or mirrors.
struct CudaMetalMaterial {
  CudaColor albedo;
  double fuzz;
};

// Metal material initialization function.
__device__ inline CudaMetalMaterial cuda_make_metal_material(CudaColor albedo,
                                                             double fuzz) {
  CudaMetalMaterial material;
  material.albedo = albedo;
  material.fuzz = fuzz;
  return material;
}

// Metal material scatter function.
__device__ inline bool
cuda_metal_material_scatter(const CudaMetalMaterial &material,
                            const CudaRay &ray, const CudaHitRecord &rec,
                            CudaScatterRecord &srec, curandState *state) {
  CudaVec3 reflected =
      cuda_vec3_reflect(cuda_vec3_unit_vector(ray.direction), rec.normal);
  CudaVec3 direction = cuda_vec3_add(
      reflected, cuda_vec3_multiply_scalar(cuda_vec3_random_unit_vector(state),
                                           material.fuzz));

  srec.attenuation = material.albedo;
  srec.skip_pdf_ray = cuda_make_ray(rec.point, direction, ray.time);
  srec.skip_pdf = true;
  return true;
}

// POD struct for Lambertian (diffuse) material.
struct CudaLambertianMaterial {
  size_t texture_index;
};

// Lambertian material initialization function.
__device__ inline CudaLambertianMaterial
cuda_make_lambertian_material(size_t texture_index) {
  CudaLambertianMaterial material;
  material.texture_index = texture_index;
  return material;
}

// Lambertian material scatter function.
__device__ inline bool
cuda_lambertian_material_scatter(const CudaLambertianMaterial &material,
                                 const CudaRay &ray, const CudaHitRecord &rec,
                                 CudaScatterRecord &srec, curandState *state) {
  extern __device__ CudaSceneContextView *d_scene_context;
  const CudaTexture &texture =
      d_scene_context->textures[material.texture_index];
  srec.attenuation = cuda_texture_value(texture, rec.u, rec.v, rec.point);

  // Create cosine PDF for this scatter.
  srec.pdf = cuda_make_pdf_cosine(rec.normal);
  srec.skip_pdf = false;

  return true;
}

// Lambertian material scattering PDF function.
__device__ inline double cuda_lambertian_material_scattering_pdf(
    const CudaLambertianMaterial &material, const CudaRay &ray,
    const CudaHitRecord &rec, const CudaRay &scattered) {
  double cos_theta = cuda_vec3_dot_product(
      rec.normal, cuda_vec3_unit_vector(scattered.direction));
  return cos_theta < 0 ? 0 : cos_theta / CUDA_PI;
}

// POD struct for transparent materials (like glass or water).
struct CudaDielectricMaterial {
  double refraction_index;
};

// Dielectric material initialization function.
__device__ inline CudaDielectricMaterial
cuda_make_dielectric_material(double refraction_index) {
  CudaDielectricMaterial material;
  material.refraction_index = refraction_index;
  return material;
}

// Dielectric material scatter function.
__device__ inline bool
cuda_dielectric_material_scatter(const CudaDielectricMaterial &material,
                                 const CudaRay &ray, const CudaHitRecord &rec,
                                 CudaScatterRecord &srec, curandState *state) {
  srec.attenuation = cuda_make_vec3(1.0, 1.0, 1.0);
  srec.skip_pdf = true;

  double ri = rec.front_face ? (1.0 / material.refraction_index)
                             : material.refraction_index;

  CudaVec3 unit_dir = cuda_vec3_unit_vector(ray.direction);
  double cos_theta =
      fmin(cuda_vec3_dot_product(cuda_vec3_negate(unit_dir), rec.normal), 1.0);
  double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

  bool cannot_refract = ri * sin_theta > 1.0;
  double r0 = (1 - ri) / (1 + ri);
  r0 = r0 * r0;
  double reflect_prob = r0 + (1 - r0) * pow((1 - cos_theta), 5);

  CudaVec3 direction;
  if (cannot_refract || reflect_prob > cuda_random_double(state))
    direction = cuda_vec3_reflect(unit_dir, rec.normal);
  else
    direction = cuda_vec3_refract(unit_dir, rec.normal, ri);

  srec.skip_pdf_ray = cuda_make_ray(rec.point, direction, ray.time);

  return true;
}

// POD struct for pure emissive surface.
struct CudaDiffuseLightMaterial {
  size_t texture_index;
};

// Diffuse light material initialization function.
__device__ inline CudaDiffuseLightMaterial
cuda_make_diffuse_light_material(size_t texture_index) {
  CudaDiffuseLightMaterial material;
  material.texture_index = texture_index;
  return material;
}

// Diffuse light material emitted function.
__device__ inline CudaColor cuda_diffuse_light_material_emitted(
    const CudaDiffuseLightMaterial &material, const CudaRay &ray,
    const CudaHitRecord &rec, double u, double v, const CudaPoint3 &p) {
  if (!rec.front_face)
    return cuda_make_vec3(0.0, 0.0, 0.0);

  extern __device__ CudaSceneContextView *d_scene_context;
  const CudaTexture &texture =
      d_scene_context->textures[material.texture_index];
  return cuda_texture_value(texture, u, v, p);
}

// POD struct for scattering material (fog, smoke, volumetric media).
struct CudaIsotropicMaterial {
  size_t texture_index;
};

// Isotropic material initialization function.
__device__ inline CudaIsotropicMaterial
cuda_make_isotropic_material(size_t texture_index) {
  CudaIsotropicMaterial material;
  material.texture_index = texture_index;
  return material;
}

// Isotropic material scatter function.
__device__ inline bool
cuda_isotropic_material_scatter(const CudaIsotropicMaterial &material,
                                const CudaRay &ray, const CudaHitRecord &rec,
                                CudaScatterRecord &srec, curandState *state) {
  extern __device__ CudaSceneContextView *d_scene_context;
  const CudaTexture &texture =
      d_scene_context->textures[material.texture_index];
  srec.attenuation = cuda_texture_value(texture, rec.u, rec.v, rec.point);

  // Create sphere PDF for isotropic scattering.
  srec.pdf = cuda_make_pdf_sphere();
  srec.skip_pdf = false;

  return true;
}

// Isotropic material scattering PDF function.
__device__ inline double cuda_isotropic_material_scattering_pdf(
    const CudaIsotropicMaterial &material, const CudaRay &ray,
    const CudaHitRecord &rec, const CudaRay &scattered) {
  return 1.0 / (4.0 * CUDA_PI);
}

// POD struct for unified material using manual dispatch pattern.
struct CudaMaterial {
  CudaMaterialType type;

  union {
    CudaMetalMaterial metal;
    CudaLambertianMaterial lambertian;
    CudaDielectricMaterial dielectric;
    CudaDiffuseLightMaterial diffuse;
    CudaIsotropicMaterial isotropic;
  };
};

// Material emitted function.
__device__ inline CudaColor cuda_material_emitted(const CudaMaterial &material,
                                                  const CudaRay &ray,
                                                  const CudaHitRecord &rec,
                                                  double u, double v,
                                                  const CudaPoint3 &p) {
  if (material.type == CudaMaterialType::MATERIAL_DIFFUSE_LIGHT)
    return cuda_diffuse_light_material_emitted(material.diffuse, ray, rec, u, v,
                                               p);
  return cuda_make_vec3(0.0, 0.0, 0.0);
}

// Material scatter function.
__device__ inline bool cuda_material_scatter(const CudaMaterial &material,
                                             const CudaRay &ray,
                                             const CudaHitRecord &rec,
                                             CudaScatterRecord &srec,
                                             curandState *state) {
  switch (material.type) {
  case CudaMaterialType::MATERIAL_METAL:
    return cuda_metal_material_scatter(material.metal, ray, rec, srec, state);
  case CudaMaterialType::MATERIAL_LAMBERTIAN:
    return cuda_lambertian_material_scatter(material.lambertian, ray, rec, srec,
                                            state);
  case CudaMaterialType::MATERIAL_DIELECTRIC:
    return cuda_dielectric_material_scatter(material.dielectric, ray, rec, srec,
                                            state);
  case CudaMaterialType::MATERIAL_ISOTROPIC:
    return cuda_isotropic_material_scatter(material.isotropic, ray, rec, srec,
                                           state);
  default:
    return false;
  }
}

// Material scattering PDF function.
__device__ inline double
cuda_material_scattering_pdf(const CudaMaterial &material, const CudaRay &ray,
                             const CudaHitRecord &rec,
                             const CudaRay &scattered) {
  switch (material.type) {
  case CudaMaterialType::MATERIAL_LAMBERTIAN:
    return cuda_lambertian_material_scattering_pdf(material.lambertian, ray,
                                                   rec, scattered);
  case CudaMaterialType::MATERIAL_ISOTROPIC:
    return cuda_isotropic_material_scattering_pdf(material.isotropic, ray, rec,
                                                  scattered);
  default:
    return 0.0;
  }
}

// Unified material constructor functions.
__device__ inline CudaMaterial
cuda_make_material_lambertian(size_t texture_index) {
  CudaMaterial material;
  material.type = CudaMaterialType::MATERIAL_LAMBERTIAN;
  material.lambertian = cuda_make_lambertian_material(texture_index);
  return material;
}

__device__ inline CudaMaterial cuda_make_material_metal(CudaColor albedo,
                                                        double fuzz) {
  CudaMaterial material;
  material.type = CudaMaterialType::MATERIAL_METAL;
  material.metal = cuda_make_metal_material(albedo, fuzz);
  return material;
}

__device__ inline CudaMaterial
cuda_make_material_dielectric(double refraction_index) {
  CudaMaterial material;
  material.type = CudaMaterialType::MATERIAL_DIELECTRIC;
  material.dielectric = cuda_make_dielectric_material(refraction_index);
  return material;
}

__device__ inline CudaMaterial
cuda_make_material_diffuse_light(size_t texture_index) {
  CudaMaterial material;
  material.type = CudaMaterialType::MATERIAL_DIFFUSE_LIGHT;
  material.diffuse = cuda_make_diffuse_light_material(texture_index);
  return material;
}

__device__ inline CudaMaterial
cuda_make_material_isotropic(size_t texture_index) {
  CudaMaterial material;
  material.type = CudaMaterialType::MATERIAL_ISOTROPIC;
  material.isotropic = cuda_make_isotropic_material(texture_index);
  return material;
}

// JSON serialization functions for CUDA materials.
inline std::string cuda_json_metal_material(const CudaMetalMaterial &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaMetalMaterial\",";
  oss << "\"address\":\"" << &obj << "\",";
  oss << "\"albedo\":" << cuda_json_vec3(obj.albedo) << ",";
  oss << "\"fuzz\":" << obj.fuzz;
  oss << "}";
  return oss.str();
}

inline std::string
cuda_json_lambertian_material(const CudaLambertianMaterial &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaLambertianMaterial\",";
  oss << "\"address\":\"" << &obj << "\",";
  oss << "\"texture_index\":" << obj.texture_index;
  oss << "}";
  return oss.str();
}

inline std::string
cuda_json_dielectric_material(const CudaDielectricMaterial &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaDielectricMaterial\",";
  oss << "\"address\":\"" << &obj << "\",";
  oss << "\"refraction_index\":" << obj.refraction_index;
  oss << "}";
  return oss.str();
}

inline std::string
cuda_json_diffuse_light_material(const CudaDiffuseLightMaterial &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaDiffuseLightMaterial\",";
  oss << "\"address\":\"" << &obj << "\",";
  oss << "\"texture_index\":" << obj.texture_index;
  oss << "}";
  return oss.str();
}

inline std::string
cuda_json_isotropic_material(const CudaIsotropicMaterial &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaIsotropicMaterial\",";
  oss << "\"address\":\"" << &obj << "\",";
  oss << "\"texture_index\":" << obj.texture_index;
  oss << "}";
  return oss.str();
}

inline std::string cuda_json_material(const CudaMaterial &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaMaterial\",";
  oss << "\"address\":\"" << &obj << "\",";
  oss << "\"material_type\":";
  switch (obj.type) {
  case CudaMaterialType::MATERIAL_LAMBERTIAN:
    oss << "\"LAMBERTIAN\",";
    oss << "\"lambertian\":" << cuda_json_lambertian_material(obj.lambertian);
    break;
  case CudaMaterialType::MATERIAL_METAL:
    oss << "\"METAL\",";
    oss << "\"metal\":" << cuda_json_metal_material(obj.metal);
    break;
  case CudaMaterialType::MATERIAL_DIELECTRIC:
    oss << "\"DIELECTRIC\",";
    oss << "\"dielectric\":" << cuda_json_dielectric_material(obj.dielectric);
    break;
  case CudaMaterialType::MATERIAL_DIFFUSE_LIGHT:
    oss << "\"DIFFUSE_LIGHT\",";
    oss << "\"diffuse\":" << cuda_json_diffuse_light_material(obj.diffuse);
    break;
  case CudaMaterialType::MATERIAL_ISOTROPIC:
    oss << "\"ISOTROPIC\",";
    oss << "\"isotropic\":" << cuda_json_isotropic_material(obj.isotropic);
    break;
  }
  oss << "}";
  return oss.str();
}

#endif // USE_CUDA