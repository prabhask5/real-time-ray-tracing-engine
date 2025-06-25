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

enum class CudaMaterialType {
  MATERIAL_LAMBERTIAN,
  MATERIAL_METAL,
  MATERIAL_DIELECTRIC,
  MATERIAL_DIFFUSE_LIGHT,
  MATERIAL_ISOTROPIC
};

// Models metallic (specular) surfaces like brushed aluminum or mirrors, where
// rays reflect off the surface with some optional fuzziness.
struct CudaMetalMaterial {
  CudaColor albedo;
  double fuzz;

  __device__ CudaMetalMaterial() {} // Default constructor.
  __host__ __device__ CudaMetalMaterial(CudaColor _albedo, double _fuzz)
      : albedo(_albedo), fuzz(_fuzz) {}

  __device__ bool scatter(const CudaRay &ray, const CudaHitRecord &rec,
                          CudaScatterRecord &srec, curandState *state) const {
    // Reflect the incoming ray around the surface normal.
    CudaVec3 reflected =
        cuda_reflect(cuda_unit_vector(ray.direction), rec.normal);

    // Add some random fuzz to the direction (if fuzz > 0) and normalize to keep
    // the ray unit-length.
    CudaVec3 direction = reflected + fuzz * cuda_random_unit_vector(state);

    // Set the attenuation to the albedo of the material, and set the skip pdf
    // ray to represent a mirror reflection instead of diffuse scattering.
    srec.attenuation = albedo;
    srec.skip_pdf_ray = CudaRay(rec.point, direction, ray.time);
    srec.pdf_pointer = nullptr;
    srec.skip_pdf = true;
    return true;
  }
};

// Defines a Lambertian (diffuse) material — a surface that scatters light
// uniformly in all directions from the point of intersection.
struct CudaLambertianMaterial {
  CudaTexture *texture;

  __device__ CudaLambertianMaterial() {} // Default constructor.
  __host__ __device__ CudaLambertianMaterial(CudaTexture *_texture)
      : texture(_texture) {}

  __device__ bool scatter(const CudaRay &ray, const CudaHitRecord &rec,
                          CudaScatterRecord &srec, curandState *state) const {
    // Produces diffuse, hemisphere-biased scattering, simulating real-world
    // matte surfaces by adding a random unit vector to the surface normal,
    // generating a new direction roughly biased around the surface normal.

    // Set the attenuation to the albedo of the material.
    srec.attenuation = texture->value(rec.u, rec.v, rec.point);

    // This assigns a cosine-weighted probability density function (PDF) for
    // generating outgoing rays. This is ideal for diffuse materials, where
    // light scatters in many directions but is stronger near the normal.
    srec.pdf_pointer = new CudaPDF(cuda_make_cosine_pdf(rec.normal));
    srec.skip_pdf = false;

    return true;
  }

  __device__ double scattering_pdf(const CudaRay &ray, const CudaHitRecord &rec,
                                   const CudaRay &scattered) const {
    double cos_theta =
        cuda_dot_product(rec.normal, cuda_unit_vector(scattered.direction));

    // PDF formula for Lambertian materials.
    return cos_theta < 0 ? 0 : cos_theta / CUDA_PI;
  }
};

// Models transparent materials (like glass or water) that can both reflect and
// refract rays.
struct CudaDielectricMaterial {
  double refraction_index;

  __device__ CudaDielectricMaterial() {} // Default constructor.
  __host__ __device__ CudaDielectricMaterial(double _refraction_index)
      : refraction_index(_refraction_index) {}

  __device__ bool scatter(const CudaRay &ray, const CudaHitRecord &rec,
                          CudaScatterRecord &srec, curandState *state) const {
    // No absorption or color change — fully transmits light.
    srec.attenuation = CudaColor(1.0, 1.0, 1.0);

    // Set the skip pdf ray to represent a mirror reflection instead of diffuse
    // scattering.
    srec.pdf_pointer = nullptr;
    srec.skip_pdf = true;

    // If hitting the front face -> divide (air -> glass).
    // If hitting from inside -> use material index (glass -> air).
    double ri = rec.front_face ? (1.0 / refraction_index) : refraction_index;

    // Use Snell's Law to determine whether total internal reflection occurs.
    // If cannot_refract == true, ray must reflect.
    CudaVec3 unit_dir = cuda_unit_vector(ray.direction);
    double cos_theta = fmin(cuda_dot_product(-unit_dir, rec.normal), 1.0);
    double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    // Use Schlick's approximation for reflectance.
    bool cannot_refract = ri * sin_theta > 1.0;
    double r0 = (1 - ri) / (1 + ri);
    r0 = r0 * r0;
    double reflect_prob = r0 + (1 - r0) * pow((1 - cos_theta), 5);

    CudaVec3 direction;

    // Use Schlick's approximation to probabilistically reflect or refract.
    if (cannot_refract || reflect_prob > cuda_random_double(state))
      direction = cuda_reflect(unit_dir, rec.normal);
    else
      direction = cuda_refract(unit_dir, rec.normal, ri);

    // Set the new scattered ray.
    srec.skip_pdf_ray = CudaRay(rec.point, direction, ray.time);

    return true;
  }
};

// Represents a pure emissive surface—a material that emits light rather than
// reflecting or scattering it. Think of glowing surfaces like light bulbs,
// lava, or emissive panels.
struct CudaDiffuseLightMaterial {
  CudaTexture *texture;

  __device__ CudaDiffuseLightMaterial() {} // Default constructor.
  __host__ __device__ CudaDiffuseLightMaterial(CudaTexture *_texture)
      : texture(_texture) {}

  __device__ CudaColor emitted(const CudaRay &ray, const CudaHitRecord &rec,
                               double u, double v, const CudaPoint3 &p) const {
    // Only emits from the front face of the surface (checked via
    // rec.front_face) to prevent backface lighting.
    if (!rec.front_face)
      return CudaColor(0, 0, 0);

    // Determines the color of the light emitted from the texture of the
    // material
    return texture->value(u, v, p);
  }
};

// Represents a scattering material where rays scatter equally in all
// directions—used for things like volumetric fog, smoke, or constant-density
// media.
struct CudaIsotropicMaterial {
  CudaTexture *texture;

  __device__ CudaIsotropicMaterial() {} // Default constructor.
  __host__ __device__ CudaIsotropicMaterial(CudaTexture *_texture)
      : texture(_texture) {}

  __device__ bool scatter(const CudaRay &ray, const CudaHitRecord &rec,
                          CudaScatterRecord &srec, curandState *state) const {
    srec.attenuation = texture->value(rec.u, rec.v, rec.point);

    // Uses a uniform sphere PDF (sphere_pdf) to scatter rays randomly in all
    // directions, regardless of normal or incident direction.
    srec.pdf_pointer = new CudaPDF(cuda_make_sphere_pdf());
    srec.skip_pdf = false;

    return true;
  }

  __device__ double scattering_pdf(const CudaRay &ray, const CudaHitRecord &rec,
                                   const CudaRay &scattered) const {
    // Uses a uniform sphere PDF (sphere_pdf) to scatter rays randomly in all
    // directions, regardless of normal or incident direction.
    return 1 / (4 * CUDA_PI);
  }
};

struct CudaMaterial {
  CudaMaterialType type;

  union {
    CudaMetalMaterial *metal;
    CudaLambertianMaterial *lambertian;
    CudaDielectricMaterial *dielectric;
    CudaDiffuseLightMaterial *diffuse;
    CudaIsotropicMaterial *isotropic;
  };

  __host__ __device__ CudaMaterial() {} // Default constructor.

  // Used to simulate the light producing surfaces of light emitting objects.
  // Default: Color(0, 0, 0), emits black when the object does not produce
  // light. Uses u, v as the texture coordinates to control how textures emit
  // different light at different points. Point is the hit point.
  __device__ CudaColor emitted(const CudaRay &ray, const CudaHitRecord &rec,
                               double u, double v, const CudaPoint3 &p) const {
    if (type == CudaMaterialType::MATERIAL_DIFFUSE_LIGHT)
      return diffuse->emitted(ray, rec, u, v, p);
    return CudaColor(0, 0, 0);
  }

  // This method defines how a ray scatters (bounces, refracts, reflects) when
  // it hits a surface made of this material. NOTE: attenuation is the output
  // color multiplier (e.g., surface absorbs some light). And scattered is the
  // output ray (reflected or refracted). Returns false if the ray is absorbed
  // (no output ray), true if it's scattered and we can continue calculation.
  __device__ bool scatter(const CudaRay &ray, const CudaHitRecord &rec,
                          CudaScatterRecord &srec, curandState *state) const {
    switch (type) {
    case CudaMaterialType::MATERIAL_METAL:
      return metal->scatter(ray, rec, srec, state);
    case CudaMaterialType::MATERIAL_LAMBERTIAN:
      return lambertian->scatter(ray, rec, srec, state);
    case CudaMaterialType::MATERIAL_DIELECTRIC:
      return dielectric->scatter(ray, rec, srec, state);
    case CudaMaterialType::MATERIAL_ISOTROPIC:
      return isotropic->scatter(ray, rec, srec, state);
    default:
      // Note: This should never happen in well-formed code, but we throw an
      // error for debugging
      return false; // Keep original behavior as this may be called from device
                    // code.
    }
  }

  // Returns the probability density of scattering a ray from hit ray to
  // scattered at the given point. Needed when using importance sampling for
  // physically-based rendering.
  __device__ double scattering_pdf(const CudaRay &ray, const CudaHitRecord &rec,
                                   const CudaRay &scattered) const {
    switch (type) {
    case CudaMaterialType::MATERIAL_LAMBERTIAN:
      return lambertian->scattering_pdf(ray, rec, scattered);
    case CudaMaterialType::MATERIAL_ISOTROPIC:
      return isotropic->scattering_pdf(ray, rec, scattered);
    default:
      // Note: This should never happen in well-formed code, but we return 0.0
      // for safety in device code
      return 0.0; // Keep original behavior as this may be called from device
                  // code.
    }
  }
};

// Helper constructor functions.

__device__ inline CudaMaterial
cuda_make_lambertian_material(CudaTexture *texture) {
  CudaMaterial material;
  material.type = CudaMaterialType::MATERIAL_LAMBERTIAN;
  material.lambertian = new CudaLambertianMaterial(texture);
  return material;
}

__device__ inline CudaMaterial cuda_make_metal_material(CudaColor albedo,
                                                        double fuzz) {
  CudaMaterial material;
  material.type = CudaMaterialType::MATERIAL_METAL;
  material.metal = new CudaMetalMaterial(albedo, fuzz);
  return material;
}

__device__ inline CudaMaterial
cuda_make_dielectric_material(double refraction_index) {
  CudaMaterial material;
  material.type = CudaMaterialType::MATERIAL_DIELECTRIC;
  material.dielectric = new CudaDielectricMaterial(refraction_index);
  return material;
}

__device__ inline CudaMaterial
cuda_make_diffuse_light_material(CudaTexture *texture) {
  CudaMaterial material;
  material.type = CudaMaterialType::MATERIAL_DIFFUSE_LIGHT;
  material.diffuse = new CudaDiffuseLightMaterial(texture);
  return material;
}

__device__ inline CudaMaterial
cuda_make_isotropic_material(CudaTexture *texture) {
  CudaMaterial material;
  material.type = CudaMaterialType::MATERIAL_ISOTROPIC;
  material.isotropic = new CudaIsotropicMaterial(texture);
  return material;
}

#endif // USE_CUDA
