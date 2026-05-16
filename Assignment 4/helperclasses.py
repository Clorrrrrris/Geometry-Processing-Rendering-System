# Cloris Liu 261010926
import taichi as ti
import taichi.math as tm

@ti.dataclass
class Ray:
    origin: tm.vec3
    direction: tm.vec3
    time: float

@ti.func
def getRayDistance(ray: Ray, point: tm.vec3) -> float:
    return tm.length(point - ray.origin)

@ti.func
def getRayPoint(ray: Ray, t: float) -> tm.vec3:
    return ray.origin + ray.direction * t

@ti.func
def changeRayFrame(ray: Ray, M: tm.mat4) -> Ray:
    # TODO: Objective 4: Ray and Geometry Transformations

    origin_h = tm.vec4(ray.origin, 1.0)
    dir_h    = tm.vec4(ray.direction, 0.0)
    new_origin = (M @ origin_h).xyz
    new_dir    = (M @ dir_h).xyz
    return Ray(origin=new_origin, direction=new_dir, time=ray.time)

    #return ray # change this placeholder to return a transformed ray

@ti.dataclass
class Material:
        id: int
        diffuse: tm.vec3    # kd diffuse coefficient
        specular: tm.vec3    # ks specular coefficient
        shininess: tm.vec3    # specular exponent
        reflectivity: tm.vec3 # mirror reflection weight (0 = none)
        ior: float          # index of refraction (for Fresnel), default 1.5
        transmission: tm.vec3 # refraction weight (0 = opaque)
        has_tex: ti.i32      # 1 if texture bound
        tex_id: ti.i32       # index into texture array
        tex_w: ti.i32
        tex_h: ti.i32

@ti.dataclass
class Light:
        ltype: int       # type is either 0 for "directional" or 1 for "point"
        id: int
        colour: tm.vec3   # colour and intensity of the light
        vector: tm.vec3    # position, or normalized direction towards light, depending on the light type
        attenuation: tm.vec3   # attenuation coeffs [quadratic, linear, constant] for point lights
        u_dir: tm.vec3    # half-width vector for area light (0 otherwise)
        v_dir: tm.vec3    # half-height vector for area light (0 otherwise)
        normal: tm.vec3   # area light surface normal (0 otherwise)
        samples: ti.i32   # area light sample count (1 otherwise)

@ti.dataclass 
class Intersection:
        # All fields will be set to zero on creation, otherwise specified in this order on construction
        is_hit: bool
        t: float
        normal: tm.vec3
        position: tm.vec3
        mat: Material
        uv: tm.vec2

@ti.func
def changeIntersectFrame(intersect: Intersection, M: tm.mat4, M_inv: tm.mat4) -> Intersection:

    # TODO: Objective 4: Ray and Geometry Transformations
    
    pos_h = tm.vec4(intersect.position, 1.0)
    n_h   = tm.vec4(intersect.normal, 0.0)
    world_pos  = (M @ pos_h).xyz
    world_norm = tm.normalize(((M_inv.transpose()) @ n_h).xyz)
    return Intersection(
        is_hit=intersect.is_hit,
        t=intersect.t,
        normal=world_norm,
        position=world_pos,
        mat=intersect.mat,
        uv=intersect.uv
    )

    #return intersect # change this placeholder to return a transformed intersection
