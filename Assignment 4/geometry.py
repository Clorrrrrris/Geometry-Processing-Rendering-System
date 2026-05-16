# Cloris Liu 261010926
from helperclasses import Ray, Intersection, Material, changeRayFrame, getRayPoint, changeIntersectFrame
import config

import taichi as ti
import taichi.math as tm

EPSILON = 10 ** (-5)

@ti.dataclass
class Sphere:
    id: int
    material: Material
    radius: float
    M: tm.mat4
    M_inv: tm.mat4
    velocity: tm.vec3

@ti.func
def intersectSphere(sphere: Sphere, ray: Ray, t_min: float, t_max: float) -> Intersection:
    hit = Intersection() # default is no intersection (is_hit = False)

    moving_ray = Ray(ray.origin - sphere.velocity * ray.time, ray.direction, ray.time)
    local_ray = changeRayFrame(moving_ray, sphere.M_inv)

    center = tm.vec3(0.0, 0.0, 0.0)
    p = local_ray.origin - center

    a = tm.dot(local_ray.direction, local_ray.direction)
    b = 2.0 * tm.dot(p, local_ray.direction)
    c = tm.dot(p, p) - sphere.radius * sphere.radius
    delta = b * b - 4.0 * a * c

    if delta > 0.0:
        sqrt_delta = tm.sqrt(delta)
        t1 = (-b - sqrt_delta) / (2.0 * a)
        t2 = (-b + sqrt_delta) / (2.0 * a)
        t = t1
        if t < t_min:
            t = t2
        if t > t_min and t < t_max:
            pos = local_ray.origin + t * local_ray.direction
            n = tm.normalize(pos - center)
            hit = Intersection(
                is_hit = True,
                t = t,
                position = pos,
                normal = n,
                mat = sphere.material,
                uv = tm.vec2(0.0, 0.0)
            )

    out_hit = hit
    if hit.is_hit:
        out_hit = changeIntersectFrame(hit, sphere.M, sphere.M_inv)
    return out_hit


@ti.dataclass
class Plane:
    id: int
    two_materials: bool
    material1: Material
    material2: Material
    normal: tm.vec3
    M: tm.mat4
    M_inv: tm.mat4
    velocity: tm.vec3

@ti.func
def intersectPlane(plane: Plane, ray: Ray, t_min: float, t_max: float) -> Intersection:
    hit = Intersection() # default is no intersection (is_hit = False)

    moving_ray = Ray(ray.origin - plane.velocity * ray.time, ray.direction, ray.time)
    local_ray = changeRayFrame(moving_ray, plane.M_inv)
    n = tm.normalize(plane.normal)

    denom = tm.dot(n, local_ray.direction)
    if ti.abs(denom) > 1e-6:
        t = -tm.dot(n, local_ray.origin) / denom
        if t_min < t < t_max:
            local_hit_pos = local_ray.origin + t * local_ray.direction
            local_hit_normal = n

            mat = plane.material1
            if plane.two_materials:
                checker = (tm.floor(local_hit_pos.x) + tm.floor(local_hit_pos.z)) % 2
                if checker != 0:
                    mat = plane.material2

            local_hit = Intersection(
                is_hit=True,
                t=t,
                position=local_hit_pos,
                normal=local_hit_normal,
                mat=mat,
                uv = tm.vec2(0.0, 0.0)
            )

            hit = changeIntersectFrame(local_hit, plane.M, plane.M_inv)

    return hit


@ti.dataclass
class AABox:
    id: int
    material: Material
    minpos: tm.vec3
    maxpos: tm.vec3
    M: tm.mat4
    M_inv: tm.mat4
    velocity: tm.vec3

@ti.func
def intersectAABox(aabox: AABox, ray: Ray, t_min: float, t_max: float) -> Intersection:
    hit = Intersection() # default is no intersection (is_hit = False)

    moving_ray = Ray(ray.origin - aabox.velocity * ray.time, ray.direction, ray.time)
    local_ray = changeRayFrame(moving_ray, aabox.M_inv)

    inv_dir = 1.0 / local_ray.direction
    t0 = (aabox.minpos - local_ray.origin) * inv_dir
    t1 = (aabox.maxpos - local_ray.origin) * inv_dir
    tlow = tm.min(t0, t1)
    thigh = tm.max(t0, t1)

    t_entry = tm.max(tlow.x, tm.max(tlow.y, tlow.z))
    t_exit  = tm.min(thigh.x, tm.min(thigh.y, thigh.z))

    if t_entry <= t_exit and t_exit >= t_min and t_entry < t_max:
        t_hit = t_entry
        if t_hit < t_min:
            t_hit = t_exit

        if t_hit >= t_min and t_hit <= t_max:
            local_hit_pos = getRayPoint(local_ray, t_hit)
            local_hit_normal = tm.vec3(0.0)
            if ti.abs(local_hit_pos.x - aabox.minpos.x) < EPSILON:
                local_hit_normal = tm.vec3(-1.0, 0.0, 0.0)
            elif ti.abs(local_hit_pos.x - aabox.maxpos.x) < EPSILON:
                local_hit_normal = tm.vec3(1.0, 0.0, 0.0)
            elif ti.abs(local_hit_pos.y - aabox.minpos.y) < EPSILON:
                local_hit_normal = tm.vec3(0.0, -1.0, 0.0)
            elif ti.abs(local_hit_pos.y - aabox.maxpos.y) < EPSILON:
                local_hit_normal = tm.vec3(0.0, 1.0, 0.0)
            elif ti.abs(local_hit_pos.z - aabox.minpos.z) < EPSILON:
                local_hit_normal = tm.vec3(0.0, 0.0, -1.0)
            else:
                local_hit_normal = tm.vec3(0.0, 0.0, 1.0)

            local_hit = Intersection(
                is_hit = True,
                t = t_hit,
                position = local_hit_pos,
                normal = local_hit_normal,
                mat = aabox.material,
                uv = tm.vec2(0.0, 0.0)
            )
            hit = changeIntersectFrame(local_hit, aabox.M, aabox.M_inv)

    return hit


@ti.dataclass
class Mesh:
    id: int
    material: Material
    faces_ids_start: ti.i32
    faces_ids_count: ti.i32
    bbox_min: tm.vec3
    bbox_max: tm.vec3
    M: tm.mat4
    M_inv: tm.mat4
    velocity: tm.vec3

@ti.func
def intersectMesh(mesh: Mesh,
                  meshes_verts: ti.template(),
                  meshes_faces: ti.template(),
                  meshes_uv: ti.template(),
                  ray: Ray,
                  t_min: float,
                  t_max: float
) -> Intersection:
    out_intersect = Intersection()

    moving_ray = Ray(ray.origin - mesh.velocity * ray.time, ray.direction, ray.time)
    local_ray = changeRayFrame(moving_ray, mesh.M_inv)

    # AABB cull in local space; only traverse if ray hits bbox
    inv_dir = 1.0 / (local_ray.direction + 1e-8)
    t1 = (mesh.bbox_min - local_ray.origin) * inv_dir
    t2 = (mesh.bbox_max - local_ray.origin) * inv_dir
    tmin_x = ti.min(t1.x, t2.x); tmax_x = ti.max(t1.x, t2.x)
    tmin_y = ti.min(t1.y, t2.y); tmax_y = ti.max(t1.y, t2.y)
    tmin_z = ti.min(t1.z, t2.z); tmax_z = ti.max(t1.z, t2.z)
    t_enter = ti.max(tmin_x, ti.max(tmin_y, tmin_z))
    t_exit  = ti.min(tmax_x, ti.min(tmax_y, tmax_z))

    hit_box = (t_exit >= t_min) and (t_enter <= t_max) and (t_enter <= t_exit)
    if hit_box:
        for f in range(mesh.faces_ids_start, mesh.faces_ids_start + mesh.faces_ids_count):
            vidx = meshes_faces[f]
            v0 = meshes_verts[vidx[0]]
            v1 = meshes_verts[vidx[1]]
            v2 = meshes_verts[vidx[2]]
            uv0 = meshes_uv[vidx[0]]
            uv1 = meshes_uv[vidx[1]]
            uv2 = meshes_uv[vidx[2]]
            e1 = v1 - v0
            e2 = v2 - v0

            pvec = tm.cross(local_ray.direction, e2)
            det = tm.dot(e1, pvec)
            if ti.abs(det) < 1e-6:
                continue

            inv_det = 1.0 / det
            tvec = local_ray.origin - v0
            u = tm.dot(tvec, pvec) * inv_det
            if u < 0.0 or u > 1.0:
                continue

            qvec = tm.cross(tvec, e1)
            v = tm.dot(local_ray.direction, qvec) * inv_det
            if v < 0.0 or u + v > 1.0:
                continue

            t = tm.dot(e2, qvec) * inv_det

            if t_min < t < t_max:
                normal = tm.normalize(tm.cross(e1, e2))
                position = local_ray.origin + t * local_ray.direction
                w = 1.0 - u - v
                uv = uv0 * w + uv1 * u + uv2 * v
                
                local_hit = Intersection(
                    is_hit=True,
                    t=t,
                    normal=normal,
                    position=position,
                    mat=mesh.material,
                    uv=uv
                )
                out_intersect = changeIntersectFrame(local_hit, mesh.M, mesh.M_inv)
                t_max = t

    return out_intersect


@ti.dataclass
class Quadric:
    id: int
    material: Material
    qtype: ti.i32      # 0=cylinder, 1=cone
    radius: float
    y_min: float
    y_max: float
    closed: ti.i32
    M: tm.mat4
    M_inv: tm.mat4
    velocity: tm.vec3

@ti.func
def intersectQuadric(q: Quadric, ray: Ray, t_min: float, t_max: float) -> Intersection:
    hit = Intersection()

    moving_ray = Ray(ray.origin - q.velocity * ray.time, ray.direction, ray.time)
    local_ray = changeRayFrame(moving_ray, q.M_inv)
    ox, oy, oz = local_ray.origin.x, local_ray.origin.y, local_ray.origin.z
    dx, dy, dz = local_ray.direction.x, local_ray.direction.y, local_ray.direction.z

    best_t = t_max
    best_n = tm.vec3(0.0)

    if q.qtype == 0:
        a = dx*dx + dz*dz
        b = 2.0*(ox*dx + oz*dz)
        c = ox*ox + oz*oz - q.radius*q.radius
        disc = b*b - 4.0*a*c
        if disc > 0 and a != 0:
            s = tm.sqrt(disc)
            t0 = (-b - s) / (2.0*a)
            t1 = (-b + s) / (2.0*a)
            y0 = oy + t0*dy
            if q.y_min <= y0 <= q.y_max and t_min < t0 < t_max:
                n0 = tm.normalize(tm.vec3(ox + t0*dx, 0.0, oz + t0*dz))
                if t0 < best_t and t0 > t_min:
                    best_t = t0
                    best_n = n0
            y1 = oy + t1*dy
            if q.y_min <= y1 <= q.y_max and t_min < t1 < t_max:
                n1 = tm.normalize(tm.vec3(ox + t1*dx, 0.0, oz + t1*dz))
                if t1 < best_t and t1 > t_min:
                    best_t = t1
                    best_n = n1
        if q.closed != 0 and ti.abs(dy) > 1e-6:
            tcap = (q.y_min - oy) / dy
            if t_min < tcap < t_max:
                x = ox + tcap*dx
                z = oz + tcap*dz
                if x*x + z*z <= q.radius*q.radius + 1e-6:
                    n = tm.vec3(0.0, -1.0, 0.0)
                    if tcap < best_t and tcap > t_min:
                        best_t = tcap
                        best_n = n
            tcap2 = (q.y_max - oy) / dy
            if t_min < tcap2 < t_max:
                x2 = ox + tcap2*dx
                z2 = oz + tcap2*dz
                if x2*x2 + z2*z2 <= q.radius*q.radius + 1e-6:
                    n2 = tm.vec3(0.0, 1.0, 0.0)
                    if tcap2 < best_t and tcap2 > t_min:
                        best_t = tcap2
                        best_n = n2
    else:
        h = q.y_max - q.y_min
        if h != 0:
            k = (q.radius / h)*(q.radius / h)
            oy2 = oy - q.y_min
            a = dx*dx + dz*dz - k*dy*dy
            b = 2.0*(ox*dx + oz*dz - k*oy2*dy)
            c = ox*ox + oz*oz - k*oy2*oy2
            disc = b*b - 4.0*a*c
            if disc > 0 and a != 0:
                s = tm.sqrt(disc)
                t0 = (-b - s)/(2.0*a)
                t1 = (-b + s)/(2.0*a)
                y0 = oy + t0*dy
                if q.y_min <= y0 <= q.y_max and t_min < t0 < t_max:
                    yprime = y0 - q.y_min
                    n0 = tm.normalize(tm.vec3(ox + t0*dx, -k*2.0*yprime, oz + t0*dz))
                    if t0 < best_t and t0 > t_min:
                        best_t = t0
                        best_n = n0
                y1 = oy + t1*dy
                if q.y_min <= y1 <= q.y_max and t_min < t1 < t_max:
                    yprime1 = y1 - q.y_min
                    n1 = tm.normalize(tm.vec3(ox + t1*dx, -k*2.0*yprime1, oz + t1*dz))
                    if t1 < best_t and t1 > t_min:
                        best_t = t1
                        best_n = n1
            if q.closed != 0 and ti.abs(dy) > 1e-6:
                tcap = (q.y_max - oy)/dy
                if t_min < tcap < t_max:
                    x = ox + tcap*dx
                    z = oz + tcap*dz
                    if x*x + z*z <= q.radius*q.radius + 1e-6:
                        n = tm.vec3(0.0, 1.0, 0.0)
                        if tcap < best_t and tcap > t_min:
                            best_t = tcap
                            best_n = n

    if best_t < t_max and best_t > t_min:
        local_hit = Intersection(is_hit=True, t=best_t, position=getRayPoint(local_ray, best_t), normal=best_n, mat=q.material)
        hit = changeIntersectFrame(local_hit, q.M, q.M_inv)

    return hit


@ti.dataclass
class Metaball:
    id: int
    material: Material
    centers_start: ti.i32
    centers_count: ti.i32
    threshold: float
    M: tm.mat4
    M_inv: tm.mat4
    velocity: tm.vec3

@ti.func
def eval_metaball(mb: Metaball, centers: ti.template(), radii: ti.template(), p: tm.vec3) -> float:
    accum = 0.0
    for i in range(mb.centers_start, mb.centers_start + mb.centers_count):
        c = centers[i]
        r = radii[i]
        d = tm.length(p - c)
        if r > 0:
            # smooth falloff to get peanut-like blend rather than a flattened blob
            accum += tm.exp(-(d * d) / (r * r))
    return accum - mb.threshold

@ti.func
def intersectMetaball(mb: Metaball, centers: ti.template(), radii: ti.template(), ray: Ray, t_min: float, t_max: float) -> Intersection:
    hit = Intersection()
    moving_ray = Ray(ray.origin - mb.velocity * ray.time, ray.direction, ray.time)
    local_ray = changeRayFrame(moving_ray, mb.M_inv)
    t = t_min
    for _ in range(config.MB_MAX_STEPS):
        p = getRayPoint(local_ray, t)
        f = eval_metaball(mb, centers, radii, p)
        if f > -config.MB_HIT_EPS:
            # estimate normal via central differences
            h = 1e-3
            nx = eval_metaball(mb, centers, radii, p + tm.vec3(h,0,0)) - eval_metaball(mb, centers, radii, p - tm.vec3(h,0,0))
            ny = eval_metaball(mb, centers, radii, p + tm.vec3(0,h,0)) - eval_metaball(mb, centers, radii, p - tm.vec3(0,h,0))
            nz = eval_metaball(mb, centers, radii, p + tm.vec3(0,0,h)) - eval_metaball(mb, centers, radii, p - tm.vec3(0,0,h))
            n = tm.normalize(tm.vec3(nx, ny, nz))
            pos_out = p + n * config.MB_HIT_EPS * 2.0
            local_hit = Intersection(is_hit=True, t=t, position=pos_out, normal=n, mat=mb.material)
            hit = changeIntersectFrame(local_hit, mb.M, mb.M_inv)
            break
        step = tm.max(ti.abs(f), 0.01)
        t += step
        if t > t_max or t > config.MB_MAX_DIST:
            break
    return hit
