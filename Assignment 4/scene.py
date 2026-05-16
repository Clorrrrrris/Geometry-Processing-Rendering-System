# Cloris Liu 261010926
import geometry as geom
from helperclasses import Ray, Intersection
from camera import Camera
import config

import numpy as np

import taichi as ti
import taichi.math as tm

shadow_epsilon = 10**(-2)

@ti.data_oriented
class Scene:
    def __init__(self,
                 jitter: bool,
                 samples: int,
                 camera: Camera,
                 ambient: tm.vec3,
                 env_map_path: str,
                 lights: ti.template(),
                 nb_lights: int,
                 nb_materials: int,
                 spheres: ti.template(),
                 nb_spheres: int,
                 planes: ti.template(),
                 nb_planes: int,
                 aaboxes: ti.template(),
                 nb_aaboxes: int,
                 quadrics: ti.template(),
                 nb_quadrics: int,
                 metaballs: ti.template(),
                 nb_metaballs: int,
                 mb_centers_np: np.array,
                 mb_radii_np: np.array,
                 meshes: ti.template(),
                 nb_meshes: int,
                 meshes_verts: np.array,
                 meshes_faces: np.array,
                 meshes_uv_np: np.array,
                 textures_np: np.array,
                 tex_dims: list,
                 use_motion: bool = False,
                 use_reflection: bool = False,
                 use_refraction: bool = False,
                 use_textures: bool = False,
                 use_metaballs: bool = False,
                 use_bump: bool = False,
                 ):
        self.jitter_flag = ti.field(dtype=ti.i32, shape=())
        self.jitter_flag[None] = 1 if jitter else 0  # should rays be jittered
        self.samples = samples  # number of rays per pixel
        self.camera = camera
        self.ambient = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.ambient[None] = ambient  # ambient lighting
        self.lights = lights  # all lights in the scene
        self.nb_lights = ti.field(dtype=ti.i32, shape=())
        self.nb_lights[None] = nb_lights
        self.spheres = spheres
        self.planes = planes
        self.aaboxes = aaboxes
        self.quadrics = quadrics
        self.metaballs = metaballs
        self.meshes = meshes
        self.nb_materials = ti.field(dtype=ti.i32, shape=())
        self.nb_materials[None] = nb_materials
        self.nb_spheres = ti.field(dtype=ti.i32, shape=())
        self.nb_spheres[None] = nb_spheres
        self.nb_planes = ti.field(dtype=ti.i32, shape=())
        self.nb_planes[None] = nb_planes
        self.nb_aaboxes = ti.field(dtype=ti.i32, shape=())
        self.nb_aaboxes[None] = nb_aaboxes
        self.nb_quadrics = ti.field(dtype=ti.i32, shape=())
        self.nb_quadrics[None] = nb_quadrics
        self.nb_metaballs = ti.field(dtype=ti.i32, shape=())
        self.nb_metaballs[None] = nb_metaballs
        self.nb_meshes = ti.field(dtype=ti.i32, shape=())
        self.nb_meshes[None] = nb_meshes
        self.use_motion = ti.field(dtype=ti.i32, shape=())
        self.use_motion[None] = 1 if use_motion else 0
        # python-side flags for ti.static pruning
        self.flag_motion = use_motion
        self.use_reflection = ti.field(dtype=ti.i32, shape=())
        self.use_reflection[None] = 1 if use_reflection else 0
        self.flag_reflection = use_reflection
        self.use_refraction = ti.field(dtype=ti.i32, shape=())
        self.use_refraction[None] = 1 if use_refraction else 0
        self.flag_refraction = use_refraction
        self.use_textures = ti.field(dtype=ti.i32, shape=())
        self.use_textures[None] = 1 if use_textures else 0
        self.flag_textures = use_textures
        self.use_metaballs = ti.field(dtype=ti.i32, shape=())
        self.use_metaballs[None] = 1 if use_metaballs else 0
        self.flag_metaballs = use_metaballs
        self.use_bump = ti.field(dtype=ti.i32, shape=())
        self.use_bump[None] = 1 if use_bump else 0
        self.flag_bump = use_bump
        # metaball buffers
        self.mb_centers = ti.Vector.field(3, dtype=ti.f32, shape=config.MAX_MB_POINTS)
        self.mb_radii = ti.field(dtype=ti.f32, shape=config.MAX_MB_POINTS)
        centers_pad = np.zeros((config.MAX_MB_POINTS, 3), dtype=np.float32)
        centers_pad[:mb_centers_np.shape[0]] = mb_centers_np
        self.mb_centers.from_numpy(centers_pad)
        radii_pad = np.zeros((config.MAX_MB_POINTS,), dtype=np.float32)
        radii_pad[:mb_radii_np.shape[0]] = mb_radii_np
        self.mb_radii.from_numpy(radii_pad)

        if meshes_verts.shape[0] > config.MAX_MESH_VERTS:
            raise ValueError(f"Mesh verts {meshes_verts.shape[0]} exceed MAX_MESH_VERTS={config.MAX_MESH_VERTS}")
        if meshes_faces.shape[0] > config.MAX_MESH_FACES:
            raise ValueError(f"Mesh faces {meshes_faces.shape[0]} exceed MAX_MESH_FACES={config.MAX_MESH_FACES}")
        if meshes_uv_np.shape[0] > config.MAX_MESH_UV:
            raise ValueError(f"Mesh uvs {meshes_uv_np.shape[0]} exceed MAX_MESH_UV={config.MAX_MESH_UV}")
        self.meshes_verts = ti.Vector.field(3, shape=config.MAX_MESH_VERTS, dtype=ti.f32)
        padded_verts = np.zeros((config.MAX_MESH_VERTS, 3), dtype=np.float32)
        padded_verts[:meshes_verts.shape[0]] = meshes_verts
        self.meshes_verts.from_numpy(padded_verts)
        self.meshes_faces = ti.Vector.field(3, shape=config.MAX_MESH_FACES, dtype=ti.i32)
        padded_faces = np.zeros((config.MAX_MESH_FACES, 3), dtype=np.int32)
        padded_faces[:meshes_faces.shape[0]] = meshes_faces
        self.meshes_faces.from_numpy(padded_faces)
        self.meshes_uv = ti.Vector.field(2, shape=config.MAX_MESH_UV, dtype=ti.f32)
        uv_pad = np.zeros((config.MAX_MESH_UV, 2), dtype=np.float32)
        uv_pad[:meshes_uv_np.shape[0]] = meshes_uv_np
        self.meshes_uv.from_numpy(uv_pad)

        self.max_depth = 2  # recursion limit for reflections

        self.image = ti.Vector.field( n=3, dtype=ti.f32, shape=(self.camera.width, self.camera.height) )

        self.offsets = ti.field(dtype=ti.f32, shape=((config.MAX_SAMPLES - 1) * (config.MAX_SAMPLES - 1) + 1, 2))

        # env map (optional)
        self.has_env = ti.field(dtype=ti.i32, shape=())
        self.env_w = ti.field(dtype=ti.i32, shape=())
        self.env_h = ti.field(dtype=ti.i32, shape=())
        if env_map_path is not None:
            import matplotlib.image as mpimg
            import pathlib
            p = pathlib.Path(env_map_path)
            if not p.is_file():
                # try relative to scenes folder
                p = pathlib.Path(__file__).parent / env_map_path
            img = mpimg.imread(p)
            if img.dtype != np.float32:
                img = img.astype(np.float32) / (255.0 if img.max() > 1.0 else 1.0)
            if img.shape[-1] == 4:
                img = img[..., :3]
            h, w = img.shape[0], img.shape[1]
            self.env_map = ti.Vector.field(3, dtype=ti.f32, shape=(w, h))
            self.env_map.from_numpy(np.swapaxes(img[:, :, :3], 0, 1))
            self.env_w[None] = w
            self.env_h[None] = h
            self.has_env[None] = 1
        else:
            # dummy env
            self.env_map = ti.Vector.field(3, dtype=ti.f32, shape=(1, 1))
            self.env_map.from_numpy(np.zeros((1, 1, 3), dtype=np.float32))
            self.env_w[None] = 1
            self.env_h[None] = 1
            self.has_env[None] = 0
        # textures
        self.textures = ti.Vector.field(3, dtype=ti.f32, shape=(config.MAX_TEXTURES, config.TEX_H, config.TEX_W))
        self.textures.from_numpy(np.transpose(textures_np, (0,2,1,3))) # [tex, H, W, 3]
        self.tex_w = ti.field(dtype=ti.i32, shape=config.MAX_TEXTURES)
        self.tex_h = ti.field(dtype=ti.i32, shape=config.MAX_TEXTURES)
        for i, (w,h) in enumerate(tex_dims):
            self.tex_w[i] = w
            self.tex_h[i] = h

    @ti.kernel

    def render( self, iteration_count: int ):
        for x,y in ti.ndrange(self.camera.width, self.camera.height):
            if (y == x) and x%10 == 0: print(".",end='')
            ray = self.camera.create_ray( x, y, self.jitter_flag[None] != 0, self.use_motion[None] != 0 )
            intersect = self.intersect_scene(ray, 0, float('inf'))
            sample_colour = tm.vec3(0, 0, 0) # background colour
            if intersect.is_hit:
                sample_colour = self.compute_shading(intersect, ray)
            else:
                sample_colour = self.sample_env(ray.direction)
            self.image[x,y] += (sample_colour - self.image[x,y]) / iteration_count
        print() # end of line after one dot per 10 rows


    @ti.func
    def intersect_scene(self, ray: Ray, t_min: float, t_max: float) -> Intersection:
        best = Intersection() # default is no intersection (is_hit = False)        
        ti.loop_config(serialize=True) 
        for i in range(self.nb_spheres[None]):
            hit = geom.intersectSphere(self.spheres[i], ray, t_min, t_max )
            if hit.is_hit: best = hit; t_max = hit.t # keep best hit only
        ti.loop_config(serialize=True) 
        for i in range(self.nb_planes[None]):
            hit = geom.intersectPlane(self.planes[i], ray, t_min, t_max )
            if hit.is_hit: best = hit; t_max = hit.t                
        ti.loop_config(serialize=True) 
        for i in range(self.nb_aaboxes[None]):
            hit = geom.intersectAABox(self.aaboxes[i], ray, t_min, t_max )
            if hit.is_hit: best = hit; t_max = hit.t
        ti.loop_config(serialize=True) 
        for i in range(self.nb_quadrics[None]):
            hit = geom.intersectQuadric(self.quadrics[i], ray, t_min, t_max)
            if hit.is_hit: best = hit; t_max = hit.t
        ti.loop_config(serialize=True) 
        if ti.static(self.flag_metaballs):
            for i in range(self.nb_metaballs[None]):
                hit = geom.intersectMetaball(self.metaballs[i], self.mb_centers, self.mb_radii, ray, t_min, t_max)
                if hit.is_hit: best = hit; t_max = hit.t
        ti.loop_config(serialize=True) 
        for i in range(self.nb_meshes[None]):
            hit = geom.intersectMesh(self.meshes[i], self.meshes_verts, self.meshes_faces, self.meshes_uv, ray, t_min, t_max)
            if hit.is_hit: best = hit; t_max = hit.t
        return best


    @ti.func
    def shade_direct(self, intersect: Intersection, ray: Ray) -> tm.vec3:
        sample_colour = tm.vec3(0, 0, 0)
        sample_colour += self.ambient[None] * intersect.mat.diffuse
        base_diffuse = intersect.mat.diffuse
        if ti.static(self.flag_textures) and intersect.mat.has_tex != 0 and intersect.mat.tex_id >= 0:
            base_diffuse = base_diffuse * self.sample_texture(intersect.mat.tex_id, intersect.mat.tex_w, intersect.mat.tex_h, intersect.uv)

        ti.loop_config(serialize=True) 
        for l in range(self.nb_lights[None]):
            
            light = self.lights[l]
            light_dir = tm.vec3(0.0) # light direction
            light_dist = 0.0
            attenuation = 1.0

            if light.ltype == 0:
                light_dir = -tm.normalize(light.vector)
                light_dist = float('inf')
                attenuation = 1.0
            elif light.ltype == 1:
                light_dir = light.vector - intersect.position
                light_dist = tm.length(light_dir)
                light_dir = tm.normalize(light_dir)
                attenuation = 1.0 / (light.attenuation[2] + 
                               light.attenuation[1] * light_dist + 
                               light.attenuation[0] * light_dist * light_dist)
            else:
                # area light: sample multiple points on a rectangle
                samples = ti.max(light.samples, 1)
                accum = tm.vec3(0.0)
                for s in range(samples):
                    ru = ti.random()
                    rv = ti.random()
                    # map [0,1) to [-0.5,0.5]
                    offset = (ru - 0.5) * light.u_dir + (rv - 0.5) * light.v_dir
                    p_light = light.vector + offset
                    light_dir = p_light - intersect.position
                    light_dist = tm.length(light_dir)
                    light_dir = tm.normalize(light_dir)
                    # cosine terms
                    N = tm.normalize(intersect.normal)
                    L = light_dir
                    I = light.colour
                    diff_intensity = tm.max(tm.dot(N, L), 0.0)
                    cos_area = tm.max(tm.dot(light.normal, -L), 0.0)
                    # if light faces away, skip this sample
                    if cos_area <= 0:
                        continue
                    attenuation = cos_area / tm.max(light_dist * light_dist, 1e-6)
                    shadow_epsilon = 1e-3
                    shadow_origin = intersect.position + shadow_epsilon * intersect.normal
                    shadow_ray = Ray(shadow_origin, light_dir, ray.time)
                    occluded = False
                    shadow_hit = self.intersect_scene(shadow_ray, shadow_epsilon, light_dist)
                    if shadow_hit.is_hit:
                        occluded = True
                    if not occluded:
                        diffuse = base_diffuse * diff_intensity * I * attenuation
                        V = tm.normalize(-ray.direction)
                        H = tm.normalize(L + V)
                        spec_intensity = tm.pow(tm.max(tm.dot(N, H), 0.0), intersect.mat.shininess[0])
                        specular = intersect.mat.specular * spec_intensity * I * attenuation
                        accum += diffuse + specular
                if samples > 0:
                    sample_colour += accum / samples
                continue


            # diffuse Lambertian
            N = self.bump_normal(intersect.position, intersect.normal) if self.flag_bump else tm.normalize(intersect.normal)
            L = light_dir
            I = light.colour
            diff_intensity = tm.max(tm.dot(N, L), 0.0)
            diffuse = base_diffuse * diff_intensity * I * attenuation

            # Blinn-Phong specular
            V = tm.normalize(-ray.direction)
            H = tm.normalize(L + V)
            spec_intensity = tm.pow(tm.max(tm.dot(N, H), 0.0), intersect.mat.shininess[0])
            specular = intersect.mat.specular * spec_intensity * I * attenuation

            # TODO: Objective 6: Implement shadow rays
            shadow_epsilon = 1e-3
            shadow_origin = intersect.position + shadow_epsilon * intersect.normal
            shadow_ray = Ray(shadow_origin, light_dir, ray.time)

            occluded = False
            shadow_hit = self.intersect_scene(shadow_ray, shadow_epsilon, light_dist)
            if shadow_hit.is_hit:
                occluded = True

            # only add lighting if not occluded
            if not occluded:
                sample_colour += diffuse + specular

        return sample_colour

    @ti.func
    def sample_env(self, direction: tm.vec3) -> tm.vec3:
        colour = tm.vec3(0.0)
        if self.has_env[None] != 0:
            d = tm.normalize(direction)
            u = 0.5 + tm.atan2(d.z, d.x) / (2.0 * tm.pi)
            v = 0.5 - tm.asin(d.y) / tm.pi
            u = tm.clamp(u, 0.0, 0.9999)
            v = tm.clamp(v, 0.0, 0.9999)
            iw = self.env_w[None]
            ih = self.env_h[None]
            ix = tm.min(int(u * iw), iw - 1)
            iy = tm.min(int(v * ih), ih - 1)
            colour = self.env_map[ix, iy]
        return colour

    @ti.func
    def sample_texture(self, tex_id: int, w: int, h: int, uv: tm.vec2) -> tm.vec3:
        colour = tm.vec3(1.0)
        if not (tex_id < 0 or tex_id >= config.MAX_TEXTURES or w <= 0 or h <= 0):
            u = uv.x - tm.floor(uv.x)
            v = uv.y - tm.floor(uv.y)
            ix = int(u * float(w - 1))
            iy = int((1.0 - v) * float(h - 1))
            ix = tm.clamp(ix, 0, w - 1)
            iy = tm.clamp(iy, 0, h - 1)
            colour = self.textures[tex_id, iy, ix]
        return colour

    @ti.func
    def hash31(self, p: tm.vec3) -> float:
        # float hash: frac(sin(dot(p, k)) * c)
        return tm.fract(tm.sin(tm.dot(p, tm.vec3(12.9898, 78.233, 37.719))) * 43758.5453)

    @ti.func
    def value_noise(self, p: tm.vec3) -> float:
        pi = tm.vec3(ti.floor(p))
        pf = p - pi
        res = 0.0
        for dx in ti.static(range(2)):
            for dy in ti.static(range(2)):
                for dz in ti.static(range(2)):
                    w = tm.vec3(float(dx), float(dy), float(dz))
                    h = self.hash31(pi + w)
                    fade = (tm.vec3(1.0) - ti.abs(pf - w))
                    res += h * fade.x * fade.y * fade.z
        return res

    @ti.func
    def bump_normal(self, pos: tm.vec3, normal: tm.vec3) -> tm.vec3:
        if ti.static(not self.flag_bump):
            return tm.normalize(normal)
        freq = 3.0
        amp = 0.2
        eps = 0.1
        p = pos * freq
        n0 = self.value_noise(p)
        nx = self.value_noise(p + tm.vec3(eps, 0, 0)) - n0
        ny = self.value_noise(p + tm.vec3(0, eps, 0)) - n0
        nz = self.value_noise(p + tm.vec3(0, 0, eps)) - n0
        grad = tm.vec3(nx, ny, nz)
        bumped = tm.normalize(normal + amp * grad)
        return bumped
    
    @ti.func
    def compute_shading(self, intersect: Intersection, ray: Ray) -> tm.vec3:
        sample_colour = self.shade_direct(intersect, ray)

        # Feature flags
        enable_refl = ti.static(self.flag_reflection)
        enable_refr = ti.static(self.flag_refraction)

        refl = intersect.mat.reflectivity if enable_refl else tm.vec3(0.0)
        trans = intersect.mat.transmission if enable_refr else tm.vec3(0.0)
        refl_col = tm.vec3(0.0)
        refl_factor = tm.vec3(0.0)
        refr_col = tm.vec3(0.0)
        refr_factor = tm.vec3(0.0)

        # reflection
        if refl.x > 0 or refl.y > 0 or refl.z > 0 or trans.x > 0 or trans.y > 0 or trans.z > 0:
            N = tm.normalize(intersect.normal)
            I = tm.normalize(ray.direction)
            R = I - 2.0 * tm.dot(I, N) * N
            shadow_epsilon = 1e-3
            refl_origin = intersect.position + shadow_epsilon * N
            refl_ray = Ray(refl_origin, R, ray.time)
            refl_hit = self.intersect_scene(refl_ray, shadow_epsilon, float('inf'))
            if refl_hit.is_hit:
                refl_col = self.shade_direct(refl_hit, refl_ray)
            else:
                refl_col = self.sample_env(refl_ray.direction)

            cos_theta = tm.max(-tm.dot(I, N), 0.0)
            r0 = tm.pow((intersect.mat.ior - 1.0) / (intersect.mat.ior + 1.0), 2.0)
            fresnel = r0 + (1.0 - r0) * tm.pow(1.0 - cos_theta, 5.0)
            refl_factor = refl + (1.0 - refl) * fresnel
            
            # refraction
            if enable_refr and (trans.x > 0 or trans.y > 0 or trans.z > 0):
                eta = 1.0 / intersect.mat.ior
                n = N
                cosi = -tm.dot(I, n)
                if cosi < 0:
                    cosi = -cosi
                    n = -n
                    eta = intersect.mat.ior
                k = 1.0 - eta * eta * (1.0 - cosi * cosi)
                if k >= 0:
                    refr_dir = tm.normalize(eta * I + (eta * cosi - tm.sqrt(k)) * n)
                    refr_origin = intersect.position + shadow_epsilon * refr_dir
                    refr_ray = Ray(refr_origin, refr_dir, ray.time)
                    refr_hit = self.intersect_scene(refr_ray, shadow_epsilon, float('inf'))
                    
                    # if still inside the same material, continue to next hit to exit
                    if refr_hit.is_hit and refr_hit.mat.id == intersect.mat.id:
                        pass_origin = refr_hit.position + shadow_epsilon * refr_dir
                        pass_ray = Ray(pass_origin, refr_dir, ray.time)
                        refr_hit = self.intersect_scene(pass_ray, shadow_epsilon, float('inf'))
                    
                    if refr_hit.is_hit:
                        refr_col = self.shade_direct(refr_hit, refr_ray)
                    else:
                        refr_col = self.sample_env(refr_ray.direction)
                    refr_factor = trans * (tm.vec3(1.0) - fresnel)

            remain = tm.vec3(1.0) - refl_factor - refr_factor
            remain = tm.max(remain, 0.0)
            sample_colour = sample_colour * remain + refl_factor * refl_col + refr_factor * refr_col

        return sample_colour
