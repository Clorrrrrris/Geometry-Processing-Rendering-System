# Cloris Liu 261010926
import json5 as json
import helperclasses as hc
from camera import Camera
import geometry as geom
import scene
import trimesh
import numpy as np
import taichi.math as tm
from pyglm import glm
import config
import matplotlib.image as mpimg
import pathlib

geom_id = -1  # global geometry ID counter
meshes_total_nb_verts = 0  # global counter of total number of mesh vertices in the scene
meshes_total_nb_faces = 0  # global counter of total number of mesh faces in the scene
scene_meshes_verts = np.empty((0, 3), dtype=np.float32)
scene_meshes_faces = np.empty((0, 3), dtype=np.int32)
mb_points_count = 0
mb_centers = np.zeros((config.MAX_MB_POINTS, 3), dtype=np.float32)
mb_radii = np.zeros((config.MAX_MB_POINTS,), dtype=np.float32)
scene_meshes_uv = np.empty((0, 2), dtype=np.float32)
textures = np.zeros((config.MAX_TEXTURES, config.TEX_W, config.TEX_H, 3), dtype=np.float32)
tex_dims = []

def load_scene(infile: str, image_scale_factor: float = 1.0) -> scene.Scene:
    ''' Load a scene from a json file 
    Args:
        infile (str): path to the scene json file
        image_scale_factor (float): scale factor for image resolution
    Returns:
        scene.Scene: the loaded scene object

        The json file can define a hierarchy of nodes, but they will be flattend into a list of geometries in the returned scene
        by accumulating the transformation matrices down the hierarchy.  Instances are provided as a convenience for
    '''
    global geom_id, meshes_total_nb_verts, meshes_total_nb_faces, scene_meshes_verts, scene_meshes_faces, scene_meshes_uv, mb_points_count, mb_centers, mb_radii, textures, tex_dims

    # reset globals so multiple scenes in one run start fresh while keeping shapes fixed by config
    geom_id = -1
    meshes_total_nb_verts = 0
    meshes_total_nb_faces = 0
    scene_meshes_verts = np.empty((0, 3), dtype=np.float32)
    scene_meshes_faces = np.empty((0, 3), dtype=np.int32)
    scene_meshes_uv = np.empty((0, 2), dtype=np.float32)
    mb_points_count = 0
    mb_centers = np.zeros((config.MAX_MB_POINTS, 3), dtype=np.float32)
    mb_radii = np.zeros((config.MAX_MB_POINTS,), dtype=np.float32)
    textures = np.zeros((config.MAX_TEXTURES, config.TEX_W, config.TEX_H, 3), dtype=np.float32)
    tex_dims = []
    print("Parsing file:", infile)
    f = open(infile)
    data = json.load(f)
    fast_mode = bool(data.get("fast_mode", True))
    features = data.get("features", {})
    # defaults to False for speed unless explicitly enabled per scene
    feature_reflection = bool(features.get("reflections", False))
    feature_refraction = bool(features.get("refractions", False))
    feature_textures = bool(features.get("textures", False))
    feature_metaballs = bool(features.get("metaballs", False))
    feature_bump = bool(features.get("bump", False))
    scene_has_motion = bool(data.get("motion_blur", False))

    # Loading resolution
    default_resolution = [960, 540]    
    width = int(image_scale_factor * data.get("resolution", default_resolution)[0])
    height = int(image_scale_factor * data.get("resolution", default_resolution)[1])
    use_common_res = data.get("force_common_res", True)
    if use_common_res:
        width = config.COMMON_RES_W
        height = config.COMMON_RES_H
    elif fast_mode:
        width = min(width, 512)
        height = min(height, 512)
        
    # Loading camera
    cam_pos = glm.vec3(data["camera"]["position"])
    cam_lookat = glm.vec3(data["camera"]["lookAt"])
    cam_up = glm.vec3(data["camera"]["up"])
    cam_fovy = data["camera"]["fovy"]
    cam_aperture = data["camera"].get("aperture", 0.0)
    cam_focal = data["camera"].get("focal_dist", 0.0)
    camera = Camera(width, height, cam_pos, cam_lookat, cam_up, cam_fovy, cam_aperture, cam_focal)

    # Loading ambient light
    ambient = tm.vec3(data.get("ambient", [0.1, 0.1, 0.1])) # set a reasonable default ambient light
    env_map_path = data.get("env_map", None)
    if fast_mode:
        env_map_path = None

    # Loading Anti-Aliasing options    
    jitter = data.get( "AA_jitter", False ) # default to no jitter
    samples = data.get( "AA_samples", 1 ) # default to no supersampling
    if fast_mode:
        samples = 1
    
    # Loading scene lights
    lights_tmp = []
    for light in data.get("lights", []):
        l_type = light["type"]
        l_name = light["name"]
        l_colour = tm.vec3(light["colour"])
        l_power = light.get( "power", 1.0 ) # The power scales the specified light colour
        if l_type == "point":
            l_vector = tm.vec3(light["position"])
            l_attenuation = tm.vec3(0,0,1) if "attenuation" not in light else tm.vec3( light["attenuation"] )
            l_type = 1
            u_dir = tm.vec3(0.0)
            v_dir = tm.vec3(0.0)
            l_normal = tm.vec3(0.0)
            l_samples = 1
        elif l_type == "directional":
            l_vector = tm.normalize(tm.vec3(light["direction"]))
            l_attenuation = tm.vec3(0,0,0)
            l_type = 0
            u_dir = tm.vec3(0.0)
            v_dir = tm.vec3(0.0)
            l_normal = tm.vec3(0.0)
            l_samples = 1
            if "attenuation" in light:
                print("Directional light", l_name, "has attenuation, ignoring")
        elif l_type == "area":
            # rectangular area light defined by center (position) and two half-extent vectors
            l_vector = tm.vec3(light["position"])
            u_dir_list = light.get("u_dir", [1.0, 0.0, 0.0])
            v_dir_list = light.get("v_dir", [0.0, 0.0, 1.0])
            u_dir = tm.vec3(u_dir_list)
            v_dir = tm.vec3(v_dir_list)
            # compute normal in python scope using glm to avoid taichi calls here
            if "normal" in light:
                n = glm.normalize(glm.vec3(light["normal"]))
            else:
                n = glm.normalize(glm.cross(glm.vec3(u_dir_list), glm.vec3(v_dir_list)))
            l_normal = tm.vec3(n.x, n.y, n.z)
            l_attenuation = tm.vec3(0,0,1)
            l_samples = int(light.get("samples", 8))
            l_type = 2
        else:
            print("Unkown light type", l_type, ", skipping initialization")
            continue
        lights_tmp.append(hc.Light(l_type, len(lights_tmp), l_colour * l_power, l_vector, l_attenuation, u_dir, v_dir, l_normal, l_samples))

    # Taichi does not accept a zero-length field, so we create a field of size 1 if the list is empty.
    # This means we need to keep track of the actual number of lights separately.
    nb_lights = len(lights_tmp)
    if nb_lights > config.MAX_LIGHTS:
        raise ValueError(f"Scene has {nb_lights} lights, exceeds MAX_LIGHTS={config.MAX_LIGHTS}")
    lights = hc.Light.field(shape=config.MAX_LIGHTS)
    for i in range(len(lights_tmp)):
        lights[i] = lights_tmp[i]

    # Loading materials
    material_by_name = {} # materials dictionary
    mat_id = 0
    for material in data["materials"]:
        mat_name = material["name"]
        mat_diffuse = tm.vec3(material["diffuse"])
        mat_specular = tm.vec3(material["specular"])
        mat_shininess = 0 if "shininess" not in material else material["shininess"]
        mat_reflect = tm.vec3(material.get("reflectivity", [0.0, 0.0, 0.0]))
        mat_ior = float(material.get("ior", 1.5))
        mat_trans = tm.vec3(material.get("transmission", [0.0, 0.0, 0.0]))
        tex_id = -1
        tex_w = 0
        tex_h = 0
        has_tex = 0
        if not fast_mode and feature_textures and "texture" in material:
            tex_path = material["texture"]
            p = pathlib.Path(tex_path)
            if not p.is_file():
                p = pathlib.Path(__file__).parent / tex_path
            img = mpimg.imread(p)
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
            if img.shape[-1] == 4:
                img = img[...,:3]
            h, w = img.shape[0], img.shape[1]
            tex_id = len(tex_dims)
            if tex_id >= config.MAX_TEXTURES:
                raise ValueError("Texture count exceeds MAX_TEXTURES")
            tex_dims.append((w,h))
            # resize or pad to TEX_W/H
            canvas = np.zeros((config.TEX_H, config.TEX_W, 3), dtype=np.float32)
            copy_h = min(h, config.TEX_H)
            copy_w = min(w, config.TEX_W)
            canvas[:copy_h, :copy_w, :] = img[:copy_h, :copy_w, :]
            textures[tex_id] = canvas
            tex_w, tex_h = w, h
            has_tex = 1
        material_by_name[mat_name] = hc.Material(mat_id, mat_diffuse, mat_specular, mat_shininess, mat_reflect, mat_ior, mat_trans, has_tex, tex_id, tex_w, tex_h)
        mat_id += 1
    if mat_id > config.MAX_MATERIALS:
        raise ValueError(f"Scene has {mat_id} materials, exceeds MAX_MATERIALS={config.MAX_MATERIALS}")

    # load geometires
    objects = {"sphere": [],
               "plane": [],
               "box": [],
               "mesh": [],
               "quadric": [],
               "metaball": []}  # lists of loaded object geometries and hierarchy roots
    node_by_name = {}  # dictionary of geometries by name (for instances)

    M_parent = tm.mat4(np.eye(4))  # identity matrix as the initial parent transformation

    for geometry in data["objects"]:
        g_type = geometry["type"]
        if g_type == "node":
            g = load_node(geometry, material_by_name, node_by_name, M_parent)
            for obj_type, obj in g:
                objects[obj_type].append(obj)
        elif g_type == "instance":
            g = load_instance(geometry, node_by_name)
            for obj_type, obj in g:
                objects[obj_type].append(obj)
        else:
            g = load_geometry(geometry, material_by_name, M_parent)
            if g is not None:
                target_type = g_type
                if g_type == "cylinder" or g_type == "cone":
                    target_type = "quadric"
                objects[target_type].append(g)
                # check if "name" field exists
                if "name" in geometry:
                    node_by_name[geometry["name"]] = g

    print("Loaded", geom_id + 1, "geometric objects")

    # Taichi does not accept a zero-length field, so we create a field of size 1 if the list is empty.
    # This means we need to keep track of the actual number of objects separately.
    nb_spheres = len(objects["sphere"])
    if nb_spheres > config.MAX_SPHERES:
        raise ValueError(f"Scene has {nb_spheres} spheres, exceeds MAX_SPHERES={config.MAX_SPHERES}")
    spheres = geom.Sphere.field(shape=config.MAX_SPHERES)
    for i in range(len(objects["sphere"])):
        spheres[i] = objects["sphere"][i]

    nb_planes = len(objects["plane"])
    if nb_planes > config.MAX_PLANES:
        raise ValueError(f"Scene has {nb_planes} planes, exceeds MAX_PLANES={config.MAX_PLANES}")
    planes = geom.Plane.field(shape=config.MAX_PLANES)
    for i in range(len(objects["plane"])):
        planes[i] = objects["plane"][i]

    nb_boxes = len(objects["box"])
    if nb_boxes > config.MAX_BOXES:
        raise ValueError(f"Scene has {nb_boxes} boxes, exceeds MAX_BOXES={config.MAX_BOXES}")
    boxes = geom.AABox.field(shape=config.MAX_BOXES)
    for i in range(len(objects["box"])):
        boxes[i] = objects["box"][i]

    nb_quadrics = len(objects["quadric"])
    if nb_quadrics > config.MAX_QUADRICS:
        raise ValueError(f"Scene has {nb_quadrics} quadrics, exceeds MAX_QUADRICS={config.MAX_QUADRICS}")
    quadrics = geom.Quadric.field(shape=config.MAX_QUADRICS)
    for i in range(len(objects["quadric"])):
        quadrics[i] = objects["quadric"][i]

    nb_metaballs = len(objects["metaball"])
    if nb_metaballs > config.MAX_METABALLS:
        raise ValueError(f"Scene has {nb_metaballs} metaballs, exceeds MAX_METABALLS={config.MAX_METABALLS}")
    metaballs = geom.Metaball.field(shape=config.MAX_METABALLS)
    for i in range(len(objects["metaball"])):
        metaballs[i] = objects["metaball"][i]

    nb_meshes = len(objects["mesh"])
    if nb_meshes > config.MAX_MESHES:
        raise ValueError(f"Scene has {nb_meshes} meshes, exceeds MAX_MESHES={config.MAX_MESHES}")
    meshes = geom.Mesh.field(shape=config.MAX_MESHES)
    for i in range(len(objects["mesh"])):
        meshes[i] = objects["mesh"][i]

    if meshes_total_nb_verts > config.MAX_MESH_VERTS or scene_meshes_uv.shape[0] > config.MAX_MESH_UV:
        raise ValueError(f"Scene has {meshes_total_nb_verts} mesh vertices, exceeds MAX_MESH_VERTS={config.MAX_MESH_VERTS}")
    if meshes_total_nb_faces > config.MAX_MESH_FACES:
        raise ValueError(f"Scene has {meshes_total_nb_faces} mesh faces, exceeds MAX_MESH_FACES={config.MAX_MESH_FACES}")

    return scene.Scene( jitter, samples,  # General settings
                camera,  # Camera settings
                ambient, env_map_path, lights, nb_lights,  # Light settings
                mat_id,  # material count
                spheres, nb_spheres,
                planes, nb_planes,
                boxes, nb_boxes,
                quadrics, nb_quadrics,
                metaballs, nb_metaballs, mb_centers, mb_radii,
                meshes, nb_meshes, scene_meshes_verts, scene_meshes_faces, scene_meshes_uv,
                textures, tex_dims,
                scene_has_motion and not fast_mode,
                feature_reflection and not fast_mode,
                feature_refraction and not fast_mode,
                feature_textures and not fast_mode,
                feature_metaballs and not fast_mode,
                feature_bump and not fast_mode)  # Geometry settings

def mat4_glm_to_ti( M_glm: glm.mat4 ) -> tm.mat4:
    return tm.mat4( glm.transpose(M_glm).to_list() )

def load_geometry_transformation_matrix(geometry, M_parent: tm.mat4) -> (tm.mat4, tm.mat4):
    g_pos = glm.vec3(geometry.get("position", [0, 0, 0]))
    g_r = glm.vec3(geometry.get("rotation", [0, 0, 0]))  # not really useful for a sphere...
    g_s = geometry.get("scale", [1, 1, 1])
    if type(g_s) == float or type(g_s) == int:
        g_s = [g_s, g_s, g_s]
    g_s = glm.vec3(g_s)
    scale = glm.scale( g_s )
    rot_x = glm.rotate( glm.radians(g_r.x), glm.vec3(1,0,0) )
    rot_y = glm.rotate( glm.radians(g_r.y), glm.vec3(0,1,0) )
    rot_z = glm.rotate( glm.radians(g_r.z), glm.vec3(0,0,1) )
    translate = glm.translate( g_pos )
    M_parent_glm = glm.mat4( M_parent.to_numpy() )
    M = M_parent_glm * translate * rot_x * rot_y * rot_z * scale
    M_inv = glm.inverse(M)
    return mat4_glm_to_ti(M), mat4_glm_to_ti(M_inv)

def load_geometry(geometry, material_by_name, M_parent: tm.mat4 ):
    global geom_id, meshes_total_nb_verts, meshes_total_nb_faces, scene_meshes_verts, scene_meshes_faces, scene_meshes_uv, mb_points_count, mb_centers, mb_radii

    # Elements common to all objects: name, type, and material(s)
    g_type = geometry["type"]
    g_materials = [ material_by_name[material_name] for material_name in geometry.get("materials",[]) ]

    geom_id += 1

    if g_type == "sphere":
        g_radius = geometry.get("radius",1)
        vel = tm.vec3(geometry.get("velocity", [0,0,0]))
        M, M_inv = load_geometry_transformation_matrix(geometry, M_parent)
        return geom.Sphere(geom_id, g_materials[0], g_radius, M, M_inv, vel)
    elif g_type == "plane":
        g_normal = tm.vec3(geometry.get("normal",[0,1,0]))
        vel = tm.vec3(geometry.get("velocity", [0,0,0]))
        M, M_inv = load_geometry_transformation_matrix(geometry, M_parent)
        two_materials = True if len(g_materials) > 1 else False
        mat1 = g_materials[0]
        mat2 = g_materials[1] if two_materials else g_materials[0]
        return geom.Plane(geom_id, two_materials, mat1, mat2, g_normal, M, M_inv, vel)
    elif g_type == "box":
        minpos = tm.vec3(geometry.get("min",[-1,-1,-1]))
        maxpos = tm.vec3(geometry.get("max",[1,1,1]))
        vel = tm.vec3(geometry.get("velocity", [0,0,0]))
        M, M_inv = load_geometry_transformation_matrix(geometry, M_parent)
        return geom.AABox(geom_id, g_materials[0], minpos, maxpos, M, M_inv, vel)
    elif g_type == "cylinder" or g_type == "cone":
        radius = geometry.get("radius", 1.0)
        y_min = geometry.get("y_min", 0.0)
        y_max = geometry.get("y_max", 1.0)
        closed = 1 if geometry.get("closed", True) else 0
        qtype = 0 if g_type == "cylinder" else 1
        vel = tm.vec3(geometry.get("velocity", [0,0,0]))
        M, M_inv = load_geometry_transformation_matrix(geometry, M_parent)
        return geom.Quadric(geom_id, g_materials[0], qtype, radius, y_min, y_max, closed, M, M_inv, vel)
    elif g_type == "mesh":
        g_path = geometry["filepath"]
        vel = tm.vec3(geometry.get("velocity", [0,0,0]))
        M, M_inv = load_geometry_transformation_matrix(geometry, M_parent)
        mesh = trimesh.load_mesh(g_path)
        verts = mesh.vertices
        faces = mesh.faces
        bbox_min_np = verts.min(axis=0) if len(verts) > 0 else np.zeros(3)
        bbox_max_np = verts.max(axis=0) if len(verts) > 0 else np.zeros(3)
        # small padding to avoid numerical issues
        bbox_min = tm.vec3(bbox_min_np - 1e-4)
        bbox_max = tm.vec3(bbox_max_np + 1e-4)

        scene_meshes_verts = np.resize(scene_meshes_verts, (meshes_total_nb_verts + len(verts), 3))
        scene_meshes_faces = np.resize(scene_meshes_faces, (meshes_total_nb_faces + len(faces), 3))
        scene_meshes_uv = np.resize(scene_meshes_uv, (meshes_total_nb_verts + len(verts), 2))
        # NOTE: These should really be vectorized!
        for i in range(len(verts)):
            scene_meshes_verts[meshes_total_nb_verts + i] = np.array((verts[i, 0], verts[i, 1], verts[i, 2]))
            # planar UV projection (x,z) normalized to [0,1]
            scene_meshes_uv[meshes_total_nb_verts + i] = np.array((0.5 * verts[i,0] + 0.5, 0.5 * verts[i,2] + 0.5))
        for i in range(len(faces)):
            scene_meshes_faces[meshes_total_nb_faces + i] = np.array((faces[i, 0] + meshes_total_nb_verts,
                                                                      faces[i, 1] + meshes_total_nb_verts,
                                                                      faces[i, 2] + meshes_total_nb_verts))
        # NOTE: an opportunity to transform the verts of the mesh rather than transforming the ray later
        mesh = geom.Mesh(geom_id, g_materials[0], meshes_total_nb_faces, len(faces), bbox_min, bbox_max, M, M_inv, vel)
        meshes_total_nb_verts += len(verts)
        meshes_total_nb_faces += len(faces)
        return mesh
    elif g_type == "metaball":
        centers = geometry["centers"]
        radii = geometry["radii"]
        threshold = geometry.get("threshold", 1.0)
        if len(centers) != len(radii):
            raise ValueError("metaball centers and radii must have same length")
        start = mb_points_count
        needed = len(centers)
        if start + needed > config.MAX_MB_POINTS:
            raise ValueError(f"Metaball points exceed MAX_MB_POINTS={config.MAX_MB_POINTS}")
        for i, c in enumerate(centers):
            mb_centers[start + i] = np.array((c[0], c[1], c[2]), dtype=np.float32)
            mb_radii[start + i] = float(radii[i])
        mb_points_count = start + needed
        vel = tm.vec3(geometry.get("velocity", [0,0,0]))
        M, M_inv = load_geometry_transformation_matrix(geometry, M_parent)
        return geom.Metaball(geom_id, g_materials[0], start, needed, threshold, M, M_inv, vel)
    else:
        print("Unkown object type", g_type, ", skipping initialization")
        geom_id -= 1  # we cancel the increment of geom_id since we didn't create any geometry
        return None
    
def load_node(geometry, material_by_name, node_by_name, M_parent: tm.mat4 ):
    M, M_inv = load_geometry_transformation_matrix(geometry, M_parent)    
    # For this node, keep a list of all the childern objects
    objects = []
    node_by_name[geometry["name"]] = []
    
    for child in geometry["children"]:
        if child["type"] != "node" and child["type"] != "instance":
            g = load_geometry(child, material_by_name, M)
            target_type = child["type"]
            if target_type == "cylinder" or target_type == "cone":
                target_type = "quadric"
            objects.append((target_type, g))  # each geometry is stored as a tuple (type, object)
            node_by_name[geometry["name"]] += objects
        elif child["type"] == "node":
            objects += load_node(child, material_by_name, node_by_name, M)
            node_by_name[geometry["name"]] += objects
        elif child["type"] == "instance":
            print("Instances are not allowed inside nodes, must be defined at root level")
        else:
            print("Unkown object type", child["type"], ", skipping initialization")

    return objects

def load_instance(geometry, node_by_name):
    # instances are loaded off the root, so will have an identiy matrix as parent
    M, M_inv = load_geometry_transformation_matrix(geometry, tm.mat4( np.eye(4) ) ) 
    objects = []
    node = node_by_name[geometry["ref"]]
    for obj_type, obj in node:
        if obj is not None:
            if obj_type == "sphere":
                new_obj = geom.Sphere(obj.id, obj.material, obj.position, obj.radius, M @ obj.M, obj.M_inv @ M_inv, obj.velocity )                                      
            elif obj_type == "plane":
                new_obj = geom.Plane(obj.id, obj.two_materials, obj.material1, obj.material2, obj.position, obj.normal, M @ obj.M, obj.M_inv @ M_inv, obj.velocity )            
            elif obj_type == "box":
                new_obj = geom.AABox(obj.id, obj.material, obj.minpos, obj.maxpos, M @ obj.M, obj.M_inv @ M_inv, obj.velocity )  
            elif obj_type == "mesh":
                new_obj = geom.Mesh(obj.id, obj.material, obj.faces_ids_start, obj.faces_ids_count, obj.bbox_min, obj.bbox_max, M @ obj.M, obj.M_inv @ M_inv, obj.velocity )                          
            elif obj_type == "quadric":
                new_obj = geom.Quadric(obj.id, obj.material, obj.qtype, obj.radius, obj.y_min, obj.y_max, obj.closed, M @ obj.M, obj.M_inv @ M_inv, obj.velocity)
            else:
                new_obj = obj
            objects.append((obj_type, new_obj))

    return objects
