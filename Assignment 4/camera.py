# Cloris Liu 261010926
from helperclasses import Ray
import taichi as ti
from pyglm import glm
import taichi.math as tm

@ti.data_oriented
class Camera:
	def __init__(self, width, height, eye_position:glm.vec3, lookat:glm.vec3, up:glm.vec3, fovy, aperture:float=0.0, focal_dist:float=0.0) -> None:
		'''	Initialize the camera with given parameters.
		Args:
			width (int): width of the image in pixels
			height (int): height of the image in pixels
			eye_position (glm.vec3): position of the camera in world space
			lookat (glm.vec3): point the camera is looking at
			up (glm.vec3): up direction for the camera
			fovy (float): vertical field of view in degrees
		Returns:
			None

			Note that while glm types are provided to construct the class, the 
			member variables must be of taich tm.vec3 types so that they used 
			in the create_ray Taichi function.  
		'''
		self.width = width
		self.height = height		
		self.distance_to_plane = 1.0

		# TODO: Objective 1: Compute camera frame basis vectors, and top bottom left right for ray generation
		# NOTE: glm vectors are passed in to permit the work to be done here in python, but stored vectors must be tm.vec3
		w = glm.normalize(eye_position - lookat)
		u = glm.normalize(glm.cross(up, w))
		v = glm.cross(w, u)
		aspect = width / height

		# Compute proper values below!
		self.top = ti.field(dtype=ti.f32, shape=())
		self.bottom = ti.field(dtype=ti.f32, shape=())
		self.left = ti.field(dtype=ti.f32, shape=())
		self.right = ti.field(dtype=ti.f32, shape=())
		self.top[None] = glm.tan(glm.radians(fovy) / 2.0) * self.distance_to_plane
		self.bottom[None] = -self.top[None]
		self.left[None] = -self.top[None] * aspect
		self.right[None] = -self.left[None]

		self.u = ti.Vector.field(3, dtype=ti.f32, shape=())
		self.v = ti.Vector.field(3, dtype=ti.f32, shape=())
		self.w = ti.Vector.field(3, dtype=ti.f32, shape=())
		self.eye_position = ti.Vector.field(3, dtype=ti.f32, shape=())
		self.u[None] = tm.vec3(u.x, u.y, u.z)
		self.v[None] = tm.vec3(v.x, v.y, v.z)
		self.w[None] = tm.vec3(w.x, w.y, w.z)
		self.eye_position[None] = tm.vec3(eye_position.x, eye_position.y, eye_position.z)
		self.aperture = ti.field(dtype=ti.f32, shape=())
		self.focal_dist = ti.field(dtype=ti.f32, shape=())
		self.aperture[None] = aperture
		# 默认焦距：眼睛到lookat的距离
		default_fd = glm.length(eye_position - lookat)
		self.focal_dist[None] = focal_dist if focal_dist > 0 else default_fd



	@ti.func
	def create_ray(self, x, y, jitter=False, enable_time=True) -> Ray:
		''' Create a ray going through pixel (x,y) in image space 
		Args:
			x (int): pixel x coordinate
			y (int): pixel y coordinate
			jitter (bool): whether to apply jittering within the pixel for anti-aliasing
		Returns:
			Ray: generated ray from camera through pixel

			If jitter is True, the ray should not go through the exact center of the
			pixel, but instead be through a random point (uniform) inthe pixel area.
		'''

		# TODO: Objective 1: Generate ray from camera through pixel (col, row) with jittering
		dx = ti.random() if jitter else 0.5
		dy = ti.random() if jitter else 0.5
		u_coord = self.left[None] + (self.right[None] - self.left[None]) * (x + dx) / self.width  # 𝑢 = 𝑙 + (𝑟 − 𝑙)(𝑖 + 0.5)/𝑛𝑥
		v_coord = self.bottom[None] + (self.top[None] - self.bottom[None]) * (y + dy) / self.height  # 𝑣 = 𝑏 + (𝑡 − 𝑏)(𝑗 + 0.5)/𝑛𝑦

		# 𝐬 = 𝐞 + 𝑢𝐮 + 𝑣𝐯 − 𝑑𝐰
		s = self.eye_position[None] + u_coord * self.u[None] + v_coord * self.v[None] - self.distance_to_plane * self.w[None]
		direction = tm.normalize(s - self.eye_position[None]) # 𝐝 = 𝐬 − 𝐞

		orig = self.eye_position[None]
		dir_out = direction
		if self.aperture[None] > 0:
			p_focus = self.eye_position[None] + self.focal_dist[None] * direction
			r = ti.sqrt(ti.random()) * self.aperture[None]
			theta = 2.0 * tm.pi * ti.random()
			lens_offset = r * (ti.cos(theta) * self.u[None] + ti.sin(theta) * self.v[None])
			orig = self.eye_position[None] + lens_offset
			dir_out = tm.normalize(p_focus - orig)

		time = ti.random() if enable_time else 0.0  # [0,1) shutter time
		return Ray( orig, dir_out, time )  # temporary ray
