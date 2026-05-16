# Cloris Liu 261010926
import trimesh
import numpy as np
from pathlib import Path
import moderngl as mgl
from sortedcontainers import SortedList
from pyglm import glm
from PyQt5 import QtOpenGL
from heds import HalfEdge, Vertex, EdgeCollapseData, build_heds, CollapseRecord
from moderngl_text.text_renderer import TextRenderer


class SimplificationViewer(QtOpenGL.QGLWidget):
    """ OpenGL widget with simple viewing controls """

    def __init__(self):
        fmt = QtOpenGL.QGLFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QtOpenGL.QGLFormat.CoreProfile)
        fmt.setSampleBuffers(True)
        super(SimplificationViewer, self).__init__(fmt, None)
        self.update_UI_callback = None  # notify UI of changes (e.g., LOD slider)
        self.keyboard_callback = None  # make keyboard usable
        self.draw_quadrics = False
        self.draw_face_IDs = False
        self.draw_vertex_IDs = False
        self.draw_current_he = True
        self.scale_with_LOD = False        
        self.mesh_wireframe = False
        self.current_LOD = 0
        self.max_LOD = 0  # the number of simplification levels available
        self.collapse_history = []  # keep track of collapse records for undoing
        self.last_mouse_pos = None
        self.R = glm.mat4(1)  # current rotation matrix for view
        self.d = 4.0  # current distance of the camera

        self.vao, self.vbo, self.ibo = None, None, None  # Will be created when mesh is loaded
        self.text_renderer = None # used for rendering text labels on verts and faces
        self.ctx = None
        self.prog_tris = None  # Mesh shader
        self.prog_lines = None  # Line shader
        self.width, self.height = None, None
        self.P, self.V, self.S = None, None, None  # Projection, View, Scale matrices

        # For visualizing half edges
        self.he_geom, self.he_vbo, self.he_vao, self.current_he = None, None, None, None
        # General data
        self.indices, self.faces, self.verts = None, None, None
        self.vert_objs, self.face_objs = None, None
        self.sorted_edge_list = SortedList()

    def set_update_UI_callback(self, callback):
        self.update_UI_callback = callback

    def set_keyboard_callback(self, callback):
        self.keyboard_callback = callback

    def keyPressEvent(self, event):
        if self.keyboard_callback is not None:
            self.keyboard_callback(event)

    def initializeGL(self):
        self.ctx = mgl.create_context()
        self.text_renderer = TextRenderer(self.ctx)
        current_dir = Path(__file__).parent  # glsl folder in same directory as this code
        self.setup_mesh_shaders_and_buffers(current_dir)
        self.setup_half_edge_shaders_and_buffers(current_dir)
        mesh_file = current_dir / 'data' / 'icosphere.obj'
        self.load_mesh_from_file(str(mesh_file))

    def resizeGL(self, width, height):
        self.ctx.viewport = (0, 0, width, height)
        self.width = width
        self.height = height

    def paintGL(self):
        self.update_matrices()
        self.ctx.clear(0, 0, 0, 1)  # clear the whole drawing surface
        self.ctx.enable(mgl.DEPTH_TEST)  # always use depth test!
        self.ctx.enable(mgl.BLEND)
        # draw the mesh at the current LOD
        # no culling for triangles (helps to see backside if using boundaries (or for bugs))
        self.ctx.disable(mgl.CULL_FACE)
        if self.mesh_wireframe:
            self.ctx.wireframe = self.mesh_wireframe
            self.prog_tris['u_use_lighting'].value = False
        self.prog_tris['u_color'].value = (0.8, 0.7, 0.6, 1 - 0.5 * self.mesh_wireframe)  # set the color uniform
        if len(self.face_objs) == 0:
            self.vao.render()  # HEDS not built yet, just draw everything!
        else:
            self.vao.render( vertices = 3 * (len(self.face_objs) - 2*self.current_LOD) )# skip two faces for each LOD    
        self.ctx.wireframe = False
        self.prog_tris['u_use_lighting'].value = True
        # as selected in the UI, draw half edge, vertex IDs, face IDs
        if self.draw_current_he and self.current_LOD == self.max_LOD:
            self.ctx.enable(mgl.CULL_FACE)  # ray traced cylinders need backface culling
            self.ctx.cull_face = 'back'
            self.he_vao.render()  # draw the half-edge
            self.ctx.disable(mgl.CULL_FACE)  # ray traced cylinders need backface culling
        if self.draw_vertex_IDs:
            # New verts always added to end, so go to end of list if at highest LOD, otherwise stop before new verts
            k = self.max_LOD - self.current_LOD
            for v in self.vert_objs if k == 0 else self.vert_objs[:-k]:
                if v.removed_at_level is not None and v.removed_at_level < self.current_LOD:
                    continue
                v.draw_debug(self.P, self.V, self.text_renderer)
        if self.draw_face_IDs:
            for i in range( len(self.face_objs) - 2*self.current_LOD ):
                self.face_objs[i].draw_debug(self.P, self.V, self.faces, self.vert_objs, self.text_renderer)

    def get_vertex_count(self):
        count = 0
        k = self.max_LOD - self.current_LOD
        for v in self.vert_objs if k == 0 else self.vert_objs[:-k]:
            if v.removed_at_level is not None and v.removed_at_level < self.current_LOD:
                continue
            count += 1
        return count

    def get_face_count(self):
        return len(self.face_objs) - 2 * self.current_LOD

    def setup_mesh_shaders_and_buffers(self, current_dir):
        self.prog_tris = self.ctx.program(
            vertex_shader=open(current_dir / 'glsl' / 'tris-vert.glsl').read(),
            geometry_shader=open(current_dir / 'glsl' / 'tris-geom.glsl').read(),
            fragment_shader=open(current_dir / 'glsl' / 'tris-frag.glsl').read())
        self.prog_tris['u_color'].value = (0.8, 0.7, 0.6, 1.0)  # set the color uniform
        self.prog_tris['u_use_lighting'].value = True
        self.prog_tris['u_light_pos'].value = (15, 15, 15)  # light position in view coordinates

    def setup_half_edge_shaders_and_buffers(self, current_dir):
        self.prog_lines = self.ctx.program(
            vertex_shader=open(current_dir / 'glsl' / 'fancyline-vert.glsl').read(),
            geometry_shader=open(current_dir / 'glsl' / 'fancyline-geom.glsl').read(),
            fragment_shader=open(current_dir / 'glsl' / 'fancyline-frag.glsl').read())
        self.prog_lines['u_color'].value = (1.0, 0.0, 0.0, 1.0)
        self.prog_lines['u_light_pos'].value = (15, 15, 15)  # light position in view coordinates
        self.prog_lines['u_radius'].value = 0.02  # radius of the line in world coordinates

        # set up a half edge... and perhaps a geometry shader for it to draw lines as cylinders?
        self.he_geom = np.zeros((3, 3), dtype='f4')  # 3 vertices to form a half edge
        self.he_vbo = self.ctx.buffer(self.he_geom.tobytes())
        self.he_vao = self.ctx.vertex_array(self.prog_lines, [(self.he_vbo, '3f', 'in_position')], mode=mgl.LINE_STRIP)

    def load_mesh_from_file(self, filename: str):
        mesh = trimesh.load_mesh(filename)
        self.faces = mesh.faces  # this is easy because we have only triangles!
        self.indices = mesh.faces.flatten()
        self.verts = mesh.vertices  # shape is (num_verts, 3)
        # center and scale the mesh to fit in a -1 to 1 cube
        centroid = np.mean(self.verts, axis=0)
        self.verts -= centroid
        scale = np.max(np.linalg.norm(self.verts, axis=1))
        if scale > 1e-6:
            self.verts /= scale
        # Make vertex objects, initially without half edges,
        # 	and only for the current vertices (this list will grow as we collapse edges)
        self.vert_objs = [Vertex(i, pos, None) for i, pos in enumerate(self.verts)]
        # build the half edge data structure
        half_edges, self.face_objs = build_heds(self.faces, self.vert_objs)

        # NOTE: reserve an appropriate amount of space in the vertex buffer
        # Every collapse removes one vertex, 2 faces, and 3 edges.
        # won't go past a tetrahedron, so expect half of num faces - 4 
        # num_collapses_expected = (len(self.face_objs) - 4) // 2  # each collapse removes 2 faces
        num_collapses_expected = (self.faces.shape[0] - 4) // 2  # each collapse removes 2 faces
        self.verts = np.vstack((self.verts, np.zeros((num_collapses_expected, 3), dtype='f4')))

        self.vbo = self.ctx.buffer(self.verts.astype('f4').tobytes())
        self.ibo = self.ctx.buffer(self.faces.flatten().astype('i4').tobytes())
        self.vao = self.ctx.vertex_array(self.prog_tris, [(self.vbo, '3f', 'in_position')], self.ibo, mode=mgl.TRIANGLES)

        # once the heds is initialized, compute the quadrics for each vertex, and the edge collapse costs
        if half_edges:
            for v in self.vert_objs:
                v.compute_Q()
                v.compute_debug_viz_data()  # while we're at it, collect some debug viz data

            self.compute_edge_collapse_costs(half_edges)

            self.current_he = half_edges[0]
            self.update_half_edge_geometry()

        self.current_LOD = 0
        self.max_LOD = 0  # the number of simplification levels available
        self.collapse_history = []  # keep track of collapse records for undoing
        self.update_UI_callback()  # updates the LOD slider

    def update_half_edge_geometry(self):
        v0 = self.current_he.next.next.head.pos
        v1 = self.current_he.head.pos
        v2 = self.current_he.next.head.pos
        center = self.current_he.face.get_center()
        normal = self.current_he.face.get_normal() * glm.length(center - v0) * 0.1
        self.he_geom[0] = normal + 0.9 * v0 + 0.1 * center
        self.he_geom[1] = normal + 0.9 * v1 + 0.1 * center
        self.he_geom[2] = normal + 0.9 * (0.8 * v1 + 0.2 * v2) + 0.1 * center
        self.he_vbo.write(self.he_geom.tobytes())

    def update_matrices(self):
        self.P = glm.perspective(glm.radians(45.0), self.width / self.height, 0.1, 100.0)
        self.S = glm.mat4(1)
        if self.scale_with_LOD:
            max_lod = len(self.face_objs) // 2
            min_scale = 0.01  # 1% at maximum LOD
            # Solve: min_scale = base^max_lod  =>  base = min_scale^(1/max_lod)
            base = min_scale ** (1.0 / max_lod)
            scale_factor = base ** self.current_LOD
            self.S = glm.scale(glm.mat4(1), glm.vec3(scale_factor, scale_factor, scale_factor))
        self.V = glm.lookAt(glm.vec3(0, 0, self.d), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0)) * self.R * self.S        
        self.prog_tris['u_mv'].write(self.V)
        self.prog_tris['u_mvp'].write(self.P * self.V)
        self.prog_lines['u_view'].write(self.V)
        self.prog_lines['u_proj'].write(self.P)

    def mousePressEvent(self, event):
        self.last_mouse_pos = (event.x(), event.y())

    def mouseMoveEvent(self, event):
        new_x, new_y = event.x(), event.y()
        rx = glm.rotate(glm.mat4(1), (new_y - self.last_mouse_pos[1]) * 0.002, glm.vec3(1, 0, 0))
        ry = glm.rotate(glm.mat4(1), (new_x - self.last_mouse_pos[0]) * 0.002, glm.vec3(0, 1, 0))
        self.R = ry * rx * self.R
        self.last_mouse_pos = (new_x, new_y)

    def wheelEvent(self, event):
        self.d *= np.power(1.1, event.angleDelta().y() / 120)

    def next_half_edge(self):
        if self.current_he.twin is None:
            return
        self.current_he = self.current_he.next
        self.update_half_edge_geometry()

    def twin_half_edge(self):
        if self.current_he.twin is None:
            return
        self.current_he = self.current_he.twin
        self.update_half_edge_geometry()

    def collapse_current_half_edge(self):
        if len(self.sorted_edge_list) <= 6:  # stop when we reach a tetrahedron
            print("No more edges to collapse")
            return
        # discard current half edge form queue if it is there (note that this is different from remove)
        self.sorted_edge_list.discard(self.current_he.edge_collapse_data)
        vtail = self.current_he.tail()
        vhead = self.current_he.head
        vstar = (vtail.pos + vhead.pos) / 2
        self.current_he = self.collapse(self.current_he, vstar)
        self.update_half_edge_geometry()

    def jump_to_best_edge(self):
        if len(self.sorted_edge_list) <= 6:  # stop when we reach a tetrahedron
            print("No more edges to collapse")
            return

        edge = self.sorted_edge_list[0]
        self.current_he = edge.he
        self.update_half_edge_geometry()

    

    def collapse_best_edge(self):
        if len(self.sorted_edge_list) <= 6:  # stop when we reach a tetrahedron
            print("No more edges to collapse")
            return
        edge = self.sorted_edge_list.pop(0)  # get and remove the best edge
        self.current_he = self.collapse(edge.he, edge.v_opt, edge.cost)
        self.update_half_edge_geometry()

    def collapse_all_in_order(self):
        # collapse all the edges in order, best edge each time,
        # 	until we get to a tetrahedron (or otherwise simplest possible mesh)
        while not len(self.sorted_edge_list) <= 6:
            edge = self.sorted_edge_list.pop(0)  # get and remove the best edge
            self.current_he = self.collapse(edge.he, edge.v_opt, edge.cost)
        self.update_half_edge_geometry()

    def set_LOD(self, level: int):
        """ set the current LOD to the given level, updating the mesh faces data accordingly """
        if level < 0 or level > self.max_LOD:
            print("Invalid LOD level:", level)
            return
        while self.current_LOD < level:
            self.collapse_history[self.current_LOD].redo(self.faces)
            self.current_LOD += 1
        while self.current_LOD > level:
            self.current_LOD -= 1
            self.collapse_history[self.current_LOD].undo(self.faces)
        self.ibo.write(self.faces.flatten().astype('i4').tobytes())

    def compute_edge_collapse_costs(self, half_edges: list[HalfEdge]):
        """ compute the cost of collapsing each edge, and the optimal position for the collapse
            Store these in a sorted list for O(log(n)) addition and removal, and O(1) query of best edge.
            This is ONLY called on the first initialization, when we have a list of all half edges! """
        self.sorted_edge_list = SortedList()
        for he in half_edges:
            if he.edge_collapse_data is not None:
                continue  # already computed
            edge = EdgeCollapseData(he)
            self.sorted_edge_list.add(edge)

    def collapse_will_be_bad(self, he: HalfEdge) -> bool:
        """ check if collapsing this half-edge will create problems (e.g., more than 2 common verts in the 1-rings) """

        # TODO: Objective 4: Check if collapsing this half-edge will create problems

        v1 = he.tail()
        v2 = he.head

        if he.face is None or he.twin is None or he.twin.face is None:
            return False

        # collect all 1-ring neighbours around a vertex
        def get_one_ring(vertex):
            neighbors = set()
            start = vertex.he
            if start is None:
                return neighbors
            h = start

            while True:
                tail_vertex = h.tail()
                if tail_vertex != vertex:
                    neighbors.add(tail_vertex)

                # move around the vertex via twin-next
                if h.twin is None or h.twin.next is None:
                    break
                h = h.twin.next

                if h == start:
                    break
            
            return neighbors

        ring1 = get_one_ring(v1)
        ring2 = get_one_ring(v2)

        ring1.discard(v1)
        ring1.discard(v2)
        ring2.discard(v1)
        ring2.discard(v2)

        common = ring1.intersection(ring2)

        # check if the 1-rings of the edge vertices have more than 2 vertices in common
        if len(common) > 2:
            print(f"[BAD] Edge ({v1.index}, {v2.index}) has {len(common)} common neighbors → illegal collapse")
            return True

        return False

    def collapse(self, he: HalfEdge, vstar: glm.vec3, cost: float = 0.0) -> HalfEdge:
        """ collapse the given half-edge.
            If the collapse will not be bad, collapse the edge by:
            - Making a new Vertex and add to the end of the buffer (and updating the vbo accordingly)
            - Moving the collapsed faces to the end of the face_obj list (and updating the faces numpy array accordingly)
            - updating the half edge data structure TODO
            - updating the computed edge collapse costs for affected half-edges TODO
            - updating the collapse records  TODO

            Args:
                he: the hald-edge to collapse
                vstar: the position of the new vertex
                cost: cost of the new vertex
            Returns:
                 One of the half-edges with the new vertex as head.
        """

        # ensure we are at the coarest LOD
        self.set_LOD(self.max_LOD)

        if self.collapse_will_be_bad(he):
            print("Bad collapse detected, skipping")
            return he

        # flag old verts as removed at this LOD level (for debug viz)
        he.head.removed_at_level = self.max_LOD
        he.tail().removed_at_level = self.max_LOD

        # Create the new vertex at END OF THE LIST.
        # Set index to be next in the list, and give a half edge that will exist after the collapse
        new_vertex = Vertex(len(self.vert_objs), vstar, he.next.twin)
        new_vertex.cost = cost
        self.vert_objs.append(new_vertex)
        new_vertex.Q = he.head.Q + he.tail().Q
        # put it into the verts array, and into the vertex buffer object
        self.verts[new_vertex.index] = np.array((vstar.x, vstar.y, vstar.z), dtype='f4')
        self.vbo.write(vstar.to_bytes(), offset=3 * new_vertex.index * 4)


        # TODO: Objective 2: Collapse the given half-edge.  See notes from class!
        # NOTE: removed faces must be swapped with those at the end of face_objs list!
        # see also that you migth want to complete objective 3 to see the results as
        # the index buffer needs to be updated to correctly draw the mesh after the 
        # collapse, and that is done in objective 3 below.
        current_edges = list(self.sorted_edge_list)
        v_tail = he.tail()
        v_head = he.head
        f_left = he.face
        f_right = he.twin.face

        affected_faces = []
        old_faces_data = []
        for f_idx, face_indices in enumerate(self.faces):
            if v_tail.index in face_indices or v_head.index in face_indices:
                affected_faces.append(self.face_objs[f_idx])
                old_faces_data.append(face_indices.copy())
        old_faces = np.array(old_faces_data)

        # halfedges around the collapsing pair
        h0 = he.next.twin
        h1 = he.next.next.twin
        h2 = he.twin.next.twin
        h3 = he.twin.next.next.twin

        # reconnect twins between remaining faces
        if h0 and h1:
            h0.twin = h1
            h1.twin = h0
        if h2 and h3:
            h2.twin = h3
            h3.twin = h2

        vl = he.next.head
        vr = he.twin.next.head
        if h1:
            vl.he = h1
        if h3:
            vr.he = h3
        if h0:
            new_vertex.he = h0

        # update halfedges
        h = h0
        visited = set()
        while h and h not in visited:
            visited.add(h)
            h.head = new_vertex
            if not h.next or not h.next.twin:
                break
            h = h.next.twin
            if h == h0:
                break
        
        # move removed faces to the end of face list
        removed_faces = [f_left, f_right]
        for i, f in enumerate(removed_faces):
            idx = f.index
            last_idx = len(self.face_objs) - 1 - 2 * self.current_LOD - i
            if idx != last_idx:
                # swap face_objs
                self.face_objs[idx], self.face_objs[last_idx] = self.face_objs[last_idx], self.face_objs[idx]
                self.face_objs[idx].index = idx
                self.face_objs[last_idx].index = last_idx
                # swap face indices
                self.faces[[idx, last_idx]] = self.faces[[last_idx, idx]]

        # update all face vertex indices
        for face_obj in self.face_objs[:-2]:
            face = self.faces[face_obj.index]
            for j in range(3):
                if face[j] == v_tail.index or face[j] == v_head.index:
                    face[j] = new_vertex.index


        # TODO: Objective 6: Maintain the sorted list of edge collapse costs
        # After collapsing an edge, the quadric error metrics of the new vertex is set,
        # so we need to do some careful work to make sure the sorted list is updated to
        # have updated edge collapse data for the half-edges around the new vertex.
        # (best to remove all, and then re-add newly computed versions)
        # remove deleted halfedges

        edges_to_remove = []
        if he.edge_collapse_data:
            edges_to_remove.append(he.edge_collapse_data)
        if he.twin and he.twin.edge_collapse_data:
            edges_to_remove.append(he.twin.edge_collapse_data)

        def collect_adjacent_edges(vertex):
            if not vertex or not vertex.he:
                return
            start = vertex.he
            h = start
            visited = set()
            while h and h not in visited:
                visited.add(h)
                if h.edge_collapse_data:
                    edges_to_remove.append(h.edge_collapse_data)
                if h.twin and h.twin.edge_collapse_data:
                    edges_to_remove.append(h.twin.edge_collapse_data)
                if not h.twin or not h.twin.next:
                    break
                h = h.twin.next
                if h == start:
                    break

        collect_adjacent_edges(v_tail)
        collect_adjacent_edges(v_head)
        collect_adjacent_edges(new_vertex)

        # remove adjacent edges
        for edge_data in list(edges_to_remove):
            if edge_data in self.sorted_edge_list:
                self.sorted_edge_list.discard(edge_data)
            if edge_data.he is not None:
                h = edge_data.he
                t = h.twin
                h.edge_collapse_data = None
                if t:
                    t.edge_collapse_data = None
                if not h.face or (t and not t.face):
                    h.face = None
                    if t:
                        t.face = None

        def get_one_ring(vertex):
            neighbors = set()
            if not vertex.he:
                return neighbors
            start = vertex.he
            h = start
            visited = set()
            while h and h not in visited:
                visited.add(h)
                if h.head != vertex:
                    neighbors.add(h.head)
                if not h.twin or not h.twin.next:
                    break
                h = h.twin.next
                if h == start:
                    break
            return neighbors

        def recompute_edges(vertex):
            if not vertex or not vertex.he:
                return
            start = vertex.he
            h = start
            visited = set()
            count = 0

            while h and h not in visited:
                visited.add(h)
                count += 1
                if count > 50:
                    print(f"[WARN] Infinite loop detected around vertex {vertex.index}")
                    break
                if h.face and h.twin and h.twin.face:
                    tail = h.tail()
                    head = h.head
                    if tail is None or head is None:
                        h = h.twin.next if h.twin else None
                        continue

                    if tail.index < head.index:
                        edge_data = EdgeCollapseData(h)
                        h.edge_collapse_data = edge_data
                        h.twin.edge_collapse_data = edge_data
                        self.sorted_edge_list.add(edge_data)
                if not h.twin or not h.twin.next:
                    break
                h = h.twin.next
                if h == start:
                    break

        recompute_edges(new_vertex)
        for nb in get_one_ring(new_vertex):
            recompute_edges(nb)
        
        clean_list = SortedList()
        for e in self.sorted_edge_list:
            he = e.he
            tail = he.tail() if he else None
            if (
                he and he.head and tail
                and he.head != tail
                and he.head.removed_at_level is None
                and tail.removed_at_level is None
                and he.face and he.twin and he.twin.face
                and he.head.index < len(self.vert_objs)
                and tail.index < len(self.vert_objs)
            ):
                clean_list.add(e)
        self.sorted_edge_list = clean_list

        # remove duplicate edges
        seen = set()
        unique_list = []

        for edge in self.sorted_edge_list:
            he = edge.he
            if not he or not he.head or not he.tail():
                continue
            t_idx = he.tail().index
            h_idx = he.head.index
            key = (min(t_idx, h_idx), max(t_idx, h_idx), round(edge.cost, 8))
            if key not in seen:
                seen.add(key)
                unique_list.append(edge)

        self.sorted_edge_list = SortedList(unique_list)

        #print("Sorted list size:", len(self.sorted_edge_list))
        #print("Min cost edge:", self.sorted_edge_list[0].cost if len(self.sorted_edge_list) > 0 else None)
        

        # TODO: Objective 3: Undo / Redo by making and collecting collapse records 

        # TODO: You need to fill in the correct data for the CollapseRecord here
        #affected_faces = [] # list of N Face objects affected by this collapse (typically around 8 faces)
        #old_faces = np.array([]) # Nx3 array of old face indices before the collapse
        #new_faces = np.array([]) # Nx3 array of new face indices after the collapse
        new_faces_data = []
        for face_obj in affected_faces:
            new_faces_data.append(self.faces[face_obj.index].copy())
        
        old_faces = np.array(old_faces_data)
        new_faces = np.array(new_faces_data)

        # collapse record tells us how to move to the next LOD, or likewise, how to undo
        cr = CollapseRecord(affected_faces, old_faces, new_faces)
        self.collapse_history.append(cr)
        self.current_LOD += 1
        self.max_LOD += 1

        cr.redo(self.faces) # THIS APPLIES THE COLLAPSE TO THE FACES numpy array for drawing with opengl
        self.ibo.write(self.faces.flatten().astype('i4').tobytes()) # update the index buffer object

        # with everything all hooked up, compute the debug viz data for the new vertex
        new_vertex.compute_debug_viz_data()

        # notify the UI that LOD has changed
        self.update_UI_callback()
        return new_vertex.he  # return one of the half-edges with the new vertex as head
