import bpy
import blender_plots as bplt
from blender_plots import blender_utils as bu
import trimesh
import mathutils
from pathlib import Path
import numpy as np
import math

def add_text(text, location, color=None, name=''):
    font_curve = bpy.data.curves.new(type="FONT", name=text)
    font_curve.body = text
    font_obj = bu.new_empty(f'text_{name}', font_curve)
    color_mat = bpy.data.materials.new("Text color")
    color_mat.diffuse_color = (0., 0, 0, 0) if color is None else color
    font_obj.data.materials.append(color_mat)

    # bpy.context.scene.collection.objects.link(font_obj)
    font_obj.location = location
    font_obj.scale = [0.2, 0.2, 0.2]

    modifier = font_obj.modifiers.new(type="SOLIDIFY", name="solidify")
    modifier.thickness = 0.05
    return font_obj

def add_mesh(mesh: trimesh.Trimesh, color, name="mesh", center=False, offset=None, scale=1.):
    if offset is None:
        offset = np.array([0, 0, 0])

    mesh = mesh.copy() # make sure not to modify original mesh
    mesh.vertices *= scale
    if center:
        mesh.vertices -= mesh.vertices.mean(axis=0)
    # if offset is not None:
    #     mesh.vertices += offset
    bmesh = bpy.data.meshes.new(name=name)
    bmesh.from_pydata(mesh.vertices, [], mesh.faces)
    mesh_object = bu.new_empty(name, bmesh)
    mesh_object.location = offset

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = color
    mesh_object.data.materials.append(material)
    return mesh_object

def plot_mesh(mesh_path, color, offset=None, scale=1., name="mesh", center=False, train_points=False, text='', rotation=None, with_text=True, text_color=None):
    if offset is None:
        offset = np.zeros(3)
    if rotation is None:
        rotation = (0, 0, 0)
    if isinstance(mesh_path, trimesh.Trimesh):
        gt_mesh = mesh_path
    else:
        gt_mesh = trimesh.load(mesh_path)
    if hasattr(gt_mesh, 'vertices'):
        mesh_object = add_mesh(gt_mesh, color, name, center, offset, scale)
        mesh_object.rotation_euler = rotation
    else:
        mesh_object = None
    # gt_mesh_object.color = (1, 0, 0, 1)

    if with_text:
        font_obj = add_text(text, offset + np.array([-0., -0.8, -0.62]), name=name, color=text_color)
    if train_points:
        train_points = np.load(Path(mesh_path).parent.parent / 'train_points.npy')
        scatter = bplt.Scatter(train_points + offset, marker_type='ico_spheres', radius=0.01, subdivisions=2, color=[0, 0, 0, 0.1], name=f"train points {name}")
    else:
        scatter = None
    return mesh_object, scatter

def bounding_box(lower, upper, name, rotation=None, offset=None):
    x1, y1, z1 = lower
    x2, y2, z2 = upper

    vertices = [
        (x1, y1, z1),  # Vertex 0
        (x2, y1, z1),  # Vertex 1
        (x2, y2, z1),  # Vertex 2
        (x1, y2, z1),  # Vertex 3
        (x1, y1, z2),  # Vertex 4
        (x2, y1, z2),  # Vertex 5
        (x2, y2, z2),  # Vertex 6
        (x1, y2, z2)   # Vertex 7
    ]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting faces
    ]

    mesh = bpy.data.meshes.new(name)
    obj = bu.new_empty(name, mesh)
    mesh.from_pydata(vertices, edges, [])
    if rotation is not None:
        obj.rotation_euler = rotation
    if offset is not None:
        obj.location = offset
    return obj

def create_camera(location, rotation):
    if "Camera" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Camera"])

    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=location, rotation=rotation)
    bpy.context.scene.camera = bpy.data.objects['Camera']

def render_image(output_path, resolution, samples=100):
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.cycles.samples = samples
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

def setup_scene(clear=False, camera_location=None, camera_rotation=None, resolution=None):
    if "Cube" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Cube"])

    if camera_location is None:
        camera_location = np.array([0, -5.321560, 2.042498]) * 0.6
    if camera_rotation is None:
        camera_rotation = [math.radians(68.4), 0., 0.]

    if clear:
        bpy.ops.wm.read_homefile()
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.data.scenes["Scene"].cycles.samples = 256

    if "Sun" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Sun"])
    bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.data.objects["Sun"].data.energy = 30.
    bpy.data.objects["Sun"].data.angle = np.pi / 2
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs["Strength"].default_value = 0.5

    bpy.context.scene.render.film_transparent = True
    create_camera(camera_location, camera_rotation)
    if resolution is not None:
        bpy.context.scene.render.resolution_x = resolution[0]
        bpy.context.scene.render.resolution_y = resolution[1]

def euler_to_R(euler):
    return np.array(mathutils.Euler(euler).to_matrix())
