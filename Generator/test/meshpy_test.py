import os
import copy
import numpy as np
import open3d as o3d

from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile

home_dir = os.environ['HOME']
file_dir = home_dir + "/Projects/GPD_PointNet/extract_normals"

of = ObjFile(file_dir + "/mesh_ascii_with_normal_crop.obj")
obj = of.read()
center_of_mass = obj.center_of_mass

print(center_of_mass)

cloud = o3d.io.read_point_cloud(file_dir + "/mesh_ascii_with_normal.ply")
mesh = o3d.io.read_triangle_mesh(file_dir + "/mesh_ascii_with_normal.ply")

mesh_sphere = o3d.geometry.create_mesh_sphere(radius=0.005)
mesh_sphere.compute_vertex_normals()
mesh_sphere.paint_uniform_color([1.0, 0.0, 0.0])

transformation = np.identity(4)
transformation[:3, 3] = center_of_mass
mesh_sphere.transform(transformation)

o3d.draw_geometries([cloud, mesh_sphere])


print(np.asarray(mesh.triangles)[0])
mesh2 = copy.deepcopy(mesh)
mesh2.triangles = o3d.utility.Vector3iVector(
        np.asarray(mesh.triangles)[:len(mesh.triangles) // 2, :])
mesh2.compute_vertex_normals()

o3d.draw_geometries([mesh2])
