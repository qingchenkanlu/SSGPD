# -*- coding: utf-8 -*-

import pcl
import time
import numpy as np
import open3d as o3d


def main():
    cloud = pcl.load('./cloud.pcd')
    normals = pcl.load('./normals_as_xyz.pcd')  # normals were saved as PointXYZ formate
    print('Loaded ' + str(cloud.width * cloud.height) +
          ' data points from cloud.pcd with the following fields: ')

    start = time.perf_counter()
    kd_tree = cloud.make_kdtree_flann()
    point = pcl.PointCloud(cloud.to_array()[1500].reshape(1, 3))
    kd_indices, sqr_distances = kd_tree.radius_search_for_cloud(point, 0.005, 27)
    print("pcl take: ", time.perf_counter()-start)

    cloud = cloud.to_array()
    normals = normals.to_array()

    print("cloud[0]", cloud[0])
    print("normals[0]", normals[0])

    if np.isnan(np.sum(normals[0])):
        print("[DEBUG] nan in normal")

    # for i in range(0, cloud.size):
    #     print('x: ' + str(cloud[i][0]) + ', y : ' +
    #           str(cloud[i][1]) + ', z : ' + str(cloud[i][2]))

    """ use open3d """
    pcd = o3d.io.read_point_cloud("./cloud.pcd")
    print(pcd.points[0])

    start = time.perf_counter()

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    print("Find its neighbors with distance less than 0.2, paint green.")
    [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[1500], 0.005)
    print("o3d take: ", time.perf_counter() - start)

    print(k)
    print(idx)

    print("Visualize the point cloud.")
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', sort='time')
    main()
