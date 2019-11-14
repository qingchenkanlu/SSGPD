#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;

//#define SHOW


pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr loadPointCloud(const std::string &filename) {
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR); // ignore some warning

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    std::string extension = filename.substr(filename.size() - 3);

    if (extension == "pcd" &&
        pcl::io::loadPCDFile<pcl::PointXYZRGBNormal>(filename, *cloud) == -1) {
        printf("Couldn't read PCD file: %s\n", filename.c_str());
        cloud->points.resize(0);
    } else if (extension == "ply" &&
               pcl::io::loadPLYFile<pcl::PointXYZRGBNormal>(filename, *cloud) == -1) {
        printf("Couldn't read PLY file: %s\n", filename.c_str());
        cloud->points.resize(0);
    }

    return cloud;
}

void save_normals(const string& filename, const pcl::PointCloud<pcl::Normal>::Ptr& normals_out) {
    /// save mesh_normals use PointXYZ formate

    std::ofstream myfile;
    myfile.open(filename);

    myfile << "# .PCD v0.7 - Point Cloud Data file format" << "\n";
    myfile << "VERSION 0.7" << "\n" << "FIELDS x y z" << "\n";
    myfile << "SIZE 4 4 4" << "\n";
    myfile << "TYPE F F F" << "\n" << "COUNT 1 1 1"<< "\n";
    myfile << "WIDTH " << std::to_string(normals_out->width) << "\n";
    myfile << "HEIGHT " << std::to_string(normals_out->height) << "\n";
    myfile << "VIEWPOINT 0 0 0 1 0 0 0" << "\n";
    myfile << "POINTS " << std::to_string(normals_out->size()) << "\n";
    myfile << "DATA ascii" << "\n";


    for (int i=0; i < normals_out->size(); i++) {
        const float &normal_x = normals_out->points[i].normal_x;
        const float &normal_y = normals_out->points[i].normal_y;
        const float &normal_z = normals_out->points[i].normal_z;

        myfile << boost::lexical_cast<std::string>(normal_x) << " "
               << boost::lexical_cast<std::string>(normal_y) << " "
               << boost::lexical_cast<std::string>(normal_z) << "\n";
    }
    myfile.close();
}

int main(int argc, char * argv[]) {
    double t_ = omp_get_wtime();

    string data_path;
    if (argc > 1) {
        data_path = argv[1];
        printf("[INFO] Data path: %s\n", argv[1]);
    } else {
        data_path = "..";
    }

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr surface_cloud_no_normal (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normal (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    surface_cloud_no_normal = loadPointCloud(data_path + "/surface_cloud.ply");
    cloud_with_normal = loadPointCloud(data_path + "/mesh_ascii_with_normal.ply");

    // 滤除工作空间外点云
    std::vector<float> workspace = {0.0, 2.0, -0.4, 0.4, -0.1, 1.0};
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_workspace_filter(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    for (int i = 0; i < surface_cloud_no_normal->size(); i++) {
        const pcl::PointXYZRGBNormal &p = surface_cloud_no_normal->points[i];
        if (p.x > workspace[0] && p.x < workspace[1] && p.y > workspace[2] &&
            p.y < workspace[3] && p.z > workspace[4] && p.z < workspace[5]) {
            cloud_workspace_filter->push_back(p);
        }
    }

    // 体素栅格降采样
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_workspace_voxel(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::VoxelGrid<pcl::PointXYZRGBNormal> sor;
    sor.setInputCloud (cloud_workspace_filter);
    sor.setLeafSize (0.005f, 0.005f, 0.005f);
    sor.filter (*cloud_workspace_voxel);

    // 滤除桌面点云
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_plane_filter(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    for (int i = 0; i < cloud_workspace_filter->size(); i++) {
        const pcl::PointXYZRGBNormal &p = cloud_workspace_filter->points[i];
        if (p.z > 0.0) cloud_plane_filter->push_back(p);
    }

    // 创建表面有法线点云指针
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr surface_cloud_with_normal (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::copyPointCloud(*cloud_plane_filter,*surface_cloud_with_normal); // 获取点云

    // kdtree对象
    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> kdtree;
    // 输入点云
    kdtree.setInputCloud (cloud_with_normal);

    /**********************************************************************************
                                                K近邻
       ********************************************************************************/

//    // K 个最近点去搜索 nearest neighbor search
//    int K = 1;
//    // 两个向量来存储搜索到的K近邻，两个向量中，一个存储搜索到查询点近邻的索引，另一个存储对应近邻的距离平方
//    std::vector<int> pointIdxNKNSearch(K);//最近临搜索得到的索引
//    std::vector<float> pointNKNSquaredDistance(K);//平方距离
//
//    for(int i = 0; i < surface_cloud_with_normal->size(); ++i)
////    for(int i = 0; i < 3; ++i)
//    {
//        pcl::PointXYZRGBNormal *itSP = &surface_cloud_with_normal->points[i];
//
//        std::cout << "K nearest neighbor search at (" << itSP->x
//              << " " << itSP->y
//              << " " << itSP->z
//              << ") with K=" << K << std::endl;
//
//        if ( kdtree.nearestKSearch (*itSP, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
//        {
//            for (size_t j = 0; j < pointIdxNKNSearch.size (); ++j)
//                std::cout << " " << cloud_with_normal->points[ pointIdxNKNSearch[j] ].x
//                          << " " << cloud_with_normal->points[ pointIdxNKNSearch[j] ].y
//                          << " " << cloud_with_normal->points[ pointIdxNKNSearch[j] ].z
//                          << " (squared distance: "
//                          << pointNKNSquaredDistance[j]
//                          << ")"
//                          << std::endl;
//            itSP->normal_x = cloud_with_normal->points[ pointIdxNKNSearch[0] ].normal_x;
//            itSP->normal_y = cloud_with_normal->points[ pointIdxNKNSearch[0] ].normal_y;
//            itSP->normal_z = cloud_with_normal->points[ pointIdxNKNSearch[0] ].normal_z;
//        }
//    }

    /**********************************************************************************
                                        指定半径内近邻
       ********************************************************************************/

    // 半径内最近领搜索 Neighbors within radius search
    std::vector<int> pointIdxRadiusSearch; //存储查询点近邻索引
    std::vector<float> pointRadiusSquaredDistance; //存储近邻点对应距离平方
    float radius = 0.005; //搜索半径

    for(int i = 0; i < surface_cloud_with_normal->size(); ++i) {
        pcl::PointXYZRGBNormal *itSP = &surface_cloud_with_normal->points[i];
        //打印相关信息
//        std::cout << "Neighbors within radius search at (" << itSP->x
//                  << " " << itSP->y
//                  << " " << itSP->z
//                  << ") with radius="
//                  << radius << std::endl;
        // 开始搜索
        if (kdtree.radiusSearch(*itSP, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
            // 获取最近点的法线
            surface_cloud_with_normal->points[i].normal_x = cloud_with_normal->points[pointIdxRadiusSearch[0]].normal_x;
            surface_cloud_with_normal->points[i].normal_y = cloud_with_normal->points[pointIdxRadiusSearch[0]].normal_y;
            surface_cloud_with_normal->points[i].normal_z = cloud_with_normal->points[pointIdxRadiusSearch[0]].normal_z;
            // 仅取最近点
//            std::cout << " " << cloud_with_normal->points[pointIdxRadiusSearch[0]].x
//                      << " " << cloud_with_normal->points[pointIdxRadiusSearch[0]].y
//                      << " " << cloud_with_normal->points[pointIdxRadiusSearch[0]].z
//                      << " (squared distance: " << pointRadiusSquaredDistance[0] << ")" << std::endl;
        }
    }

    // remove nan
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*surface_cloud_with_normal, *surface_cloud_with_normal, indices);
    pcl::removeNaNNormalsFromPointCloud(*surface_cloud_with_normal, *surface_cloud_with_normal, indices);

    // 获取处理后的点云和法线
    pcl::PointCloud<pcl::Normal>::Ptr surface_normals (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr surface_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::copyPointCloud(*surface_cloud_with_normal, *surface_normals);
    pcl::copyPointCloud(*surface_cloud_with_normal, *surface_cloud);

    // 保存处理后的点云及法线
    pcl::PCDWriter writer;
    writer.writeASCII("../cloud.pcd", *cloud_plane_filter); // 保存桌面以上，计算过法线的点云
    writer.writeASCII("../cloud_voxel.pcd", *cloud_workspace_voxel); // 保存未滤除桌面，降采样之后的点云
    save_normals("../normals_as_xyz.pcd", surface_normals); // 以PointXYZ格式保存法线

    // 打印法线处理耗时
    t_ = omp_get_wtime() - t_;
    printf("Process surface normals in %3.4fs.\n", t_);

#ifdef SHOW
    /// 显示处理后的表面点云及法线
    pcl::visualization::PCLVisualizer viewer("surface_cloud_with_normal");
//    viewer.setBackgroundColor(0, 0, 0);//背景黑色
    viewer.setBackgroundColor (1, 1, 1);//白色
    viewer.addCoordinateSystem (0.1f, "global");//坐标系
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBNormal> cloud_color_handler (surface_cloud, 0, 0, 255);//红色
    viewer.addPointCloud<pcl::PointXYZRGBNormal>(surface_cloud, cloud_color_handler, "original point cloud");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "original point cloud");

    viewer.addPointCloudNormals<pcl::PointXYZRGBNormal, pcl::Normal>(surface_cloud, surface_normals, 1, 0.02, "surface_cloud_with_normal normals");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "surface_cloud_with_normal normals");
    viewer.initCameraParameters();


    /// 显示mesh点云及法线
    pcl::PointCloud<pcl::Normal>::Ptr mesh_normals (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr mesh_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::copyPointCloud(*cloud_with_normal, *mesh_normals);
    pcl::copyPointCloud(*cloud_with_normal, *mesh_cloud);

    pcl::visualization::PCLVisualizer viewer1("cloud_with_normal");
//    viewer1.setBackgroundColor(0, 0, 0);//背景黑色
    viewer1.setBackgroundColor (1, 1, 1);//白色
    viewer1.addCoordinateSystem (0.1f, "global");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBNormal> cloud_color_handler1 (cloud_with_normal, 0, 0, 255);//红色
    viewer1.addPointCloud<pcl::PointXYZRGBNormal>(cloud_with_normal, cloud_color_handler1, "point cloud");
    viewer1.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "point cloud");

    viewer1.addPointCloudNormals<pcl::PointXYZRGBNormal, pcl::Normal>(mesh_cloud, mesh_normals, 1, 0.02, "cloud_with_normal normals");
    viewer1.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "cloud_with_normal normals");
    viewer1.initCameraParameters();

    while (!viewer.wasStopped() && !viewer1.wasStopped()) {
        viewer.spinOnce();
        viewer1.spinOnce();
    }

#endif

    return 0;
}