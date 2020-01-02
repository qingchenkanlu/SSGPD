# Self-supervised Grasp Pose Detector

#### 问题汇总：

**1、优化接触点查找**

**2、使用tsdf-fusion获得的mesh文件转换为sdf之后，法线方向不准确**

- [ ] 方案1：直接从tsdf转换为sdf（优先）未找到转换方法，即使转换成功，得到的表面法线依然不准确


- [x] 方案2：舍弃sdf，直接使用点云处理

  	背景：表面点云可直接获取表面点但无法线，重建的mesh有法线但还包含非表面点表

​	面法线获取方法：  ​	

​	**方法1、**直接使用PCL计算表面法线，须设置相机位置，但只设置一个相机无法正确估计各个法线，须使	


  ​		      用多相机位置(分为四块较好)进行法线估计。方案：PCA法计算桌面以上点云的质心，并获得包

  ​		      围盒，以质心为原点， 划分四个象限，每个象限上方放置一相机。



  	**方法2、**将重建的mesh通过pcl转换为点云，此点云包含点及法线信息但包含非表面点，通过表面点云可以获得表面点的坐标，通过kdtree在mesh转换所得点云中查找各表面点附近点的法线即可。经测试，优于方法1。

<img src="/home/sdhm/Projects/SSGPD/Readme/compute 3d centroid by PCA.png" alt="compute 3d centroid by PCA" style="zoom: 67%;" />

<img src="/home/sdhm/Projects/SSGPD/Readme/calculate surface normal in voxel.png" alt="calculate surface normal in voxel" style="zoom:67%;" />

##### 3、单视角点云会导致闭合区域点数太少

![点数太少-无法判断可否抓取-须提高最小点数阈值](/home/sdhm/Projects/SSGPD/Readme/点数太少-无法判断可否抓取-须提高最小点数阈值.png)

闭合区域点数太少，导致肉眼都无法判断是否可以抓取，应提高最小点数阈值。

##### 4、手爪厚度对获取的点云信息量有影响

![应增加手抓厚度-以获取更多闭合区域点云信息](/home/sdhm/Projects/SSGPD/Readme/应增加手抓厚度-以获取更多闭合区域点云信息.png)

应增加手爪厚度以获取更多的闭合区域点云信息，便于获取丰富的特征信息，**可针对手爪厚度做对比实验。**

### Before Install

All the code should be installed in the following directory:
```
mkdir -p $HOME/code/
cd $HOME/code/
```
### Install all the requirements (Using a virtual environment is recommended)
1. Make sure you are  in the Python3 environment.

1. Clone this repository:
    ```bash
    cd $HOME
    git clone https://github.com/MrRen-sdhm/SSGPD
    mv PointNetGPD grasp-pointnet
    ```

1. Install requirements in `requirements.txt`
    ```bash
    cd SSGPD
    pip install -r requirements.txt
    ```

1. Install our modified autolab_core (Modify from [Berkeley Automation Lab: autolab_core](https://github.com/BerkeleyAutomation/autolab_core.git))

    ```
    cd SSGPD/3rdparty/autolab_core
    python setup.py install
    ```

1. Install our modified autolab_perception (Modify from [Berkeley Automation Lab: autolab_perception](https://github.com/BerkeleyAutomation/perception.git))

    ```
    cd SSGPD/3rdparty/autolab_perception
    pip install -e .
    ```

1. Modify the gripper configurations to your own gripper

    ```bash
    gedit SSGPD/Generator/gripper_params.yaml
    ```

### Generate Grasp Dataset

If use ycb dataset，generate point cloud first.

```
cd Dataset/ycb
python ycb_cloud_generate.py
```

Generate dataset by running the code:

```bash
cd Generator
python grasps_generator.py
python dataset_generator.py
```
### Visualization tools

  mlab: https://docs.enthought.com/mayavi/mayavi/mlab.html

- Visualization grasps
    ```bash
    cd Generator/utils
    python grasps_show.py
    ```

- Visualization dataset

  ```bash
  cd Generator/utils
  python dataset_show.py
  ```

### Training the network

1. Data prepare:
    ```bash
    cd Classifier
    ```

1. Run the experiments:
   
    - Launch a tensorboard for monitoring
    
	```bash
    tensorboard --logdir ./assets/log --port 8080
	```

    - run an experiment
	
    ```
	python main_1v.py --mode train
	```
    
	- reload pretrained model
	
    ```
	python main_1v.py --mode train --load-model default_120.model --load-	 epoch 120
	```

File name and corresponding experiment:

```
main_1v.py        --- 1-viewed point cloud, 2 class
main_1v_mc.py     --- 1-viewed point cloud, 3 class
main_fullv.py     --- Full point cloud, 2 class
main_fullv_mc.py  --- Full point cloud, 3 class
```




#### Acknowledgement

- [gpg](https://github.com/atenpas/gpg)
- [gpd](https://github.com/atenpas/gpd)
- [dex-net](https://github.com/BerkeleyAutomation/dex-net)
- [mayavi](https://github.com/enthought/mayavi)

