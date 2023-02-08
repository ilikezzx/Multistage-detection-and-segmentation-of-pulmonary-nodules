#多阶段肺结节智能检测和分割算法
____
1.主要流程：
```
1. CT的输入及其预处理过程
2. 第一阶段：使用Yolov5网络在切面层面上粗定位候选结节
3. 第二阶段：使用CNS算法在三维层面上去除冗余候选
4. 按照检测位置进行切块
5. 第三阶段：基于集成学习的分类网络在切块层面上细分类候选结节
6. 第四阶段：使用3D-ResUNet改进版在切块层面上细分割候选结节
```
2.整体框架设计图：
![image](https://github.com/ilikezzx/Multistage-detection-and-segmentation-of-pulmonary-nodules/blob/master/image/overall.tif)

3.本算法使用窗体进行可视化，文件夹简介如下：
```
./gui 窗体构造
./image 存放本地图像
./loading 图像输入
./models 四个阶段的模型
```
