本文件是caffe中添加新层FeatReshapeCommon所需的.cpp、.hpp、.cu文件。
参考博客https://blog.csdn.net/u012426298/article/details/81016600
作者：tq
时间:2018-7-12
---------------------------------------------------------------------------
功能：实现任意倍数sample_step下采样（如8倍），channel增加（sample_step * sample_step）倍
1.feat_reshape_common_layer.cpp和feat_reshape_common_layer.cu位于caffe_yolo_fine_order\src\caffe\layers下

2.caffe.proto位于caffe_yolo_fine_order\src\caffe\proto下
-------------------------------------------------------------------------------
这些文件参考自inference文件中的文件（只能实现2倍下采样，channel增大为原来4倍）
