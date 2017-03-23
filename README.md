# layers
You need to install [caffe](https://github.com/BVLC/caffe) or [caffe-windows](https://github.com/microsoft/caffe)。
### bn_layer is used for person-reID task<br>

```cpp
export CAFFE_HOME=/path/to/caffe
cp layers/bn_layer/bn_layer.hpp $CAFFE_HOME/include/caffe/layers/bn_layer.hpp
cp layers/bn_layer/bn_layer.cpp $CAFFE_HOME/src/caffe/layers/bn_layer.cpp
cp layers/bn_layer/bn_layer.cu $CAFFE_HOME/src/caffe/layers/bn_layer.cu
```
Modify `$CAFFE_HOME/src/caffe/proto/caffe.proto` according to `layers/bn_layer/caffe.proto`
### normalize_layer and Aggregate_layer is used for image retrieval task
```cpp
cp layers/normalize_layer/normalize_layer.hpp $CAFFE_HOME/include/caffe/layers/normalize_layer.hpp
cp layers/normalize_layer/normalize_layer.cpp $CAFFE_HOME/src/caffe/layers/normalize_layer.cpp
cp layers/Aggregate_layer/Aggregate_layer.hpp $CAFFE_HOME/include/caffe/layers/Aggregate_layer.hpp
cp layers/Aggregate_layer/Aggregate_layer.cpp $CAFFE_HOME/src/caffe/layers/Aggregate_layer.cpp
```
No need to modify `caffe.proto`<br>

compile caffe
