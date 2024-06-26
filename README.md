## 项目名称
> 基于mmpose的通用场景骨骼点检测  



## 环境准备
```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"

cd third_party/mmpose
pip install -r requirements.txt
pip install .
```



## 运行说明

Pose模型转换：

```
python3 third_party/mmdeploy/tools/deploy.py \
    --dump-info --device cuda --work-dir /workspace/works/exp_1123/4070ti \
    /workspace/third_party/mmdeploy/configs/mmpose/pose-detection_simcc_tensorrt-fp16_dynamic-384x288.py \
    /workspace/configs/model/mmpose/castpose_l_ll_upbody-384×288.py \
    /workspace/works/exp_1123/dis_2_new_9.pth \
    /workspace/third_party/mmdeploy/demo/resources/human-pose.jpg

python3 third_party/mmdeploy/tools/deploy.py \
    --dump-info --device cuda --work-dir /workspace/works/exp_1123/jetson_orin_32g \
    /workspace/third_party/mmdeploy/configs/mmpose/pose-detection_simcc_tensorrt-fp16_dynamic-384x288.py \
    /workspace/configs/model/mmpose/castpose_300e_distill-2x2_I-II-ubody_384x288.py \
    /workspace/model_zoo/upperpose/castpose_300e_distill-2x2_I-II-ubody_384x288.pth \
    /workspace/third_party/mmdeploy/demo/resources/human-pose.jpg
```

Det模型转换
```
python3 third_party/mmdeploy/tools/deploy.py \
    --dump-info --device cuda --work-dir /workspace/exports/rtmdet-s/4070ti \
    /workspace/third_party/mmdeploy/configs/mmdet/detection/detection_tensorrt-fp16_static-640x640.py \
    /workspace/third_party/mmpose/projects/rtmpose/rtmdet/person/rtmdet_s_8xb32-300e_humanart.py \
    /workspace/model_zoo/detection/rtmdet_s_8xb32-300e_humanart-af5bd52d.pth \
    /workspace/third_party/mmdeploy/demo/resources/human-pose.jpg
```



