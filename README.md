# RepDistiller-NAS



## Installation

This repo was tested with Ubuntu 16.04.5 LTS, Python 3.5, PyTorch 0.4.0, and CUDA 9.0. But it should be runnable with recent PyTorch versions >=0.4.0

## Running

1. 蒸馏第一个student代码如下:

    ```
    PYTHONPATH=./ python train_student_imagenet.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s MobileNetV2 -r 0.1 -a 0.9 -b 0 --trial 1 --path_config ./proxyless_nas/config/net0.config
    
   
   ```
   第二个student需要修改
   
   ```
   --path_config ./proxyless_nas/config/net0.config
   ```
   
   为
   
   ```
   --path_config ./proxyless_nas/config/net1.config
   ```
   
   后续同理
