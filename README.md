# Physically Adversarial Infrared Patches with Learnable Shapes and Locations

## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>
  
[**Python>=3.6.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/shighghyujie/infrared_patch_attack/requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):
  
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->


```bash
$ git clone https://github.com/shighghyujie/infrared_patch_attack.git
$ cd infrared_patch_attack
$ pip install -r requirements.txt
```

</details>

<details open>
<summary>Preparation Before Attack</summary>
  You can obtain infrared pedestrian data through the following dataset:

  The Opening Database [FLIR ADAS](https://www.flir.com/oem/adas/adas-dataset-form/) to simulate the physical attacks.
  The Opening Database CVC-14 [CVC-14](http://adas.cvc.uab.es/webfiles/datasets/CVC-14-Visible-Fir-Day-Night/CVC-14.rar)
  
  Here, we provide the our trained weights of YOLOv3 model (based on yolov3.pt) at yolov3\weights\best.pt and some images to conduct attacks in datasets\pedestrian.
  
  Firstly, you should put the folder "weights" in "yolov3".
  
  **The weight can be downloaded form : https://pan.baidu.com/s/1cvG_FDyjMedQfixnVvHAkA, code:q1w2**
</details>

<details open>
<summary>Attack</summary>

Running this command for attacks:
```bash
$ cd infrared_patch_attack
$ python shaped_patch_attack.py
```

If you want to use your own data, you can pass the dataset path to the "victim_imgs" parameter.

For more attack settings, you can change the values of parameters at config/config.py:

width, height:  the width and height of optimized object(infrared mask). If the value is too large, it may prevent the mask from aggregating properly.

emp_iterations: the steps of optimization.

cover_rate:     the coverage rate of infrared patches on the target.

content:        the gray value of infared patch in the thermal image.
