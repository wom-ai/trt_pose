#!/usr/bin/env python3
# coding: utf-8

# First, let's load the JSON file which describes the human pose task.  This is in COCO format, it is the category descriptor pulled from the annotations file.  We modify the COCO category slightly, to add a neck keypoint.  We will use this task description JSON to create a topology tensor, which is an intermediate data structure that describes the part linkages, as well as which channels in the part affinity field each linkage corresponds to.

import json
import trt_pose.coco

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)
#
#
## Next, we'll load our model.  Each model takes at least two parameters, *cmap_channels* and *paf_channels* corresponding to the number of heatmap channels
## and part affinity field channels.  The number of part affinity field channels is 2x the number of links, because each link has a channel corresponding to the
## x and y direction of the vector field for each link.
#
#import trt_pose.models
#
#num_parts = len(human_pose['keypoints'])
#num_links = len(human_pose['skeleton'])
#
#model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
#
#
## Next, let's load the model weights.  You will need to download these according to the table in the README.
#
## In[ ]:
#
#
import torch
#
#MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
#
#model.load_state_dict(torch.load(MODEL_WEIGHTS))
#
#
## In order to optimize with TensorRT using the python library *torch2trt* we'll also need to create some example data.  The dimensions
## of this data should match the dimensions that the network was trained with.  Since we're using the resnet18 variant that was trained on
## an input resolution of 224x224, we set the width and height to these dimensions.
#
#
#WIDTH = 224
#HEIGHT = 224
#
#data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
#
#
## Next, we'll use [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) to optimize the model.  We'll enable fp16_mode to allow optimizations to use reduced half precision.
#
#
#import torch2trt
#
#model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
#
#
## The optimized model may be saved so that we do not need to perform optimization again, we can just load the model.  Please note that TensorRT has device specific optimizations, so you can only use an optimized model on similar platforms.
#
#
OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
#
#torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)


# We could then load the saved model using *torch2trt* as follows.

from torch2trt import TRTModule

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))


# Next, let's define a function that will preprocess the image, which is originally in BGR8 / HWC format.

import cv2
import torchvision.transforms as transforms
import PIL.Image

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


# Next, we'll define two callable classes that will be used to parse the objects from the neural network, as well as draw the parsed objects on an image.

from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)


# Finally, we'll define the main execution loop.  This will perform the following steps
# 
# 1.  Preprocess the camera image
# 2.  Execute the neural network
# 3.  Parse the objects from the neural network output
# 4.  Draw the objects onto the camera image
# 5.  Convert the image to JPEG format and stream to the display widget

global start
global end

from jetcam.utils import bgr8_to_jpeg
def execute(image):
    global start
    global end
    data = preprocess(image)
    end = time.time()
    print("Prep: {}".(end - start))
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)
    draw_objects(image, counts, objects, peaks)
    image = cv2.resize(image, (1000, 1000), interpolation = cv2.INTER_AREA)
    cv2.imshow("result", image)
    cv2.waitKey(1)

from simple_pyspin import Camera
import numpy as np
import math
import time

map_x = np.zeros((1100, 4400), dtype=np.float32) 
map_y = np.zeros((1100, 4400), dtype=np.float32) 
for i in range(1100):
    for j in range(4400):
        theta = math.pi * j / (2 * 1100)
        map_x[i, j] = i * np.cos(theta) + 1100
        map_y[i, j] = -i * np.sin(theta) + 1100

with Camera() as cam:
    cam.stop()
    cam.Width = 2200
    cam.Height = 2200
    cam.OffsetX = 504
    cam.start()
    while True:
        img = cam.get_array()
        start = time.time()
        img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
        img = cv2.remap(img, map_x, map_y, cv2.INTER_NEAREST)
        img = img[:, 2200:3300]
        resized = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
        execute(resized)
