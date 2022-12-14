import os
import os.path
import ntpath

import cv2
import torch
import torch.nn as nn
import numpy as np
from numpy import random
from PIL import Image
from skimage import transform
from numpy import random
import sys
from tqdm import tqdm

import torchvision.transforms as transforms

sys.path.insert(0, r'damage_detector/yolov7')

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel



class DamageDetector(nn.Module):
    """
    Object detector class that initializes the object detector setup and allows for processing all instances of the inspectionWalkthrough object.
    To use the object detector call object() (do not use object.forward()). This will return bbox coordinates and the labels for each identified object.

    @authors: Benjamin Sanati
    """

    def __init__(self, image_size, conf_thresh, iou_thresh, num_classes):
        """
        @brief: Initializes the object detector for processing. Sets up object detector once, reducing the total processing time compared to setting up on every inference call.
                NMS breakdown:
                    1) Discard all the boxes having probabilities less than or equal to a pre-defined threshold (say, 0.5)
                    2) For the remaining boxes:
                        a) Pick the box with the highest probability and take that as the output prediction
                        b) Discard any other box which has IoU greater than the threshold with the output box from the above step
                    3) Repeat step 2 until all the boxes are either taken as the output prediction or discarded
        Args:
            image_size: size of input image (1280 for YOLOv7-e6 model)
            conf_thresh: Minimum confidence requirement from YOLOv7 model output (~0.55 is seen to be the best from the object detector training plots)
            iou_thresh: IoU threshold for NMS
            num_classes: Number of classes that can be defined (number of the types of signs)

        @authors: Benjamin Sanati
        """
        super(DamageDetector, self).__init__()

        sys.stdout = open(os.devnull, 'w')  # block printing momentarily

        # Initialize data and hyperparameters (to be made into argparse arguments)
        self.device = select_device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights = "damage_detector/finetuned_models/best_e6_50_epochs.pt"
        self.image_size = image_size  # input  image should be (1280 x 1280)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.num_classes = num_classes
        self.classes = list(range(0, self.num_classes))

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device).half()  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.image_size = check_img_size(self.image_size, s=self.stride)  # check img_size

        # Get names
        self.names = ['conforming', 'damaged']

        # model warmup
        self.model(
            torch.zeros(1, 3, self.image_size, self.image_size).to(self.device).type_as(next(self.model.parameters())))

        sys.stdout = sys.__stdout__  # enable printing

    def forward(self, data_src):
        """
        @brief: Runs object detection model on each image in inspectionWalkthrough. Uploads processed images to firestore storage. Returns bbox coordinate and labels for each object in each image.
        Args:
            data_src: source of images in local storage folder

        Returns:
            bboxes - A list of [# images, # objects, 4] bbox coordinates
            labels - A list of [# images, # objects] object labels

        @authors: Benjamin Sanati
        """

        # Set Dataloader
        dataset = LoadImages(data_src, img_size=self.image_size, stride=self.stride)

        labels = []
        loop = tqdm(enumerate(dataset), total=len(dataset))
        for index, (path, img, im0s, vid_cap) in loop:  # for every image in data path
            # STEP 2.4.1: Run Object Detector on each image
            head, tail = ntpath.split(path)
            img = torch.from_numpy(img).to(self.device).half().unsqueeze(0)
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            # Inference
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=None)[0]
                
            # Apply NMS
            predictions = non_max_suppression(pred, 0, 0, classes=self.classes, agnostic=False)[0]
            confs = predictions[..., -2]
            label = predictions[..., -1]

            # there can only be one classification, therefore, get the label with the highest confidence score
            _, highest_confidence_index = torch.max(confs, 0)
            labels.append(self.names[int(label[..., highest_confidence_index].item())])

            # STEP 2.4.2: DELETE LOCAL IMAGES
            unprocessed_file = os.path.join(data_src, tail)
            if os.path.exists(unprocessed_file):
                os.remove(unprocessed_file)

            # update progress bar
            loop.set_description(f"\tSign [{index + 1}/{len(dataset)}]")

        return labels
