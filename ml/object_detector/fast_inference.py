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
import sys
from tqdm import tqdm

sys.path.insert(0, r'yolov7')

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def resize_bbox(bbox, image_height, image_width):
    x_scale = (image_height / 1280)
    y_scale = (image_width / 1280)

    bbox[0] *= x_scale
    bbox[1] *= y_scale
    bbox[2] *= x_scale
    bbox[3] *= y_scale

    return bbox


class ObjectDetector(nn.Module):
    """
    Object detector class that initializes the object detector setup and allows for processing all instances of the inspectionWalkthrough object.
    To use the object detector call object() (do not use object.forward()). This will return bbox coordinates and the labels for each identified object.

    @authors: Benjamin Sanati
    """

    def __init__(self, conf_thresh, iou_thresh, view_img):
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
            view_img: A bool to view the output of a processed image during processing

        @authors: Benjamin Sanati
        """
        super(ObjectDetector, self).__init__()

        sys.stdout = open(os.devnull, 'w')  # block printing momentarily

        # Initialize data and hyperparameters (to be made into argparse arguments)
        self.device = torch.device('cuda:0')
        self.weights = r"finetuned_models\best.pt"
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.view_img = view_img
        self.imgsz = 1280

        """
        # Load model
        self.model = attempt_load(self.weights, map_location=self.device).half().to(self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.image_size = check_img_size(self.image_size, s=self.stride)  # check img_size

        # Get names and colors
        self.names = ['Exit', 'Exit Straight', 'Fire Extinguisher', 'Fire Extinguisher Straight', 'Seat Numbers',
                      'Wheelchair Seat Numbers', 'Seat Utilities', 'Cycle Reservation', 'Wi-Fi', 'Toilet',
                      'Wheelchair Area', 'Wheelchair Assistants Area', 'Priority Seat', 'Priority Seating Area',
                      'Overhead Racks Warning', 'Mind The Gap', 'CCTV Warning', 'Call Cancellation', 'Telephone S.O.S',
                      'Push To Stop Train', 'Emergency Door Release', 'Emergency Information', 'Litter Bin',
                      'Smoke Alarm', 'Toilet Door Latch', 'Hand Washing', 'Toilet Tissue', 'Toilet Warning', 'Handrail',
                      'Caution Magnet', 'Baby Changing Bed', 'C3', 'AC', 'Electricity Hazard', 'Ladder']
        self.colours = [[random.randint(0, 255) for _ in range(3)] for _ in
                        self.names]  # define random colours for each label in the dataset

        # model warmup
        self.model(
            torch.zeros(1, 3, self.image_size, self.image_size).type_as(next(self.model.parameters())))
        """

        # Load model
        model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(model.stride.max())  # model stride
        self.model = model.half()
        self.names = model.module.names if hasattr(model, 'module') else model.names
        self.classes = list(range(0, len(self.names)))
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # warmup
        self.model(
            torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(model.parameters())))  # run once

        sys.stdout = sys.__stdout__  # enable printing

    def forward(self):
        """
        @brief: Runs object detection model on each image in inspectionWalkthrough. Uploads processed images to firestore storage. Returns bbox coordinate and labels for each object in each image.
        Args:
            data_src: source of images in local storage folder
            processed_destination: destination of processed image to be saved to local storage folder

        Returns:
            bboxes - A list of [# images, # objects, 4] bbox coordinates
            labels - A list of [# images, # objects] object labels

        @authors: Benjamin Sanati
        """

        # load images
        dataset = LoadImages('test_folder', img_size=self.imgsz, stride=self.stride)

        loop = tqdm(enumerate(dataset), total=len(dataset))
        for idx, (path, img, im0s, vid_cap) in loop:
            img = torch.from_numpy(img).to(self.device)
            img = img.half()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # inference
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = self.model(img)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh, classes=self.classes,
                                       agnostic=True)


if __name__ == '__main__':
    print(f"Configuring Model...")
    objdet = ObjectDetector(conf_thresh=0.5, iou_thresh=0.65, view_img=True)
    print(f"Model Configured!")
    objdet()
