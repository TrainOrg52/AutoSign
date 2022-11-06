import os
import os.path
import ntpath

import cv2
import torch
import torch.nn as nn
from numpy import random
import sys
from tqdm import tqdm

sys.path.insert(0, r'C:\Users\benja\Documents\University\UniWork\GDP\firebase_api\setup\yolov7_e6\yolov7')

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class ObjectDetector(nn.Module):
    def __init__(self, image_size, conf_thresh, iou_thresh, num_classes, view_img):
        super(ObjectDetector, self).__init__()

        # Initialize data and hyperparameters (to be made into argparse arguments)
        self.device = select_device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights = r"object_detector\finetuned_models\best_e6_50_epochs.pt"
        self.image_size = image_size  # input  image should be (1280 x 1280)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.num_classes = num_classes
        self.classes = list(range(0, self.num_classes))
        self.view_img = view_img

        # Load model
        blockPrint()
        self.model = attempt_load(self.weights, map_location=self.device).half()  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.image_size = check_img_size(self.image_size, s=self.stride)  # check img_size

        # Get names and colors
        self.names = ['In Emergency Pull Handle Down', 'No Smoking Or Vaping', 'Call For Aid', 'Push Button When Lit',
                      'In Emergency Pull To Operate', 'First Aid Equipment', 'Fire Extinguisher Beneath The Seats',
                      'Emergency Exit Forwards', 'Priority Space For Wheelchair Users',
                      'Train-To-Track Evacuation Instructions',
                      'Please Give Up This Seat For Someone Less Able To Stand Than You']  # self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colours = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # warmup
        self.model(
            torch.zeros(1, 3, self.image_size, self.image_size).to(self.device).type_as(next(self.model.parameters())))
        enablePrint()

    def forward(self, data_src, processed_destination, storage, storage_roots):
        # Set Dataloader
        dataset = LoadImages(data_src, img_size=self.image_size, stride=self.stride)

        bboxes = []
        labels = []
        loop = tqdm(enumerate(zip(dataset, storage_roots)), total=len(dataset))
        for index, ((path, img, im0s, vid_cap), storage_root) in loop:  # for every image in data path
            # STEP 2.4.1: Run Object Detector on each image
            head, tail = ntpath.split(path)
            img = torch.from_numpy(img).to(self.device).half().unsqueeze(0)
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            # Inference
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=None)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh, classes=self.classes, agnostic=False)[0]

            # get bbox and label info
            bboxes.append(pred[:, :4].tolist())
            labels.append([self.names[int(p.item())] for p in pred[:, -1]])

            # save image with bbox predictions overlay
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()

            for info in pred:
                box = info[:4]
                label = int(info[-1])
                cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), tuple(self.colours[label]),
                              5)
                cv2.putText(im0, self.names[label], (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            tuple(self.colours[label]), 3, cv2.LINE_AA)

            # save image
            data_dst = os.path.join(processed_destination, tail)
            cv2.imwrite(data_dst, im0)

            # STEP 2.4.2: UPLOADING PROCESSED IMAGE TO STORAGE
            storage.child(f'{storage_root}/processed.png').put(data_dst)

            # delete image (both processed and non-processed images)
            processed_file = os.path.join(processed_destination, tail)
            unprocessed_file = os.path.join(data_src, tail)
            if os.path.exists(processed_file):
                os.remove(processed_file)
            if os.path.exists(unprocessed_file):
                os.remove(unprocessed_file)

            # view img with bbox predictions overlay
            if self.view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(0)  # number in millisecond
                cv2.destroyAllWindows()

            # update progress bar
            loop.set_description(f"Image [{index + 1}/{len(dataset)}]")

        return bboxes, labels
