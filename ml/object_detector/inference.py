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

import matplotlib.pyplot as plt
from torchvision.transforms import transforms

sys.path.insert(0, r'object_detector/yolov7')

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

    def __init__(self, image_size, conf_thresh, iou_thresh, num_classes, view_img):
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
        self.device = select_device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights = r"object_detector\finetuned_models\real_best_e6_200_epochs.pt"
        self.weights = r"object_detector\finetuned_models\real_best_e6_200_epochs.pt"
        self.image_size = image_size  # input  image should be (1280 x 1280)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.num_classes = num_classes
        self.classes = list(range(0, self.num_classes))
        self.view_img = view_img

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device).half()  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.image_size = check_img_size(self.image_size, s=self.stride)  # check img_size

        # Get names and colors
        self.names = ['Exit Right', 'Exit Left', 'Exit Straight', 'Fire Extinguisher Right', 'Fire Extinguisher Left',
                      'Fire Extinguisher Straight', 'Seat Numbers', 'Wheelchair Seat Numbers', 'Seat Utilities',
                      'Cycle Reservation', 'Wi-Fi', 'Toilet Right', 'Toilet Left', 'Wheelchair Area',
                      'Wheelchair Assistants Area', 'Priority Seat', 'Priority Seating Area', 'Overhead Racks Warning',
                      'Mind The Gap', 'CCTV Warning', 'Call Cancellation', 'Telephone S.O.S', 'Push To Stop Train',
                      'Emergency Door Release', 'Emergency Information', 'Litter Bin', 'Smoke Alarm',
                      'Toilet Door Latch', 'Hand Washing', 'Toilet Tissue', 'Toilet Warning', 'Handrail',
                      'Caution Magnet', 'Baby Changing Bed', 'C3', 'AC', 'Electricity Hazard', 'Ladder']
        self.colours = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]  # define random colours for each label in the dataset

        # model warmup
        self.model(
            torch.zeros(1, 3, self.image_size, self.image_size).to(self.device).type_as(next(self.model.parameters())))

        sys.stdout = sys.__stdout__  # enable printing

    def forward(self, data_src, processed_destination):
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

        # Set Dataloader
        dataset = LoadImages(data_src, img_size=self.image_size, stride=self.stride)

        bboxes, labels = [], []
        w, h = 1280, 1280
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
            pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh, classes=self.classes, agnostic=False)[0]

            # get bbox and label info
            bboxes.append(pred[:, :4].tolist())
            __labels__ = []
            for p in pred[:, -1]:
                names = self.names[int(p.item())]
                if (names == 'Exit Right') or (names == 'Exit Left'):
                    names = 'Exit Left/Right'
                elif (names == 'Fire Extinguisher Right') or (names == 'Fire Extinguisher Left'):
                    names = 'Fire Extinguisher Left/Right'
                elif (names == 'Toilet Right') or (names == 'Toilet Left'):
                    names = 'Toilet Left/Right'

                __labels__.append(names)

            labels.append(__labels__)

            # save image with bbox predictions overlay
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()

            # resize image
            resize_image = transforms.Resize([w, h])
            im0 = np.array(resize_image(Image.fromarray(im0)))

            for info in pred:
                # add bboxes around objects
                box = info[:4]
                label = int(info[-1])

                # rescale bboxes
                box = resize_bbox(box, h, w)

                # add bboxes to image
                cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), tuple(self.colours[label]),
                              10)

                # add filled bboxes with object label above bboxes
                c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                line_thickness = 5  # line/font thickness
                tf = max(line_thickness - 1, 1)  # font thickness
                t_size = cv2.getTextSize(self.names[label], 0, fontScale=line_thickness / 3, thickness=tf)[0]
                c2 = int(box[0]) + t_size[0], int(box[1]) - t_size[1] - 3
                cv2.rectangle(im0, c1, c2, self.colours[label], -1, cv2.LINE_AA)  # filled
                cv2.putText(im0, self.names[label], (c1[0], c1[1] - 2), 0, line_thickness / 3, [225, 255, 255],
                            thickness=tf, lineType=cv2.LINE_AA)

            # save image
            data_dst = os.path.join(processed_destination, tail)
            cv2.imwrite(data_dst, im0)

            # STEP 2.4.2: SAVE NORMALIZED SIGN IMAGES

            for i, (box) in enumerate(pred[:, :4].cpu().numpy()):
                image = img.squeeze().cpu()
                image = image.permute(1, 2, 0).numpy()

                x1, y1 = box[0], box[1]
                x2, y2 = box[2], box[3]
                src = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]).reshape((4, 2))
                dst = np.array([[0, 0], [0, image.shape[1]], [image.shape[0], image.shape[1]], [image.shape[0], 0]]).reshape((4, 2))

                tform = transform.estimate_transform('projective', src, dst)
                tf_img = transform.warp(image, tform.inverse)

                # plotting the transformed image
                fig, ax = plt.subplots()
                ax.imshow(tf_img)

                plt.axis('off')
                plt.savefig(f"samples/normalized_images/{i}.png", bbox_inches='tight', pad_inches=0)
                plt.close()

            # STEP 2.4.3: DELETE LOCAL IMAGES

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

    def video_forward(self, data_src, processed_destination):
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

        # Set Dataloader
        dataset = LoadImages(data_src, img_size=self.image_size, stride=self.stride)

        bboxes, labels = [], []
        loop = tqdm(enumerate(dataset), total=len(dataset))
        for index, (path, img, im0s, vid_cap) in loop:  # for every image in data path

            # STEP 2.4.1: Run Object Detector on each image
            head, tail = ntpath.split(path)
            img = torch.from_numpy(img).to(self.device).half().unsqueeze(0)
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            # print(f"Path: {tail}")

            # Inference
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=None)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh, classes=self.classes, agnostic=False)[0]

            # get bbox and label info
            bboxes.append(pred[:, :4].tolist())
            __labels__ = []
            for p in pred[:, -1]:
                names = self.names[int(p.item())]
                if (names == 'Exit Right') or (names == 'Exit Left'):
                    names = 'Exit Left/Right'
                elif (names == 'Fire Extinguisher Right') or (names == 'Fire Extinguisher Left'):
                    names = 'Fire Extinguisher Left/Right'
                elif (names == 'Toilet Right') or (names == 'Toilet Left'):
                    names = 'Toilet Left/Right'

                __labels__.append(names)

            labels.append(__labels__)
            # labels.append([self.names[int(p.item())] for p in pred[:, -1]])

            # save image with bbox predictions overlay
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()

            for info in pred:
                # add bboxes around objects
                box = info[:4]
                label = int(info[-1])
                cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), tuple(self.colours[label]),
                              5)

                # add filled bboxes with object label above bboxes
                c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                line_thickness = 2  # line/font thickness
                tf = max(line_thickness - 1, 1)  # font thickness
                t_size = cv2.getTextSize(self.names[label], 0, fontScale=line_thickness / 3, thickness=tf)[0]
                c2 = int(box[0]) + t_size[0], int(box[1]) - t_size[1] - 3
                cv2.rectangle(im0, c1, c2, self.colours[label], -1, cv2.LINE_AA)  # filled
                cv2.putText(im0, self.names[label], (c1[0], c1[1] - 2), 0, line_thickness / 3, [225, 255, 255],
                            thickness=tf, lineType=cv2.LINE_AA)

            # save image
            data_dst = os.path.join(processed_destination, tail)
            cv2.imwrite(data_dst, im0)

            # delete image (processed images)
            processed_file = os.path.join(processed_destination, tail)
            if os.path.exists(processed_file):
                os.remove(processed_file)

            # view img with bbox predictions overlay
            if self.view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(0)  # number in millisecond
                cv2.destroyAllWindows()

            # update progress bar
            loop.set_description(f"\tVideo Frame [{index + 1}/{len(dataset)}]")

        return bboxes, labels
