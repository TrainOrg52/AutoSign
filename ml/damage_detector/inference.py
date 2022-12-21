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

from transformers import BeitForImageClassification, BeitFeatureExtractor


class DamageDetector(nn.Module):
    """
    Damage detection class that initializes the ViT setup and allows for processing all signs in all inspectionWalkthrough media.
    To use the damage detector, call object() (do not use object.forward()). This will return the labels for each sign.

    @authors: Benjamin Sanati
    """

    def __init__(self):
        """
        @brief: Initializes the damage detector for processing. Sets up the classifier once, reducing the total processing time compared to
        setting up on every inference call.

        @authors: Benjamin Sanati
        """
        super(DamageDetector, self).__init__()

        sys.stdout = open(os.devnull, 'w')  # block printing momentarily
        repo_name = r"damage_detector\finetuned_models\beit-base-patch16-224-pt22k-finetuned-autosign"
        self.feature_extractor = BeitFeatureExtractor.from_pretrained(repo_name)
        self.model = BeitForImageClassification.from_pretrained(repo_name)

        sys.stdout = sys.__stdout__  # enable printing

    def forward(self, data_src):
        """
        @brief: Runs damage detection model on each sign in model. Returns labels for each object in each image.
        Args:
            data_src: source of images in local storage folder

        Returns:
            labels - A list of [# images, # objects] object labels

        @authors: Benjamin Sanati
        """

        labels = []
        loop = tqdm(enumerate(os.listdir(data_src)), total=len(os.listdir(data_src)))
        for index, (filename) in loop:
            image_path = os.path.join(data_src, filename)
            image = Image.open(image_path)

            # prepare image for the model
            encoding = self.feature_extractor(image.convert("RGB"), return_tensors="pt")

            # ############## #
            # GET PREDICTION #
            # ############## #

            # forward pass
            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits

            # prediction
            predicted_class_idx = logits.argmax(-1).item()
            predicted_class = self.model.config.id2label[predicted_class_idx]
            labels.append(predicted_class)

            # update progress bar
            loop.set_description(f"\t\tSign [{index + 1}/{len(os.listdir(data_src))}]")

        """
        # STEP 2.4.2: DELETE LOCAL IMAGES
        for filename in os.listdir(data_src):
            image_path = os.path.join(data_src, filename)
            if os.path.exists(image_path):
                os.remove(image_path)
        """

        return labels
