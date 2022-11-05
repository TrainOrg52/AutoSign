import numpy as np
import matplotlib.pyplot as plt

from torchvision.io.image import read_image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes, make_grid


def show(images):
    if not isinstance(images, list):
        images = [images]
    fig, axs = plt.subplots(ncols=len(images), squeeze=False)
    for i, img in enumerate(images):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[0, i].margins(x=0)
    plt.show()


def image_viewer(func):
    def wrapper(*args):
        img = func(*args)
        img = to_pil_image(img)
        img.show()

    return wrapper


def grid_viewer(func):
    def wrapper(*args):
        img_list = func(*args)
        grid = make_grid(img_list)
        show(grid)

    return wrapper


def img_bboxes(img, bboxes, labels, colours):
    img_boxes = draw_bounding_boxes(img, boxes=bboxes,
                                    labels=labels,
                                    colors=colours,
                                    width=4,
                                    font='font/Roboto-Regular.ttf', font_size=50)

    return img_boxes


class Images:
    """
    Functionality
    ---------------
    Image media handler. Contains methods for:
        .Image Previewing
        .Image Previewing with BBox Overlay
        .Image Grid Previewing
        .Image Grid Previewing with BBox Overlay
        .Image to PyTorch tensor transformations
        .PyTorch tensor to Image visualization

    Creator/s: Ben Sanati
    """

    def __init__(self, path=None, tensor=None):
        """
        :param path: Either 1) A string of img path, or 2) A list of img paths for a grid of images
        :param tensor: A PyTorch tensor to be converted to an image
        """
        if type(path) == list:
            self.img = [read_image(p) for p in path]
        elif path is not None:
            self.img = read_image(path)

        self.tensor = tensor

        self.transform = transforms.Compose([
            transforms.PILToTensor()
        ])

    @image_viewer
    def image_preview(self):
        """
        Preview image without any processing performed
        """
        return self.img

    @image_viewer
    def image_bbox_preview(self, bboxes, labels, colours):
        """
        Preview Image with BBox Overlay
        ----------------------------------
        :param bboxes: A PyTorch Tensor in (x1, y1, x2, y2) bbox format of dimension [# boxes, 4]
        :param labels: A list containing the labels corresponding to each bbox
        :param colours: A list containing the colours of each bbox (if only one colour desired, colours is a string)

        :return: image with bbox overlay
        """
        # draw bboxes on img
        img_boxes = img_bboxes(self.img, bboxes, labels, colours)

        return img_boxes

    @grid_viewer
    def image_grid(self):
        """
        Preview Images in a Grid Format
        """
        return self.img

    @grid_viewer
    def image_bbox_grid(self, bboxes, labels, colours):
        """
        Preview Images in a Grid Format with BBox Overlay
        ---------------------------------------
        :param bboxes: A list of BBox PyTorch Tensors in (x1, y1, x2, y2) bbox format of dimension [# boxes, 4]
        :param labels: A list of the lists containing the labels corresponding to each bbox
        :param colours: A list of the lists containing the colours of each bbox (if only one colour desired, colours is a string)

        :return: grid of images with bbox overlay
        """
        # draw bboxes on each img
        img_boxes = [img_bboxes(im, bbox, label, colour) for (im, bbox, label, colour) in
                     zip(self.img, bboxes, labels, colours)]

        return img_boxes

    def image_to_tensor(self, img):
        """
        Convert image to PyTorch Tensor
        Note, the transform is very fundamental and has no preprocessing applied to it
        Once the object detector is selected, then the preprocessing phase can be applied
        :param img: A PIL image that is to be converted to a PyTorch tensor
        :return: A PyTorch tensor representing the input PIL image
        """
        return self.transform(img)

    def tensor_to_image(self):
        """
        Convert a tensor to a PIL image for viewing
        :return: A PIL image
        """
        if self.tensor is not None:
            return to_pil_image(self.tensor)
