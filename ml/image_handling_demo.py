import torch
from PIL import Image
from torchvision.io.image import read_image

from media_handling.image_handling import Images

# ============================================================================ #
# ATTENTION USER!!!
# You don't have to do anything but choose a mode and watch the magic happen
# All the required code for any operation is within a case statement
"""
The mode options include
   'img_preview'
   'img_with_bbox'
   'grid_preview'
   'grid_with_bbox'
   'img_to_tensor'
   'tensor_to_img'
"""
mode = 'img_preview'
# ============================================================================ #

match mode:
    # Image Previewing
    case 'img_preview':
        # initialize image handler with img path
        media_handler = Images('img/baseline.jpg')
        media_handler.image_preview()

    # Image Previewing with BBox Overlay
    case 'img_with_bbox':
        # initialize image handler with img path
        media_handler = Images('img/baseline.jpg')

        # dummy bbox, label and colour values
        baseline_boxes = torch.Tensor([[2138.5056, 921.8126, 3328.8040, 2669.5242],
                                       [481.3222, 694.7492, 1936.0288, 2624.1997],
                                       [2155.3606, 304.8994, 3127.1267, 777.9504]])
        baseline_labels = ['book', 'book', 'cell phone']
        baseline_colours = 'red'

        media_handler.image_bbox_preview(baseline_boxes, baseline_labels, baseline_colours)

    # Image Grid Previewing
    case 'grid_preview':
        # initialize image handler with a list of img paths
        media_handler = Images(['img/baseline.jpg', 'img/comparison_basic.jpg'])
        media_handler.image_grid()

    # Image Grid Previewing with BBox Overlay
    case 'grid_with_bbox':
        # initialize image handler with a list of img paths
        media_handler = Images(['img/baseline.jpg', 'img/comparison_basic.jpg'])

        # dummy bbox, label and colour values
        baseline_boxes = torch.Tensor([[2138.5056, 921.8126, 3328.8040, 2669.5242],
                                       [481.3222, 694.7492, 1936.0288, 2624.1997],
                                       [2155.3606, 304.8994, 3127.1267, 777.9504]])
        baseline_labels = ['book', 'book', 'cell phone']
        baseline_colours = 'red'

        comparison_boxes = torch.Tensor([[2177.4221, 832.5762, 3483.5391, 2689.5098],
                                         [555.5426, 685.4944, 2044.3572, 2655.6428]])
        comparison_labels = ['book', 'book']
        comparison_colours = 'green'

        # create a list of the bboxes, labels, and colours for each img: grid_bboxes = [bboxes_img1, bboxes_img2, ..., bboxes_imgN] etc.
        grid_bboxes = [baseline_boxes, comparison_boxes]
        grid_labels = [baseline_labels, comparison_labels]
        grid_colours = [baseline_colours, comparison_colours]

        media_handler.image_bbox_grid(grid_bboxes, grid_labels, grid_colours)

    # PIL Image to PyTorch tensor transformations
    case 'img_to_tensor':
        media_handler = Images()
        pil_img = Image.open('img/baseline.jpg')
        print(f"Input image: {pil_img}")
        tensor = media_handler.image_to_tensor(pil_img)
        print(f"Output Tensor Dimensions: {tensor.size()}")

    # PyTorch tensor to Image visualization
    case 'tensor_to_img':
        # usually you would have a tensor that is processed, but in this case I will just read an image to a tensor
        tensor = read_image('img/baseline.jpg')
        print(f"Input Tensor dimensions: {tensor.size()}")
        media_handler = Images(tensor=tensor)
        img = media_handler.tensor_to_image()
        print(f"Output Image: {img}")

    case _:
        print(f"Invalid case.\nChoose from the following:\n\t-'img_preview'\n\t-'img_with_bbox'\n\t-'grid_preview'\n\t-'grid_with_bbox'\n\t-'img_to_tensor'\n\t-'tensor_to_img'")
