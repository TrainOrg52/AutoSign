import torch
import numpy as np

from media_handling.video_handling import Videos

# ============================================================================ #
# ATTENTION USER!!!
# You don't have to do anything but choose a mode and watch the magic happen
# All the required code for any operation is within a case statement
"""
The mode options include
   'vid_preview'
   'vid_with_bbox'
   'vid_to_tensor'
   'vid_metadata'
"""
mode = 'vid_to_tensor'
# ============================================================================ #

match mode:
    # Image Previewing
    case 'vid_preview':
        # Note: It should be noted that portrait recordings with phone include black bars on the sides but are of the same size as landscape videos (1920x1080)
        # initialize video handler with video path
        video_handler = Videos('vid/test_video.mp4')
        video_handler.video_preview()

    # Video Previewing with BBox Overlay
    case 'vid_with_bbox':
        video_handler = Videos('vid/test_video.mp4')

        # dummy values
        # edges of phone video = (655, 0, 1264, 1080)
        bboxes = torch.Tensor([[900, 300, 1000, 500], [700, 700, 900, 900]]).expand(234, 2, 4)  # num_frames = 234, num_bboxes = 2, bbox dim=4
        bboxes_with_noise = torch.zeros(234, 2, 4)
        for frame in range(bboxes.size(0)):
            if frame != 0:
                bboxes_with_noise[frame, 0, ...] = bboxes[frame - 1, 0, ...] + np.random.randint(-50, 50)
                bboxes_with_noise[frame, 1, ...] = bboxes[frame - 1, 1, ...] + np.random.randint(-50, 50)
            else:
                bboxes_with_noise[frame, 0, ...] = bboxes[frame - 1, 0, ...]
                bboxes_with_noise[frame, 1, ...] = bboxes[frame - 1, 1, ...]

        labels = [['obj 1', 'obj 2'] for _ in range(234)]
        colours = [['red', 'yellow'] for _ in range(234)]

        video_handler.video_bbox_preview(bboxes_with_noise, labels, colours)

    # Video to PyTorch tensor
    case 'vid_to_tensor':
        # initialize video handler with video path
        video_handler = Videos('vid/test_video.mp4')
        tensor = video_handler.video_to_tensor()
        print(tensor.size())  # PyTorch tensor of [F, C, H, W] representing the video from self.video_src

    # Video Metadata Retrieval
    case 'vid_metadata':
        # Note: It should be noted that portrait recordings with phone include black bars on the sides but are of the same size as landscape videos (1920x1080)
        # initialize video handler with video path
        video_handler = Videos('vid/test_video.mp4')
        video_handler.__metadata__()

    case _:
        print(
            f"Invalid case.\nChoose from the following:\n\t-'vid_preview'\n\t-'vid_with_bbox'\n\t-'vid_to_tensor'\n\t-'vid_metadata'")
