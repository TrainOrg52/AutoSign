import cv2
import torch
import numpy as np
from matplotlib.colors import to_rgba as RGB

import torchvision.transforms as transforms


def video_viewer(func):
    def wrapper(*args):
        video, bboxes, labels, colours = func(*args)
        # display video (can close the window if 'q' is pressed)
        frame_index = 0
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                if bboxes is not None:
                    for bbox, label, colour in zip(bboxes[frame_index], labels[frame_index], colours[frame_index]):
                        colour = RGB(colour)
                        colour = [c * 255 for c in colour]
                        colour[0], colour[2] = colour[2], colour[0]  # colour is in rgb, opencv takes bgr

                        x1, y1, x2, y2 = tuple(bbox)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colour, 1)
                        cv2.putText(frame, label, (int(x1) + 4, int(y1) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1,
                                    cv2.LINE_AA)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            video.release()
                            cv2.destroyAllWindows()
                            break

                    cv2.imshow('frame', frame)
                    frame_index += 1
                else:
                    cv2.imshow('frame', frame)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        video.release()
                        cv2.destroyAllWindows()
                        break
            else:
                video.release()
                cv2.destroyAllWindows()

    return wrapper


class Videos:
    """
    Functionality
    ---------------
    Video media handler. Contains methods for:
        .Video Previewing
        .Video Previewing with BBox Overlay
        .Video to PyTorch tensor transformations
        .Video Metadata Retrieval

    Creator/s: Ben Sanati
    """

    def __init__(self, video_src, video_dst=''):
        self.video = cv2.VideoCapture(video_src, apiPreference=cv2.CAP_MSMF)
        self.video_src = video_src
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.num_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))

        self.video_dst = video_dst

        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    @video_viewer
    def video_preview(self):
        """
        Preview video without any processing performed

        :return: Video displayed
        """
        return self.video, None, None, None

    @video_viewer
    def video_bbox_preview(self, bboxes, labels, colours):
        """
        Preview video with bbox overlayed onto the video
        :param bboxes: A PyTorch Tensor in (x1, y1, x2, y2) bbox format of dimension [# frames, # boxes, 4]
        :param labels: A list of [len = # frames], containing the labels corresponding to each bbox in each frame
        :param colours: A list of [len = # frames], containing the colours corresponding to each bbox in each frame

        :return: Video displayed with bbox overlay
        """
        return self.video, bboxes, labels, colours

    def video_to_tensor(self):
        """
        Video to PyTorch tensor
        :return: A PyTorch tensor of [F, C, H, W] representing the video from self.video_src
        """
        frames = []
        for _ in range(self.num_frames):
            ret, frame = self.video.read()  # read one frame from the 'capture' object; img is (H, W, C)
            if ret:
                frames.append(frame)
        video = np.stack(frames, axis=0)  # dimensions (F, H, W, C)
        video_tensor = torch.Tensor(video).permute(0, 3, 1, 2)  # dimensions (F, C, H, W)

        return video_tensor

    def __metadata__(self):
        """
        Video Metadata Retrieval
        :return: Video metadata as tuple (num frames, video fps, video width, video height)
        """
        print(
            f"# of Frames = {self.num_frames}\nVideo FPS: {self.fps}\nVideo Dimensions: ({self.width}, {self.height})")
        return self.num_frames, self.fps, self.width, self.height
