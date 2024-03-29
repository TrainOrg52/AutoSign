import os
import sys
import cv2
import numpy as np
import pandas as pd
from skimage import transform


# ########################## #
# VIDEO PROCESSING FUNCTIONS #
# ########################## #

def video_decorator(func):
    def wrapper(*args):
        increment = 0
        video, frame_num, count, frame_root, total_num_frames = args
        while (video.isOpened()):
            ret, frame_num = func(video, frame_num, frame_root)

            increment += count
            video.set(cv2.CAP_PROP_POS_FRAMES, increment)

            if not ret:
                break

            if increment > total_num_frames:
                break

        # When everything done, release the video capture object
        video.release()

        # Closes all the frames
        cv2.destroyAllWindows()

        return frame_num

    return wrapper


@video_decorator
def video_processing(video, frame_num, frame_root):
    # Capture each frame
    ret, frame = video.read()
    if ret:
        cv2.imwrite(f'{frame_root}/{frame_num}.png',
                    cv2.resize(frame, (1280, 1280), interpolation=cv2.INTER_AREA))
        frame_num += 1

        return ret, frame_num
    else:
        return ret, frame_num


def perform_homography(coord, frame_image, num_signs, path):
    x1, y1 = coord[0], coord[1]
    x2, y2 = coord[2], coord[3]
    src = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]).reshape((4, 2))
    dst = np.array([[0, 0], [0, frame_image.shape[1]], [frame_image.shape[0], frame_image.shape[1]], [frame_image.shape[0], 0]]).reshape((4, 2))

    tform = transform.estimate_transform('projective', src, dst)
    tf_img = transform.warp(frame_image, tform.inverse) * 255

    # plotting the transformed image
    cv2.imwrite(path, tf_img)

# ####################### #
# SIGN PRESENCE FUNCTIONS #
# ####################### #

def sign_filter(identified_signs, bbox_coords, nms_diff, padding):
    for frame_index in range(len(identified_signs)):
        # get the signs in the frame
        frame_signs = identified_signs[frame_index]

        # get the coordinates of the signs in the frame
        frame_coords = bbox_coords[frame_index]

        # ######## #
        # BTEC NMS #
        # ######## #

        # find repeating signs
        repeated_signs = pd.Series(frame_signs, dtype=pd.StringDtype())[pd.Series(frame_signs, dtype=pd.StringDtype()).duplicated()].values
        if repeated_signs.size > 0:
            repeated_signs = repeated_signs[0]

            # get index of repeating sign
            indices = []
            for idx, value in enumerate(frame_signs):
                if value == repeated_signs:
                    indices.append(idx)

            x_coords, y_coords = [], []
            for indice in indices:
                x_coords.append(frame_coords[indice][0])
                y_coords.append(frame_coords[indice][1])

            x_diff = np.diff(x_coords)
            y_diff = np.diff(y_coords)

            print(x_diff, y_diff, flush=True)

            print(f"NMS Diff: {nms_diff}\ty_diff: {y_diff}\tx_diff: {x_diff}")
            if (np.all(x_diff) < nms_diff) and (np.all(y_diff) < nms_diff):
                identified_signs[frame_index].pop(indices[0])
                bbox_coords[frame_index].pop(indices[0])

        # ########################### #
        # REMOVE SIGNS OUTSIDE WINDOW #
        # ########################### #

        lag = 0
        for sign_index in range(len(frame_signs)):
            x0 = frame_coords[sign_index - lag][0]
            y0 = frame_coords[sign_index - lag][1]
            x1 = frame_coords[sign_index - lag][2]
            y1 = frame_coords[sign_index - lag][3]

            # pop identified sign if sign is in padded area
            """
            if (x0 < padding) or (y0 < padding) or (x1 > (1280 - padding)) or (
                    y1 > (1280 - padding)):
            """
            if (x0 < padding) or (x1 > (1280 - padding)):
                identified_signs[frame_index].pop(sign_index - lag)
                bbox_coords[frame_index].pop(sign_index - lag)
                lag += 1

        # ########################### #
        # REMOVE ONCE OCCURRING SIGNS #
        # ########################### #

        if frame_index >= 1:
            lag = 0
            for sign_index in range(len(identified_signs[frame_index])):
                current_sign = identified_signs[frame_index][sign_index - lag]
                if (current_sign not in identified_signs[frame_index - 1]) and (frame_index < len(identified_signs)-1):  # if current sign is not in prior frame
                    if current_sign not in identified_signs[frame_index + 1]:  # if frame index is less than the total number of frames
                        identified_signs[frame_index].pop(sign_index - lag)
                        bbox_coords[frame_index].pop(sign_index - lag)
                        lag += 1

    return identified_signs, bbox_coords


def sign_logic(identified_signs, bbox_coords, images_root, image_size):
    # ###################### #
    # GET MOVEMENT DIRECTION #
    # ###################### #

    frame_index, reoccurring_sign_found = 0, False
    while not reoccurring_sign_found:
        # we find that the general movement of the camera occurs after a few frames
        if (frame_index < len(identified_signs) - 1) and (frame_index > 2):
            for sign_index in range(len(identified_signs[frame_index])):
                if identified_signs[frame_index][sign_index] in identified_signs[frame_index+1]:
                    reoccurring_sign_found = True
                    reoccurring_sign = identified_signs[frame_index][sign_index]
                    reoccurring_coord = bbox_coords[frame_index][sign_index]
                    frame_init = frame_index
                    break
        elif frame_index > len(identified_signs) - 1:
            break

        frame_index += 1

    reoccurring_pos = identified_signs[frame_init+1].index(reoccurring_sign)
    next_reoccurring_coord = bbox_coords[frame_init+1][reoccurring_pos]

    if reoccurring_coord < next_reoccurring_coord:
        movement_direction = 'left'
    else:
        movement_direction = 'right'

    print(f"Movement Direction: {movement_direction}")

    # ################### #
    # SIGN PRESENCE LOGIC #
    # ################### #

    print("-" * 50)
    # set movement buffer to a third the frame size
    movement_buffer = image_size / 3
    video_signs, prior_signs, prior_coords = [], [], []
    num_signs = 0
    for frame_index in range(len(identified_signs)):
        print(f"Frame: {frame_index}")

        # get the signs in the frame
        frame_signs = identified_signs[frame_index]
        frame_coord = bbox_coords[frame_index]

        # frame_index = 0 always has new signs
        if frame_index == 0:
            video_signs.extend(frame_signs)
            prior_signs.extend(frame_signs)
            prior_coords.extend(frame_coord)

            # save each sign to normalized signs folder
            frame_image = cv2.imread(f"{images_root}/{frame_index}.png")

            for signs, coord in zip(frame_signs, frame_coord):
                perform_homography(coord, frame_image, num_signs, f"samples/normalized_images/{num_signs}.png")
                num_signs += 1
        else:
            # debugging
            print(f"Video Signs: {video_signs}")
            print(f"Prior: {prior_signs}\nCurrent: {frame_signs}")

            # if sign is to the opposing side of the bbox movement => new sign to append
            for sign_index in range(len(frame_signs)):
                # check to see if sign is in prior frame
                if frame_signs[sign_index] in prior_signs:
                    # get current frame and prior frame bbox positions
                    frame_bbox_sign = bbox_coords[frame_index][sign_index]
                    prior_bbox_sign = prior_coords[prior_signs.index(frame_signs[sign_index])]

                    # movement logic (used in case a new sign enters as soon as another sign leaves the frame)
                    if ((movement_direction == 'left') and (prior_bbox_sign[0] < frame_bbox_sign[0] + movement_buffer)) or \
                            ((movement_direction == 'right') and (prior_bbox_sign[0] > frame_bbox_sign[0] - movement_buffer)):
                        print(f"\n\t\t{frame_signs[sign_index]}")
                        print(f"\t\t\tPrior Coord: {prior_bbox_sign}")
                        print(f"\t\t\tCurrent Coord: {frame_bbox_sign}")

                        print(f"\t\t\tPresent sign")

                        # pop sign from prior_signs and prior coordinates
                        prior_signs.pop(prior_signs.index(frame_signs[sign_index]))
                        prior_coords.pop(prior_coords.index(prior_bbox_sign))
                    else:
                        print(f"\n\t\t{frame_signs[sign_index]}")
                        print(f"\t\t\tPrior Coord: {prior_bbox_sign}")
                        print(f"\t\t\tCurrent Coord: {frame_bbox_sign}")
                        print(f"\t\t\t**Movement Captured** New sign")

                        video_signs.append(frame_signs[sign_index])

                        # save each sign to normalized signs folder
                        frame_image = cv2.imread(f"{images_root}/{frame_index}.png")

                        perform_homography(bbox_coords[frame_index][sign_index], frame_image, num_signs, f"samples/normalized_images/{num_signs}.png")
                        num_signs += 1
                # if sign is not in the prior frame and the sign is in the window of acceptance => append sign to list
                else:
                    print(f"\n\t\t{frame_signs[sign_index]}")
                    print(f"\t\t\tNew sign")

                    video_signs.append(frame_signs[sign_index])

                    # save each sign to normalized signs folder
                    frame_image = cv2.imread(f"{images_root}/{frame_index}.png")

                    perform_homography(bbox_coords[frame_index][sign_index], frame_image, num_signs, f"samples/normalized_images/{num_signs}.png")
                    num_signs += 1

            # prior signs update
            prior_signs.clear()
            prior_signs.extend(frame_signs)
            prior_coords.clear()
            prior_coords.extend(bbox_coords[frame_index])

        print("-" * 50)

    return video_signs


class Sign_Presence:
    def __init__(self, nms_diff, padding, debug=False):
        # ########################## #
        # INITIALIZE HYPERPARAMETERS #
        # ########################## #

        self.nms_diff = nms_diff
        self.padding = padding
        self.debug = debug
        self.image_size = 1280

    def sign_presence_logic(self, identified_signs, bbox_coords, images_root):
        # ################### #
        # SIGN PRESENCE LOGIC #
        # ################### #

        # only print to console if in debug mode
        if not self.debug:
            sys.stdout = open(os.devnull, 'w')  # block printing momentarily

        # remove signs identified outside the window of acceptance, perform "BTEC" NMS and remove once occurring signs
        filtered_signs, filtered_boxes = sign_filter(identified_signs, bbox_coords, self.nms_diff, self.padding)

        # ####################### #
        # IDENTIFY SIGNS IN VIDEO #
        # ####################### #

        video_signs = sign_logic(filtered_signs, filtered_boxes, images_root, self.image_size)

        # #################### #
        # UNCACHE LOCAL IMAGES #
        # #################### #

        for filename in os.listdir(images_root):
            file_path = os.path.join(images_root, filename)
            os.remove(file_path)

        sys.stdout = sys.__stdout__  # enable printing

        return video_signs
