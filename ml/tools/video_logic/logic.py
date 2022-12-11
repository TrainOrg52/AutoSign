import os
import sys
import cv2
import numpy as np
import pandas as pd


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


def sign_presence_logic(identified_signs, bbox_coords):
    # ################### #
    # SIGN PRESENCE LOGIC #
    # ################### #

    debug = False
    nms_diff = 5

    if not debug:
        sys.stdout = open(os.devnull, 'w')  # block printing momentarily

    # remove signs identified outside the window of acceptance, perform "BTEC" NMS and remove once occurring signs
    padding = 10
    for frame_index in range(len(identified_signs)):
        # get the signs in the frame
        frame_signs = identified_signs[frame_index]

        # get the coordinates of the signs in the frame
        frame_coords = bbox_coords[frame_index]

        # ######## #
        # BTEC NMS #
        # ######## #

        # find repeating signs
        repeated_signs = pd.Series(frame_signs)[pd.Series(frame_signs).duplicated()].values
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

            if (x_diff < nms_diff) and (y_diff < nms_diff):
                identified_signs[frame_index].pop(indices[0])

        # ########################### #
        # REMOVE SIGNS OUTSIDE WINDOW #
        # ########################### #

        lag = 0
        for sign_index in range(len(frame_signs)):
            x0 = frame_coords[sign_index][0]
            y0 = frame_coords[sign_index][1]
            x1 = frame_coords[sign_index][2]
            y1 = frame_coords[sign_index][3]

            # pop identified sign if sign is in padded area
            if (x0 < padding) or (y0 < padding) or (x1 > (1280 - padding)) or (y1 > (1280 - padding)):
                identified_signs[frame_index].pop(sign_index - lag)
                lag += 1

        # ########################### #
        # REMOVE ONCE OCCURRING SIGNS #
        # ########################### #

        if frame_index >= 1:
            for sign_index in range(len(frame_signs)):
                current_sign = identified_signs[frame_index][sign_index]
                if current_sign not in identified_signs[frame_index - 1]:
                    if (frame_index < len(identified_signs)) and (
                            current_sign not in identified_signs[frame_index + 1]):
                        identified_signs[frame_index].pop(sign_index)

    # identify signs in video
    print("-" * 20)
    video_signs, prior_signs, prior_coords = [], [], []
    for frame_index in range(len(identified_signs)):
        print(f"Frame: {frame_index}")

        # get the signs in the frame
        frame_signs = identified_signs[frame_index]

        # frame_index = 0 always has new signs
        if frame_index == 0:
            video_signs.extend(frame_signs)
            prior_signs.extend(frame_signs)
            prior_coords.extend(bbox_coords[frame_index])
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

                    print(f"\n\t\t{frame_signs[sign_index]}")
                    print(f"\t\t\tPrior Coord: {prior_bbox_sign}")
                    print(f"\t\t\tCurrent Coord: {frame_bbox_sign}")

                    """
                    # Removed this block because directionless logic works better
                    # if sign is to the opposing side of the bbox movement => new sign to append
                    if ((direction == 'left') and (prior_bbox_sign[0] > frame_bbox_sign[0])) or (
                            (direction == 'right') and (prior_bbox_sign[0] < frame_bbox_sign[0])):
                        video_signs.append(frame_signs[sign_index])
                        print(f"\t\t\tNew sign")
                    else:
                        print(f"\t\t\tPresent sign")
                    """

                    print(f"\t\t\tPresent sign")

                    # pop sign from prior_signs and prior coordinates
                    prior_signs.pop(prior_signs.index(frame_signs[sign_index]))
                    prior_coords.pop(prior_coords.index(prior_bbox_sign))
                # if sign is not in the prior frame and the sign is in the window of acceptance => append sign to list
                else:
                    print(f"\n\t\t{frame_signs[sign_index]}")
                    print(f"\t\t\tNew sign")

                    video_signs.append(frame_signs[sign_index])

            # prior signs update
            prior_signs.clear()
            prior_signs.extend(frame_signs)
            prior_coords.clear()
            prior_coords.extend(bbox_coords[frame_index])

        print("-" * 20)

    sys.stdout = sys.__stdout__  # enable printing

    return video_signs