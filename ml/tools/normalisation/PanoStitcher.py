
import cv2, os
import imutils
import numpy as np
from ml.tools.normalisation.FeatureMap import FeatureMap
import matplotlib.pyplot as plt
from ml.tools.normalisation.Normalise import Normalise

class PanoStitcher:

    def __init__(self, extractor, matcher):
        self.feature_extractor = extractor # one of 'sift', 'surf', 'brisk', 'orb'
        self.feature_matching = matcher # either 'bf' or 'knn'

    # def stitch(self, frame1, frame2):
    #
    #     # new image max dimensions, basically double the x, y coords
    #     w = frame1.shape[1] + frame2.shape[1]
    #     h = frame1.shape[0] + frame2.shape[0]
    #
    #     drawnMatches, result = Normalise.featureWarpToGold(frame1, frame2, self.feature_extractor, self.feature_matching)
    #
    #     # now paste them together, not completely confident in the maths here
    #     # result[0:frame2.shape[0], 0:frame2.shape[1]] = frame2
    #     result[0:frame1.shape[0], 0:frame1.shape[1]] = frame1
    #
    #     grey = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)  # create grey result to find contours
    #     thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY)[1]  # threshold the grey image
    #
    #     # Find contours
    #     cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     cnts = imutils.grab_contours(cnts)
    #
    #     c = max(cnts, key=cv2.contourArea)  # get the max im area
    #     (x, y, w, h) = cv2.boundingRect(c)  # bounding box
    #     result = result[y:y + h, x:x + w]  # crop result
    #
    #     return result


    def featureStitch(self, frame1, frame2):
        kpA, featA = FeatureMap.detectAndDescribe(frame1, algo=self.feature_extractor)
        kpB, featB = FeatureMap.detectAndDescribe(frame2, algo=self.feature_extractor)

        if self.feature_matching == 'bf':
            matches = FeatureMap.matchKeyPointsBF(featA, featB, algo=self.feature_extractor)
            img3 = cv2.drawMatches(frame1, kpA, frame2, kpB, matches[:100],
                                   None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        elif self.feature_matching == 'knn':
            matches = FeatureMap.matchKeyPointsKNN(featA, featB, ratio=0.75, algo=self.feature_extractor)
            img3 = cv2.drawMatches(frame1, kpA, frame2, kpB, np.random.choice(matches, 100),
                                   None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        (matches, homo, status) = FeatureMap.getHomography(kpA, kpB, matches, reprojThresh=4)

        if any([matches, homo, status]) is None:
            print('no homo :(')

        # img3 shows the feature map between the two supplied images, uncomment below to show
        # plt.imshow(img3)
        # plt.show()

        # new image max dimensions, basically double the x, y coords
        w = frame1.shape[1] + frame2.shape[1]
        h = frame1.shape[0] + frame2.shape[0]

        # warp to match
        # warp is being weird atm,
        result = cv2.warpPerspective(frame2, np.linalg.inv(homo), (w, h))

        # result is now the second frame warped to fit the first frame, uncomment below to show
        # plt.imshow(result)
        # plt.axis('off')
        # plt.show()

        # now paste them together, not completely confident in the maths here
        # result[0:frame2.shape[0], 0:frame2.shape[1]] = frame2
        result[0:frame1.shape[0], 0:frame1.shape[1]] = frame1

        # show combined images before cropping
        # plt.imshow(result)
        # plt.axis('off')
        # plt.show()

        grey = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)  # create grey result to find contours
        thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY)[1]  # threshold the grey image

        # Find contours
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        c = max(cnts, key=cv2.contourArea)  # get the max im area
        (x, y, w, h) = cv2.boundingRect(c)  # bounding box
        result = result[y:y + h, x:x + w]  # crop result

        return result