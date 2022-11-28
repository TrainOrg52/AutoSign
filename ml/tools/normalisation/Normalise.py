import os
import cv2
import numpy as np
from ml.tools.normalisation.FeatureMap import FeatureMap

class Normalise:

    # unused in this class, just for reference
    absolutePath = os.path.dirname(__file__)
    goldRelPath = '../../img/golds'
    goldPath = os.path.join(absolutePath, goldRelPath)


    @staticmethod
    def featureWarpToGold(query, gold, extractor=None, matcher=None):  # query, gold are cv2 image objs
        # extractor =  one of 'sift', 'surf', 'brisk', 'orb'
        # matcher = either 'bf' or 'knn'

        if extractor is None:
            extractor = 'sift'
        if matcher is None:
            matcher = 'bf'

        kpQ, featQ = FeatureMap.detectAndDescribe(query, algo=extractor)
        kpG, featG = FeatureMap.detectAndDescribe(gold, algo=extractor)

        if matcher == 'bf':
            matches = FeatureMap.matchKeyPointsBF(featQ, featG, algo=extractor)
            img3 = cv2.drawMatches(query, kpQ, gold, kpG, matches[:100],
                                   None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        elif matcher == 'knn':
            matches = FeatureMap.matchKeyPointsKNN(featQ, featG, ratio=0.75, algo=extractor)
            img3 = cv2.drawMatches(query, kpQ, gold, kpG, np.random.choice(matches, 100),
                                   None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        (matches, homo, status) = FeatureMap.getHomography(kpQ, kpG, featQ, featG, matches, reprojThresh=4)

        if any([matches, homo, status]) is None:
            print('no homo :(')

        h, w = gold.shape
        warped = cv2.warpPerspective(query, np.linalg.inv(homo), (w, h))
        return img3, warped # matches drawn over both images, then warped query image

    @staticmethod
    def warpToGold(query : str, gold : str): # path, path
        query = cv2.imread(query)
        gold = cv2.imread(gold)

        qh, qw = query.shape
        gh, gw = gold.shape

        # top-left, top-right, bottom-right, bottom-left
        queryCorners = np.array([[0, 0], [qw, 0], [qw, qh], [0, qh]], np.int32)
        goldCorners = np.array([[0, 0], [gw, 0], [gw, gh], [0, gh]], np.int32)

        queryPts = np.float32(queryCorners)
        goldPts = np.float32(goldCorners)

        # get homography between two images and warp query accordingly
        M = cv2.getPerspectiveTransform(queryPts, goldPts)
        warped = cv2.warpPerspective(query, M, (gw, gh))
        return warped

#     @staticmethod
#     def featureWarpToGold(query, gold, featureAlgo = None): # query, gold are cv2 image objs
#         # ensure grayscale for feature match
#         if len(query.shape) == 3:
#             query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
#         if len(gold.shape) == 3:
#             gold = cv2.cvtColor(gold, cv2.COLOR_BGR2GRAY)
#
#         if featureAlgo.lower() == 'sift':
#             sift = cv2.SIFT_create()
#
#             queryKP, queryFeat = sift.detectAndCompute(query, None)
#             goldKP, goldFeat = sift.detectAndCompute(gold, None)
#
#             queryKP = np.float32([kp.pt for kp in queryKP])
#             goldKP = np.float32([kp.pt for kp in goldKP])
#
#         elif featureAlgo.lower() == 'orb':
#             orb = cv2.ORB_create()
#             queryKP, queryDesc = orb.detectAndCompute(query, None)
#             goldKP, goldDesc = orb.detectAndCompute(gold, None)
#
#         else:
#             orb = cv2.ORB_create()
#             queryKP, queryDesc = orb.detectAndCompute(query, None)
#             goldKP, goldDesc = orb.detectAndCompute(gold, None)
#
#         # Below are 3 different feature matching algorithms, each subtly different
#
#         # matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
#         # matches = matcher.match(queryDesc, goldDesc, None)
#
#         # matcher = cv2.BFMatcher_create()
#         # matches = matcher.knnMatch(queryDesc, goldDesc, k=2)
#
#         matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING2, crossCheck=False)
#         matches = matcher.match(queryDesc, goldDesc)
#
#         goodMatches = matchGoodPoints(matches, 0.6)
#
#         # image showing the matches between images
#         matchesImg = cv2.drawMatchesKnn(query, queryKP, gold, goldKP, goodMatches, None,
#                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#
#         queryPts = np.float32([queryKP[m.queryIdx]
#                               .pt for m in goodMatches]).reshape(-1, 1, 2)
#
#         goldPts = np.float32([goldKP[m.trainIdx]
#                              .pt for m in goodMatches]).reshape(-1, 1, 2)
#
#         matrix, mask = cv2.findHomography(queryPts, goldPts, cv2.RANSAC)
#         h, w = gold.shape
#         warped = cv2.warpPerspective(query, matrix, (w, h))
#
#         return matchesImg, warped
#
#
# # matches is output of matcher algo, distanceP is percentage distance between accepted points, embedded True for knn matcher
# def matchGoodPoints(matches, distanceP, embedded=False):
#
#     if embedded:
#         matches = [item for sublist in matches for item in sublist]
#     matches = list(matches)
#     matches.sort(key=lambda x: x.distance, reverse=False)
#
#     good = []
#     for m, n in matches:
#         if m.distance < distanceP * n.distance:
#             good.append(m)
#
#     return matches