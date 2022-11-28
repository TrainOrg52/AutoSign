import numpy as np
import cv2

class FeatureMap:

    @staticmethod
    def detectAndDescribe(image, algo=None): # algorithm must == 'sift', 'surf', 'orb' etc

        # detect and extract features from the image
        if algo == 'sift':
            descriptor = cv2.xfeatures2d.SIFT_create()
        elif algo == 'surf':
            descriptor = cv2.xfeatures2d.SURF_create()
        elif algo == 'brisk':
            descriptor = cv2.BRISK_create()
        elif algo == 'orb':
            descriptor = cv2.ORB_create()

        # get keypoints and descriptors
        (kps, features) = descriptor.detectAndCompute(image, None)

        return (kps, features)

    @staticmethod
    def instMatcher(algo, crosscheck):
        if algo == 'sift' or algo == 'surf':
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crosscheck)
        elif algo == 'orb' or algo == 'brisk':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crosscheck)
        return bf

    # todo create function to filter matches by angle from horizontal -> too great means not useful for panorama

    @staticmethod # matches key points using bf matcher
    def matchKeyPointsBF(featA, featB, algo):
        bf = FeatureMap.instMatcher(algo, crosscheck=True)
        matches = bf.match(featA, featB)
        goodMatches = sorted(matches, key=lambda x: x.distance)
        return goodMatches

    @staticmethod # match key points using K nearest neighbours
    def matchKeyPointsKNN(featA, featB, ratio, algo):
        bf = FeatureMap.instMatcher(algo, crosscheck=False)
        rawMatches = bf.match(featA, featB, 2)  # knn with 2
        matches = []

        for m, n in rawMatches:
            if m.distance < n.distance * ratio:
                matches.append(m)

        return matches

    @staticmethod # get the homography matrix between two images
    def getHomography(kpA, kpB, matches, reprojThresh):
        kpA = np.float32([kp.pt for kp in kpA])
        kpB = np.float32([kp.pt for kp in kpB])

        if len(matches) > 4:

            ptA = np.float32([kpA[m.queryIdx] for m in matches])
            ptB = np.float32([kpB[m.trainIdx] for m in matches])

            (H, status) = cv2.findHomography(ptA, ptB, cv2.RANSAC, reprojThresh)

            return (matches, H, status)  # H == homography

        else:
            print('not enough matches :(')
            return


