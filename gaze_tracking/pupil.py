import numpy as np
import cv2


class Pupil(object):
    """
    This class detects the iris of an eye and estimates
    the position of the pupil
    """

    def __init__(self, eye_frame, threshold):
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None

        self.detect_iris(eye_frame)

    @staticmethod
    def image_processing(eye_frame, threshold):
        """Performs operations on the eye frame to isolate the iris

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
            threshold (int): Threshold value used to binarize the eye frame

        Returns:
            A frame with a single element representing the iris
        """
        kernel = np.ones((3, 3), np.uint8)

        # Bilateral filter untuk reduce noise tapi tetap mempertahankan edges (tepi mata)
        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        # Menghilangkan titik-titik terang kecil dan meratakan edges (tepi mata)
        new_frame = cv2.erode(new_frame, kernel, iterations=3)
        # Melakukan threshold atau batasan untuk menjadikan biner berdasarkan nilai threshold yang diberikan
        new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]

        return new_frame

    def detect_iris(self, eye_frame):
        """Detects the iris and estimates the position of the iris by
        calculating the centroid.

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
        """
        # Process the eye frame to isolate the iris
        self.iris_frame = self.image_processing(eye_frame, self.threshold)

        # Mencari kontur dalam frame iris from cv2 library untuk mendeteksi iris dari frame yang kita passing
        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

        # Sort kontur berdasarkan luasnya untuk mendapatkan kontur yang paling mungkin mewakili iris
        contours = sorted(contours, key=cv2.contourArea)

        # Menghitung centroid dari kontur kedua terbesar (assumed to be the iris)
        try:
            moments = cv2.moments(contours[-2])
            # Calculate x and y coordinate of the centroid
            self.x = int(moments['m10'] / moments['m00'])
            self.y = int(moments['m01'] / moments['m00'])
        except (IndexError, ZeroDivisionError):
            pass
