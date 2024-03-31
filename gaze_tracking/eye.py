import math
import numpy as np
import cv2
from .pupil import Pupil


class Eye(object):
    """
    This class creates a new frame to isolate the eye and
    initiates the pupil detection.
    """

    # Landmark point mata kiri dan mata kanan berdasarkan 68 face_landmarks
    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

    def __init__(self, original_frame, landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None

        self._analyze(original_frame, landmarks, side, calibration)

    @staticmethod
    def _middle_point(p1, p2):
        """Returns the middle point (x,y) between two points

        Arguments:
            p1 (dlib.point): First point
            p2 (dlib.point): Second point
        """
        # Menghitung titik tengah dari coordinate x dan y
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return (x, y)

    def _isolate(self, frame, landmarks, points):
        """Isolate an eye, to have a frame without other part of the face.

        Arguments:
            frame (numpy.ndarray): Frame containing the face
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            points (list): Points of an eye (from the 68 Multi-PIE landmarks)
        """

        # Extract wilayah mata dari frame berdasarkan landmark points, seperti yang kita tau setiap titik landmark memiliki coordinate x dan y, yang menunjukkan posisinya pada gambar
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        region = region.astype(np.int32) 
        self.landmark_points = region # Store landmark point untuk mata saja

        # Applying a mask to get only the eye
        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        # Cropping on the eye
        margin = 5
        # Calculating minimum and maximum x and y coordinate for cropping
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        # Cropping on the eye dari frame berdasarkan nilai minimum dan maximum dari x dan y coordinate
        self.frame = eye[min_y:max_y, min_x:max_x]

        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        
        # Menghitung center dari eye yang sudah di crop dengan tinggi dan lebar dari frame tersebut
        self.center = (width / 2, height / 2)

    def _analyze(self, original_frame, landmarks, side, calibration):
        """Detects and isolates the eye in a new frame, sends data to the calibration
        and initializes Pupil object.

        Arguments:
            original_frame (numpy.ndarray): Frame passed by the user
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            side: Indicates whether it's the left eye (0) or the right eye (1)
            calibration (calibration.Calibration): Manages the binarization threshold value
        """

        # Menentukan landmark point berdasarkan sisi mata 0 (kiri) dan 1 (kanan)
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return

        # Isolate wilayah mata dari frame berdasarkan landmark dan point-point yang menentukan wilayah mata
        self._isolate(original_frame, landmarks, points)

        # Jika proses calibration belum selesai, maka dia akan terus mencari nilai ambang terbaik
        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        # Setelah isolate mata dan calibration selesai, nilai ambang yang diperoleh digunakan untuk menginisialisasi objek Pupil
        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)
