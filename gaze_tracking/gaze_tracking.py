from __future__ import division
import os
import cv2
import dlib
from .eye import Eye
from .calibration import Calibration


class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        # _face_detector is used to detect faces from dlib library
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face from dlib library and use a model from dlib too    
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        """Detects the face and initialize Eye objects"""
        # Convert frame ke grayscale karena face detector works in grayscale
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        faces = self._face_detector(frame)

        try:
            # Get facial landmark yang terdeteksi pertama oleh frame
            landmarks = self._predictor(frame, faces[0])
            # Initialize left eye (0) object
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            # Initialize right eye (1) object
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

        except IndexError:
            # If no face detected
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        
        # Setting frame dengan frame yang di passing
        self.frame = frame

        # Dan langsung di analisa lagi
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        # Kalau pupil berhasil di deteksi
        if self.pupils_located:
            # Hitung x dan y coordinate dari mata kiri dengan menjumlahkan coordinate x dan y mata kiri asal dengan x dan y coodrinate dari relatif pupil
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    # Sama dengan mata kiri
    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            # Calculate horizontal position of the gaze dengan merata-rata horizontal position dari pupil kiri dan kanan
            # Cara menghitung nilai-nya adalah dengan membagi coordinate dari masing-masing pupil dengan lebar dari mata secara keseluruhan
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    # Mirip dengan horizontal ratio, perbedaannya, hanya di hitung nilai-nya, dengan membagi dengan tinggi mata secara keseluruhan
    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            # Orang akan dianggap melihat ke kanan apabila ratio horizontal nya <= .35 (threshold)
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            # Orang akan dianggap melihat ke kiri apabila ratio horizontal nya >= .65 (threshold)
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            # Orang akan dianggap melihat ke tengah apabila dia tidak dianggap melihat ke kanan atau kiri
            return self.is_right() is not True and self.is_left() is not True

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""

        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            # Get coordinate pupil kiri dan kanan
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            # Nambah garis horizontal dan vertikal di mata kiri dan kanan untuk nandai posisi pupil seperti tanda (+)
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame
