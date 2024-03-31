# Importing library
import cv2 # For access webcam and image visualisation
import mediapipe as mp # Face mesh model and drawing utility
import numpy as np # Array and matrix
from gaze_tracking import GazeTracking # Eye gaze

def main():
    # Initialize Face Mesh Model, Drawing and Landmark Detection
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Initialize Gaze Tracking
    gaze = GazeTracking()

    # Load the camera
    webcam = cv2.VideoCapture(0)

    # Define initial color for the text
    color = (77, 255, 0)

    # Counter for tracking sus behavior
    counter = 0

    while True:
        #get the frame (but it's only 1 frame, so we need to loop it).
        ret, frame = webcam.read()

        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB because image processing by opencv (process by bgr format) and mediapipe (process by rgb format).
        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        frame_rgb.flags.writeable = False

        # Get the result from Face Mesh and processing the facial landmark
        results = face_mesh.process(frame_rgb)

        # To improve performance
        frame_rgb.flags.writeable = True

        # Convert the color space from RGB to BGR
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Getting the dimension from the frame by height, width, and channels.
        img_h, img_w, img_c = frame.shape 

        # List for save the coordinate from the facial landmark from 3d and 2d
        face_3d = []
        face_2d = []

        text = ""
        sus = ""

        if results.multi_face_landmarks:
            # Other library have coordinates, while mediapipe not. We need to loop through every different point of the face (return x,y,z coordinates).
            for face_landmarks in results.multi_face_landmarks:
                # Check every landmark at the face that processed and take the certain landmark that detect the face, like nose, and etc.
                for idx, lm in enumerate(face_landmarks.landmark):
                    # Calculate every landmark and save the coordinate in 2D and 3D format
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])       

                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix to convert from 3D to 2D to calculate position from the head
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                # The distortion parameters for correcting inaccuracy geometry
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP is to predict a pose (position and orientation) from 3D object, by some point in 2D. Returned a rotation vector (orientation) and translation vector (position).
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix from the rotation vector. This rotational matrix describes how objects are oriented in 3D
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles from the rotational matrix (radian).
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                # See where the user's head tilting based on the value of the rotation angle along the y and x axis.
                if y < -10:
                    text = "Looking Left"
                    color = (0, 0, 255)
                    counter += 1
                elif y > 10:
                    text = "Looking Right"
                    color = (0, 0, 255)
                    counter += 1
                elif x < -10:
                    text = "Looking Down"
                elif x > 10:
                    text = "Looking Up"
                else:
                    # If head is facing forward, use eye gaze detection
                    gaze.refresh(frame)
                    if counter < 50:
                        if gaze.is_right(): 
                            text = "You're looking left."
                            color = (0, 0, 255)
                            counter += 1
                        elif gaze.is_left():
                            text = "You're looking right."
                            color = (0, 0, 255)
                            counter += 1
                        elif gaze.is_center():
                            text = "Keep focus at the exam."
                            color = (77, 255, 0)

                if counter >= 50:
                    text = "Suspicious behavior detected..."
                    color = (0, 0, 255)

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=None)

        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        cv2.putText(frame, sus, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

        # Display the video we loaded before
        cv2.imshow('Exam Proctoring', frame)
        
        # Stop the loop by pressing q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()