"""
Hand Invert Camera
------------------
Opens your webcam, finds your hand, and inverts the colours (like a photo
negative) inside an area that is taken from your hand.

Three ways to pick the area (press M to switch):
  1. finger_box  -> a rectangle stretched between your thumb tip and index tip
  2. hand_box    -> a rectangle around your whole hand
  3. hand_shape  -> the exact outline of your hand

Keys:  M = change mode     Q = quit

Install once:   pip install opencv-python mediapipe numpy
Run:            python hand_invert_camera.py

The first run downloads a small hand-detection model file (about 8 MB)
next to this script, so you need internet the first time only.
"""

import os
import time
import urllib.request

import cv2                                   # OpenCV: camera, drawing, image operations
import mediapipe as mp                       # MediaPipe: finds the hand and its 21 landmarks
import numpy as np                           # NumPy: fast maths on image arrays
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision    # the hand-finding part of MediaPipe

# Where the hand model lives. It is saved in the same folder as this script.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# The three ways we can pick the inverted area. Pressing M cycles through them.
MODES = ["finger_box", "hand_box", "hand_shape"]

# MediaPipe landmark numbers we care about (each hand has 21 points, 0-20).
THUMB_TIP = 4
INDEX_TIP = 8


def ensure_model():
    """Download the hand model the first time, if we don't have it yet."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand model (one time only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def landmarks_to_pixels(hand, width, height):
    """
    MediaPipe gives each point as a fraction of the image (0.0 to 1.0).
    Multiply by the image width/height to get real pixel positions.
    'hand' is the list of 21 points for one hand.
    Returns a list of 21 (x, y) tuples.
    """
    return [(int(p.x * width), int(p.y * height)) for p in hand]


def build_mask(mode, hands_points, frame_shape):
    """
    Build a black-and-white "mask" image the same size as the camera frame.
    White (255) = pixels that will be inverted, black (0) = left alone.
    """
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)  # start all black

    for points in hands_points:  # loop in case two hands are visible
        if mode == "finger_box":
            # Rectangle whose opposite corners are thumb tip and index tip.
            cv2.rectangle(mask, points[THUMB_TIP], points[INDEX_TIP], 255, -1)

        elif mode == "hand_box":
            # boundingRect finds the smallest upright rectangle that
            # contains all 21 points of the hand.
            x, y, w, h = cv2.boundingRect(np.array(points))
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        else:  # "hand_shape"
            # convexHull wraps a tight outline around all the points,
            # like a rubber band stretched around the hand.
            hull = cv2.convexHull(np.array(points))
            cv2.fillConvexPoly(mask, hull, 255)

    # thickness -1 above means "fill the shape", so the mask is solid white
    return mask


def apply_invert(frame, mask):
    """
    Invert the whole frame, then keep the inverted pixels only where the
    mask is white. Everywhere else, keep the original pixels.
    """
    inverted = cv2.bitwise_not(frame)  # every pixel becomes 255 - pixel
    # mask[..., None] turns the 2D mask into 3D so it matches the colour image
    return np.where(mask[..., None] > 0, inverted, frame)


def main():
    ensure_model()

    cap = cv2.VideoCapture(0)  # 0 = your default webcam
    if not cap.isOpened():
        print("Could not open the camera. Is another app using it?")
        return

    # Settings for MediaPipe's hand finder.
    # VIDEO mode = "I will send you frames one after another", which lets
    # it track the hand smoothly instead of searching from scratch each time.
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    mode_index = 0          # which entry of MODES we are using right now
    start_time = time.time()
    last_timestamp = -1     # MediaPipe needs timestamps that always go up

    # The "with" closes the hand finder neatly when we're done.
    with vision.HandLandmarker.create_from_options(options) as landmarker:

        while True:
            ok, frame = cap.read()  # grab one picture from the camera
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror it so it feels natural
            height, width = frame.shape[:2]

            # MediaPipe expects RGB colours, OpenCV gives BGR, so convert,
            # then wrap the picture in MediaPipe's own image type.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Timestamp in milliseconds; must be bigger than the last one.
            timestamp = max(int((time.time() - start_time) * 1000), last_timestamp + 1)
            last_timestamp = timestamp

            result = landmarker.detect_for_video(mp_image, timestamp)

            output = frame
            hand_found = len(result.hand_landmarks) > 0
            if hand_found:
                # Turn every detected hand into a list of pixel positions
                hands_points = [
                    landmarks_to_pixels(hand, width, height)
                    for hand in result.hand_landmarks
                ]
                mask = build_mask(MODES[mode_index], hands_points, frame.shape)
                output = apply_invert(frame, mask)

            # Show the active mode. OpenCV drawing needs a plain, contiguous
            # uint8 image, so make sure that's what we have.
            output = np.ascontiguousarray(output, dtype=np.uint8)
            status = "hand found" if hand_found else "no hand"
            cv2.putText(
                output, f"Mode: {MODES[mode_index]} | {status} | M = change, Q = quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

            cv2.imshow("Hand Invert Camera", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                mode_index = (mode_index + 1) % len(MODES)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
