import cv2
import mediapipe as mp
import os

class HandDetector:
    def __init__(self, maxHands=1, detectionCon=0.85, trackCon=0.8):
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmList = []
        self.results = None

        # Hand connections for drawing
        self.connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (9,10),(10,11),(11,12),
            (13,14),(14,15),(15,16),
            (17,18),(18,19),(19,20),
            (0,17),(5,9),(9,13),(13,17)
        ]

        # New Tasks API
        model_path = 'hand_landmarker.task'
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=maxHands,
            min_hand_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        self.results = self.detector.detect(mp_image)

        if draw and self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                h, w, _ = img.shape
                points = []
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points.append((cx, cy))
                    cv2.circle(img, (cx, cy), 6, (255, 0, 255), cv2.FILLED)
                for start, end in self.connections:
                    cv2.line(img, points[start], points[end], (0, 255, 0), 2)
        return img

    def findPosition(self, img, handNo=0, draw=False):
        self.lmList = []
        if self.results and self.results.hand_landmarks:
            if handNo < len(self.results.hand_landmarks):
                hand_landmarks = self.results.hand_landmarks[handNo]
                h, w, _ = img.shape
                for id, lm in enumerate(hand_landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        if not self.lmList:
            return [0, 0, 0, 0, 0]

        fingers = []

        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers
