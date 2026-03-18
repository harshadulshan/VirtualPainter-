import cv2
import numpy as np
import os
import HandTrackingModule as htm

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
HEADER_HEIGHT = 125
BRUSH_THICKNESS = 15
ERASER_THICKNESS = 80
WEBCAM_WIDTH = 1280
WEBCAM_HEIGHT = 720

# ─────────────────────────────────────────
# LOAD HEADER IMAGES
# ─────────────────────────────────────────
folderPath = "Header"
myList = sorted(os.listdir(folderPath))
print("Found header images:", myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}")
    image = cv2.resize(image, (WEBCAM_WIDTH, HEADER_HEIGHT))
    overlayList.append(image)

print(f"Loaded {len(overlayList)} headers")

header = overlayList[0]
drawColor = (203, 192, 255)   # default pink

# ─────────────────────────────────────────
# SETUP WEBCAM
# ─────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(3, WEBCAM_WIDTH)
cap.set(4, WEBCAM_HEIGHT)

# ─────────────────────────────────────────
# SETUP DETECTOR & CANVAS
# ─────────────────────────────────────────
detector = htm.HandDetector(detectionCon=0.85, maxHands=1)
imgCanvas = np.zeros((WEBCAM_HEIGHT, WEBCAM_WIDTH, 3), np.uint8)

xp, yp = 0, 0

# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────
while True:
    # 1. Capture frame
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find hand
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Fingertip positions
        x1, y1 = lmList[8][1], lmList[8][2]    # Index finger
        x2, y2 = lmList[12][1], lmList[12][2]  # Middle finger

        fingers = detector.fingersUp()

        # ── SELECTION MODE (2 fingers up) ──
        if fingers[1] == 1 and fingers[2] == 1:
            xp, yp = 0, 0

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25),
                          drawColor, cv2.FILLED)

            if y1 < HEADER_HEIGHT:
                section_width = WEBCAM_WIDTH // 4

                if 0 < x1 < section_width:
                    header = overlayList[0]
                    drawColor = (203, 192, 255)      # Pink
                elif section_width < x1 < section_width * 2:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)          # Blue
                elif section_width * 2 < x1 < section_width * 3:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)          # Green
                elif section_width * 3 < x1 < section_width * 4:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)            # Eraser

        # ── DRAWING MODE (1 finger up) ──
        if fingers[1] == 1 and fingers[2] == 0:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            thickness = ERASER_THICKNESS if drawColor == (0, 0, 0) else BRUSH_THICKNESS

            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
            xp, yp = x1, y1

    # ─────────────────────────────────────────
    # MERGE CANVAS ONTO WEBCAM
    # ─────────────────────────────────────────
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # ─────────────────────────────────────────
    # OVERLAY HEADER
    # ─────────────────────────────────────────
    img[0:HEADER_HEIGHT, 0:WEBCAM_WIDTH] = header

    # ─────────────────────────────────────────
    # SHOW WINDOW
    # ─────────────────────────────────────────
    cv2.imshow("AI Virtual Painter", img)

    # q = quit | c = clear canvas
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        imgCanvas = np.zeros((WEBCAM_HEIGHT, WEBCAM_WIDTH, 3), np.uint8)

cap.release()
cv2.destroyAllWindows()