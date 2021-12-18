import cv2
import numpy as np
from scipy.spatial import distance

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Cnts", cv2.WINDOW_KEEPRATIO)

# 100 15 170
lower = (80, 5, 150)
upper = (145, 255, 255)

while cam.isOpened():
    _, image = cam.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (11, 11), 0), cv2.COLOR_BGR2HSV)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts1 = cv2.Canny(mask, 150, 170)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    for cnt in cnts:
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        if (h > y * 2 and h > 300) or (y > h * 2 and h > 300):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
            # print(f"x: {x}, y: {y}, w: {w}, h: {h}\n")
            cv2.putText(image, f"consume the french chalise", (int(x + 20), int((y + h) / 2)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0))

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    cv2.imshow("Cnts", cnts1)
    cv2.imshow("Camera", image)
    cv2.imshow("Mask", mask)
cam.release()
cv2.destroyAllWindows()
