import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
import easyocr
import os
import time

def ANPR(image):
    # Convert to gray image
    image = imutils.resize(image, width=640, height=480)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 13, 25, 25)
    # Detect edges
    edged = cv2.Canny(bfilter, 30, 200)
    #plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
    #plt.show()
    # Find contours
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]


    # Loop through contours to detect the number plate
    location = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018*peri, True)
        if len(approx) == 4:
            location = approx
            break

    # Apply mask
    if location is None:
        return

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(image, image, mask=mask)

    # Crop number plate
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = image[x1:x2, y1:y2]
    #plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    #plt.show()
    # Use EasyOCR to read number plate
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    result = reader.readtext(cropped_image)
    final_text = ""
    for item in result:
        final_text += item[1]
    print("Detected number plate:",final_text.replace(" ", ""))
    print("Car plate detection is available in result.txt!!")
    with open('result.txt', 'a') as f:
        if final_text != "":
            f.write(final_text.replace(" ", ""))
            f.write("\n")

def readFromImage(imageName):
    img = cv2.imread('Car Images/' + imageName)
    ANPR(img)

def readFromVideo(videoName):
    vidcap = cv2.VideoCapture('Car Images/' + videoName)
    success, image = vidcap.read()
    count = 0
    while success:
        ANPR(image)
        success, image = vidcap.read()
        count += 1

option = input("Choose image or video:")
if option == "image":
    with open('result.txt', 'w') as f:
        f.write("")
    print("Possible images:")
    for x in os.listdir("Car Images"):
        if x.endswith(".jpg") or x.endswith(".jpeg"):
            print(x, end="; ")
    print("\n")
    imageName = input("Please enter image name+extension:")
    print("Car plate is being detected. Please wait!\n")
    readFromImage(imageName)

elif option == "video":
    with open('result.txt', 'w') as f:
        f.write("")
    print("Possible videos:")
    for x in os.listdir("Car Images"):
        if x.endswith(".mp4"):
            print(x, end="; ")
    print("\n")
    videoName = input("Please enter video name+extension:")
    print("Car plate is being detected. Please wait!\n")
    readFromVideo(videoName)
else:
    print("Option not available!")
time.sleep(3)

