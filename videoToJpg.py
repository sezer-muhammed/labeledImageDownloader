import cv2

video = "23-ekm.mp4"
cam = cv2.VideoCapture(video)
i = 0
while True:
    ret, frame = cam.read()
    if ret == 1:
        frame = cv2.rotate(frame, 0)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        cv2.imwrite("videoToJpg/{}.jpg".format(i), frame)
    i = i + 1