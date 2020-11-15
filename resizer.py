import cv2

input_path = "videoToJpg"
output_path = "videoToJpg"
num = 10000
for i in range(0, num):
    try:
        name = input_path + "/" + str(i) + ".jpg"
        frame = cv2.imread(name)
        roi = frame[80:560, 0:480]
        roi = cv2.resize(roi, (300, 300))
        name = output_path + "/" + str(i) + ".jpg"
        cv2.imshow("roi", roi)
        cv2.waitKey(1)
        cv2.imwrite(name, roi)
    except:
        pass