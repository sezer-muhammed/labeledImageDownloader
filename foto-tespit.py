import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
import time
import pyautogui
import tensorflow as tf
import threading
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

class ekranlar:
    def __init__(self):
        self.screen = 0

ekran = ekranlar()




def grab_screen(region=None):
    while True:
        hwin = win32gui.GetDesktopWindow()

        if region:
                left,top,x2,y2 = region
                width = x2 - left + 1
                height = y2 - top + 1
        else:
            width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
            height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
            left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
            top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
        
        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (height,width,4)

        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())

        ekran.screen = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
boxx = (704, 284+106, 1216, 796-106)
tred = threading.Thread(target=grab_screen, args=(boxx,))
tred.daemon = True
tred.start()

model_path = "C:/Users/muhammedsezer/Desktop/arac/models/research/object_detection/inference_graph/saved_model"

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = tf.saved_model.load(model_path)
model_fn = model.signatures['serving_default']
#284:796, 704:1216
def run_inference_for_single_image(model, image):
    #image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]
    
    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict

while True:
    if True:
        #frame = ekran.screen
        frame = ekran.screen
        frame = cv2.resize(frame, (300, 300))
        for i in range(4):
            if i == 0:
                roi = frame[0:150, 0:150]
            if i == 1:
                roi = frame[150:300, 150:300]
            if i == 2:
                roi = frame[0:150, 150:300]
            if i == 3:
                roi = frame[150:300, 0:150]
            
            output = run_inference_for_single_image(model_fn, roi)
            if output["detection_multiclass_scores"][0][1] > 0.55 and i == 0:
                box = output["detection_boxes"][0]
                (topY, topX, bottomY, bottomX) = box
                topY = int(topY * 150)
                topX = int(topX * 150)
                bottomY = int(bottomY * 150)
                bottomX = int(bottomX * 150)
                cv2.rectangle(roi, (topX, topY), (bottomX, bottomY), (245, 255, 0), 4)
            if output["detection_multiclass_scores"][0][2] > 0.55 and i == 2:
                box = output["detection_boxes"][0]
                (topY, topX, bottomY, bottomX) = box
                topY = int(topY * 150)
                topX = int(topX * 150)
                bottomY = int(bottomY * 150)
                bottomX = int(bottomX * 150)
                cv2.rectangle(roi, (topX, topY), (bottomX, bottomY), (0, 255, 25), 4)
            if output["detection_multiclass_scores"][0][3] > 0.55 and i == 3:
                box = output["detection_boxes"][0]
                (topY, topX, bottomY, bottomX) = box
                topY = int(topY * 150)
                topX = int(topX * 150)
                bottomY = int(bottomY * 150)
                bottomX = int(bottomX * 150)
                cv2.rectangle(roi, (topX, topY), (bottomX, bottomY), (215, 0, 255), 4)
            if output["detection_multiclass_scores"][0][4] > 0.55 and i == 1:
                box = output["detection_boxes"][0]
                (topY, topX, bottomY, bottomX) = box
                topY = int(topY * 150)
                topX = int(topX * 150)
                bottomY = int(bottomY * 150)
                bottomX = int(bottomX * 150)
                cv2.rectangle(roi, (topX, topY), (bottomX, bottomY), (250, 20, 0), 4)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)


