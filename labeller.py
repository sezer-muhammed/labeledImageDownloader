import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
import time
import pyautogui
import tensorflow as tf
import threading
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
import glob

txtfiles = []
for file in glob.glob("ssdnet_all_sezers_bag/*.jpg"):
    txtfiles.append(file)

model_path = "C:/Users/muhammedsezer/Desktop/final-23-ekm/saved_model" #922423 ekm #9416 1ekm - amaa 23 ekm daha çok doğru bildi herhalde oran olarak
#conf = 0.75
physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = tf.saved_model.load(model_path)
model_fn = model.signatures['serving_default']

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
sayac = 0
total = 0
for name in txtfiles:
    dosya, foto = name.split("\\")
    isim, uzanti = foto.split(".")
    if True:
        total = total + 1
        frame = cv2.imread(name)
        frame = cv2.resize(frame, (300, 300))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = run_inference_for_single_image(model_fn, frame)
        if output["detection_multiclass_scores"][0][1] > .5:
            sayac = sayac + 1
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            box = output["detection_boxes"][0]
            (topY, topX, bottomY, bottomX) = box
            topY = int(topY * 300)
            topX = int(topX * 300)
            bottomY = int(bottomY * 300)
            bottomX = int(bottomX * 300)
            cv2.rectangle(frame, (topX, topY), (bottomX, bottomY), (90, 150, 25), 8)
            print("ratio:", sayac/total, "total images:", total)
            """
            file = open("testttt/"+isim+".xml", "w") 
            file.write("<annotation>\n")
            file.write("	<folder>testttt</folder>\n")
            file.write("	<filename>"+foto+"</filename>\n")
            file.write("	<path>C:/Users/muhammedsezer/Desktop/kendiKodlarim/testttt/"+foto+"</path>\n")
            file.write("	<source>\n")
            file.write("		<database>Unknown</database>\n")
            file.write("	</source>\n")
            file.write("	<size>\n")
            file.write("		<width>300</width>\n")
            file.write("		<height>300</height>\n")
            file.write("		<depth>3</depth>\n")
            file.write("	</size>\n")
            file.write("	<segmented>0</segmented>\n")
            file.write("	<object>\n")
            file.write("		<name>bag</name>\n")
            file.write("		<pose>Unspecified</pose>\n")
            file.write("		<truncated>0</truncated>\n")
            file.write("		<difficult>0</difficult>\n")
            file.write("		<bndbox>\n")
            file.write("			<xmin>{}</xmin>\n".format(topX))
            file.write("			<ymin>{}</ymin>\n".format(topY))
            file.write("			<xmax>{}</xmax>\n".format(bottomX))
            file.write("			<ymax>{}</ymax>\n".format(bottomY))
            file.write("		</bndbox>\n")
            file.write("	</object>\n")
            file.write("</annotation>\n")
            file.close()
            """
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            print("ratio:", sayac/total, "total images:", total)
            #cv2.imwrite("csfoto/"+foto, frame)
        frame = cv2.resize(frame, (900, 900))
        cv2.imshow("frame", frame)
        cv2.waitKey(1)