import cv2
import numpy as np
import glob

paths = glob.glob("-----------/OIDv4_ToolKit/OID/Dataset/train/*/*.jpg")
# change this path
a = 0
fotolar = []
for pat in paths:
    foto = cv2.imread(pat)
    orijinalShape = foto.shape
    y, x, c = orijinalShape
    parca = pat.split("\\")
    label = parca[0]+"/"+parca[1]+"/Label/"+parca[2].split(".")[0] + ".txt"
    xml = parca[0]+"/"+parca[1]+"/"+parca[2].split(".")[0] + ".xml"
    try:
        with open(label, "r") as reader:
            tekLabel = reader.read().split("\n")
            file = open(xml, "w") 
            file.write("<annotation>\n")
            file.write("	<folder>notimportant</folder>\n")
            file.write("	<filename>"+xml+"</filename>\n")
            file.write("	<path>"+pat+"</path>\n")
            file.write("	<source>\n")
            file.write("		<database>Unknown</database>\n")
            file.write("	</source>\n")
            file.write("	<size>\n")
            file.write("		<width>300</width>\n")
            file.write("		<height>300</height>\n")
            file.write("		<depth>3</depth>\n")
            file.write("	</size>\n")
            file.write("	<segmented>0</segmented>\n")
            for i in tekLabel:
                if i is not "":
                    final = i.split(" ")
                    kelime = len(final) - 4
                    xMin = int(float(final[kelime])) 
                    yMin = int(float(final[kelime + 1])) 
                    xMax = int(float(final[kelime + 2])) 
                    yMax = int(float(final[kelime + 3]))
                    cv2.rectangle(foto, (xMin, yMin), (xMax, yMax), (0, 0,  255), 2)
                    isim = ""
                    for kel in range(kelime):
                        isim = isim + final[kel]
                    print(isim)
                    file.write("	<object>\n")
                    file.write("		<name>"+isim+"</name>\n")
                    file.write("		<pose>Unspecified</pose>\n")
                    file.write("		<truncated>0</truncated>\n")
                    file.write("		<difficult>0</difficult>\n")
                    file.write("		<bndbox>\n")
                    file.write("			<xmin>{}</xmin>\n".format(xMin))
                    file.write("			<ymin>{}</ymin>\n".format(yMin))
                    file.write("			<xmax>{}</xmax>\n".format(xMax))
                    file.write("			<ymax>{}</ymax>\n".format(yMax))
                    file.write("		</bndbox>\n")
                    file.write("	</object>\n")

            file.write("</annotation>\n")
            file.close()
    except:
        pass
    cv2.imshow("frame", foto)
    cv2.waitKey(1)
    a = a + 1
