import shutil
import glob

txtfiles = []
for file in glob.glob("all-mosaic/*.jpg"):
    txtfiles.append(file)

sayac = 0
for i in txtfiles:
    try:
        sayac = sayac + 1
        if sayac < 48:
            xmlfile = i.split(".")
            xmlfile = xmlfile[0] + ".xml"
            shutil.copy(xmlfile, "train")
            shutil.copy(i, "train")
        if sayac >= 48:
            xmlfile = i.split(".")
            xmlfile = xmlfile[0] + ".xml"
            shutil.copy(xmlfile, "test")
            shutil.copy(i, "test")
        if sayac == 60:
            sayac = 0
    except:
        shutil.copy(i, "nontrain")

