import xml.etree.ElementTree as ET
import os


directory = "E:/Datasets/VOC2012/Annotations"
for filename in os.listdir(directory):
    #print(filename)
    tree = ET.parse(directory+"/"+filename)
    root = tree.getroot()
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    for obj in root.iter("object"):
        obj_class = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)


    print(width)
    print(height)
    print(obj_class)
    print(xmin)
    print(ymin)
    print(xmax)
    print(ymax)
    break;

#for every xml file:
    #width
    #height

    #for every object:
        #name/class
        #xmin
        #ymin
        #xmax
        #ymax



# for filename in os.listdir(directory)
# for filename in os.listdir(path):
#     if not filename.endswith('.xml'): continue
#     fullname = os.path.join(path, filename)
#     tree = ET.parse(fullname)
