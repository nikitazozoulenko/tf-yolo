from skimage.io import imread_collection
from PIL import Image

collection_dir = 'E:/Datasets/VOC2012/JPEGImages/*.jpg'

collection = imread_collection(collection_dir)
print(collection[0].shape)
print(len(collection))
image = Image.fromarray(collection[0])
image.show()
