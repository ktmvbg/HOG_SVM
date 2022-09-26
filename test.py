import numpy as np
from skimage import io
from skimage.feature import hog
from sklearn.svm import LinearSVC
from skimage.transform import resize
from joblib import dump, load
from skimage.color import rgb2gray
import os

def cal_hog(img):
    return hog(img, orientations=10, pixels_per_cell=(20, 20),cells_per_block=(1, 1))

tests = []
labels = []

triangles = os.listdir("./Triangle")[9000:9999]
squares = os.listdir("./Square")[9000:9999]
circles = os.listdir("./Circle")[9000:9999]

triangles = [x for x in triangles if x != ".DS_Store"]
squares = [x for x in squares if x != ".DS_Store"]
circles = [x for x in circles if x != ".DS_Store"]
print("Cal hog triangle")
for filename in triangles:
    img = io.imread("./Triangle/"+filename)
    img = rgb2gray(img)
    # img = resize(img, (200, 200))
    hist = cal_hog(img)
    tests.append(hist)
    labels.append(0)
print("Cal hog square")
for filename in squares:
    img = io.imread("./Square/"+filename)
    img = rgb2gray(img)
    # img = resize(img, (200, 200))
    hist = cal_hog(img)
    tests.append(hist)
    labels.append(1)
print("Cal hog circle")
for filename in circles:
    img = io.imread("./Circle/"+filename)
    img = rgb2gray(img)
    # img = resize(img, (200, 200))
    hist = cal_hog(img)
    tests.append(hist)
    labels.append(-1)
print("start testing")
clf = load("svm_model.dat")
res = clf.predict(tests)
count = 0
n = 0
size = len(circles)
for i in range(n, n + size):
    if(res[i] == labels[i]):
        count += 1
print("circle")
print(count / size)
n = n + size
count = 0
size = len(triangles)
for i in range(n, n+size):
    if(res[i] == labels[i]):
        count += 1
print("triangle")
print(count / size)
n = n + size
count = 0
size = len(squares)
for i in range(n, n+size):
    if(res[i] == labels[i]):
        count += 1
print("square")
print(count / size)