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

# Select the last 1000 shapes to test the accuracy of the model
triangles = os.listdir("./Triangle")[9000:9999]
squares = os.listdir("./Square")[9000:9999]
circles = os.listdir("./Circle")[9000:9999]

triangles = [x for x in triangles if x != ".DS_Store"]
squares = [x for x in squares if x != ".DS_Store"]
circles = [x for x in circles if x != ".DS_Store"]

# Calculate HOG of each image, save them with the appropriate label
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
# Load the model
clf = load("svm_model.dat")
# Model will predict & label each image based on its HOG (0 for triangles, 1 for squares, -1 for cicles)
# If labels predicted match the label of the image, the model has made an accurate guess
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