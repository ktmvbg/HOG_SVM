import numpy as np
from skimage import io
from skimage.feature import hog
from sklearn.svm import LinearSVC, SVC
from skimage.transform import resize
from joblib import dump, load
from skimage.color import rgb2gray
import sys
np.set_printoptions(threshold=sys.maxsize)
import os

def cal_hog(img):
    return hog(img, orientations=10, pixels_per_cell=(20, 20),cells_per_block=(1, 1))

samples = []
labels = []

tests = []
tests_labels = []

triangles = os.listdir("./Triangle")[:8500]
squares = os.listdir("./Square")[:8500]
circles = os.listdir("./Circle")[:8500]


print("cal hog triangles")
for filename in triangles:
    img = io.imread("./Triangle/"+filename)
    img = rgb2gray(img)
    # img = resize(img, (64, 64))
    hist = cal_hog(img)
    samples.append(hist)
    labels.append(0)
print("cal hog squares")
for filename in squares:
    img = io.imread("./Square/"+filename)
    img = rgb2gray(img)
    # img = resize(img, (64, 64))
    hist = cal_hog(img)
    samples.append(hist)
    labels.append(1)

print("cal hog circles")
for filename in circles:
    img = io.imread("./Circle/"+filename)
    img = rgb2gray(img)
    channel_axis: int = -1
    # img = resize(img, (64, 64))
    hist = cal_hog(img)
    samples.append(hist)
    labels.append(-1)
print("start training")
clf = SVC(decision_function_shape='ovr')
# clf = LinearSVC()
clf.fit(samples, labels)
dump(clf, "svm_model.dat")