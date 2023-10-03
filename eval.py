import os
import cv2
import numpy as np
from lenet import LeNet
import matplotlib.pyplot as plt  # get Matplotlib 

#prob = [0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01, 0.0105, 0.011, 0.0115, 0.012, 0.0125, 0.013, 0.0135, 0.014]
accuracies = []  # prob->acc

prob = [i * 0.05 for i in range(20)]

data_dir = "mnist/test"
for p in prob:
    net = LeNet(p)
    net.load("lenet.npy")
    files = os.listdir(data_dir)
    images = []
    labels = []
    for f in files:
        img = cv2.imread(os.path.join(data_dir, f), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32, 32))
        img = img.astype(np.float32).reshape(32, 32, 1) / 255.0
        images.append(img)
        labels.append(int(f[0]))

    x = np.array(images)
    y = np.array(labels)

    predict = net.predict(x)
    tp = np.sum(predict == y)
    accuracy = float(tp) / len(files)
    accuracies.append(accuracy)
    print("prob = %f" % p,"accuracy=%f" % accuracy)

plt.plot(prob, accuracies, marker='o', linestyle='-')
plt.xlabel('prob')
plt.ylabel('accuracy')
plt.title('Conv-fix0.001wrong: Accuracy by Prob')
plt.grid(True)
plt.show()

