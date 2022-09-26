import sys
import os
import cv2
from tkinter import filedialog
import numpy as np
from joblib import dump, load


def loadModel():
    while True:
        directory = os.getcwd()
        filetypes = (
            ("*.joblib", "*.joblib"),
            ("All files", "*.*")
        )

        file = filedialog.askopenfilename(
            title="Select a trained model:",
            initialdir=directory,
            filetypes=filetypes)

        if len(file) != 0:
            try:
                print("  -->", file)
                model = load(file)
                print("  --> Model successfully loaded")
                return model
            except:
                print(" Something went wrong..")
                continue

        sys.exit("Application closed")


def loadImage():
    while True:
        directory = os.getcwd()
        filetypes = (
            ("*.jpg", "*.jpg"),
            ("*.png", "*.png"),
            ("All files", "*.*")
        )

        file = filedialog.askopenfilename(
            title="Select a image",
            initialdir=directory,
            filetypes=filetypes)

        if len(file) != 0:
            try:
                print("  -->", file)
                image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                print("  --> Image successfully loaded")
                return image
            except:
                print(" Something went wrong..")
                continue

        sys.exit("Application closed")


def main():
    print("Select a trained model")
    model = loadModel()
    print("Select a image to predict a number")
    image = loadImage()
    print("\nLoaded image with shape", image.shape, ":\n", image)

    # Format image to fit (8,8) arrays = 64 dimensions
    image_resized = cv2.resize(image, (8, 8))
    print("\nImage resized to shape", image_resized.shape, ":\n", image_resized)

    # Invert Image to white number with black background
    image_resized = cv2.bitwise_not(image_resized)
    print("\nImage inverted to white number with black background:\n", image_resized)

    # Normalize data to fit model from 0-255 to 0-16
    oldRange = 255
    newRange = 16
    a = [((((jj - 0) * newRange) / oldRange) + 0).astype(int) for j in image_resized for jj in j]
    a = np.array(a).reshape(1, -1)
    print("\nImage normalized to fit model:\n", a.reshape(8, 8))

    # Predict Number
    # prediction = model.predict(a)
    prediction_prob = model.predict_proba(a)

    result = {}
    print("\nProbabilities of classification:")
    for x in model.classes_:
        tmp = prediction_prob[0][x] * 100
        print("Number", x, ":", tmp, "%")
        result[x] = tmp

    prediction = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

    print("\nThe image is with [" + str(list(prediction.items())[0][1]), "%] probability number", [list(prediction.items())[0][0]])


if __name__ == '__main__':
    main()
