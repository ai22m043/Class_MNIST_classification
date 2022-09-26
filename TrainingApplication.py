import sys
from tkinter import filedialog

import cv2
from sklearn import datasets, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load
import os


def saveModel(model):
    while True:
        directory = os.getcwd()
        filetypes = (
            ("*.joblib", "*.joblib"),
            ("All files", "*.*")
        )

        file = filedialog.asksaveasfilename(
            title="Save the trained model:",
            initialdir=directory,
            filetypes=filetypes,
            defaultextension=filetypes)

        try:
            dump(model, file)
            print("  --> Model successfully saved in:")
            print("     ", file)
            return
        except:
            print("  --> Model not saved")
            sys.exit("Application closed")


def main():
    # Load the MNIST dataset
    digits = datasets.load_digits()

    # Create features and targets
    x = digits.data
    y = digits.target
    #print(digits)

    # Classifier implementing the k-nearest neighbors vote and support vector machine
    neighbors = KNeighborsClassifier(n_neighbors=20)
    clf = svm.SVC(gamma=0.001, probability=True)

    # Fit the knn and svc classifier from the training dataset
    neighbors.fit(x, y)
    clf.fit(x, y)

    print("Model successfully trained")
    print("Save your trained Model")
    #saveModel(neighbors)
    saveModel(clf)


if __name__ == '__main__':
    main()
