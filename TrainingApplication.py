import sys
from tkinter import filedialog
from matplotlib import pyplot as plt
from sklearn import datasets, svm
from sklearn.neighbors import KNeighborsClassifier
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
    # print(digits)

    # Classifier implementing the k-nearest neighbors vote and support vector machine
    clf_neighbors = KNeighborsClassifier(n_neighbors=20)
    clf_svm = svm.SVC(gamma=0.001, probability=True)

    # Fit the knn and svc classifier from the training dataset
    clf_neighbors.fit(x, y)
    clf_svm.fit(x, y)

    # Save trained model
    print("Model successfully trained")
    print("Save your trained Model")
    # saveModel(clf_neighbors)
    saveModel(clf_svm)

    # Visualize a digit
    # plt.imshow(digits.images[0]);
    # plt.gray()
    # plt.show()

if __name__ == '__main__':
    main()
