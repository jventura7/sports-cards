from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import pandas 
from sklearn.metrics import confusion_matrix
import hough_transform
import cv2

def splitCard(file):
    features = pandas.read_csv(file)
    y = features['grade']
    x = features.drop(['fileName', 'grade'], 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    return x_train, x_test, y_train, y_test

def datasetSVM(file):
    x_train, x_test, y_train, y_test = splitCard(file)
    classifier = SVR()
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    y_test = y_test.tolist()
    print(y_test)
    print(prediction)
    count = 0
    for i in range(len(prediction)):
        if prediction[i] < y_test[i] + 1 and prediction[i] > y_test[i] - 1:
            count += 1
    accuracy = float(count)/len(y_test)
    print(count)
    print("\nAccuracy: ", accuracy)

def predictCard(card, data):
    features = pandas.read_csv(data)
    y = features['grade']
    x = features.drop(['fileName', 'grade'], 1)
    classifier = SVR()
    classifier.fit(x, y)
    img = cv2.imread(card, 0)
    verticalRatio, horizontalRatio, c1, c2, c3, c4, rotate = hough_transform.test(img)
    input = [[verticalRatio, horizontalRatio, c1, c2, c3, c4, rotate]]
    #input.reshape(1,-1)
    prediction = classifier.predict(input)
    split = card.split("_", 1)[0]
    grade = int(split[3:])
    print("Prediction: ", prediction)
    print("Original:", grade)

datasetSVM("test.csv")
#predictCard("psa8_10.jpg", "test.csv")
