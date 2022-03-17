from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas 
from sklearn.metrics import confusion_matrix
import hough_transform

def splitCard(file):
    features = pandas.read_csv(file)
    y = features['grade']
    x = features.drop(['fileName', 'grade'], 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    return x_train, x_test, y_train, y_test

def datasetSVM(file):
    x_train, x_test, y_train, y_test = splitCard(file)
    classifier = SVC()
    classifier.fit(x_train, y_train)
    print(x_test)
    prediction = classifier.predict(x_test)
    print(y_test.shape)
    print(prediction.shape)
    count = 0
    for i in prediction:
        for j in y_test:
            if i == j:
                count += 1
    accuracy = float(count/len(y_test))
    print("\nAccuracy: ", accuracy)

def predictCard(card, data):
    features = pandas.read_csv(data)
    y = features['grade']
    x = features.drop(['fileName', 'grade'], 1)
    classifier = SVC()
    classifier.fit(x, y)
    verticalRatio, horizontalRatio, c1, c2, c3, c4 = hough_transform.test(card)
    input = [[verticalRatio, horizontalRatio, c1, c2, c3, c4]]
    #input.reshape(1,-1)
    prediction = classifier.predict(input)
    split = card.split("_", 1)[0]
    grade = int(split[3:])
    print("Prediction: ", prediction)
    print("Original:", grade)

#datasetSVM("test.csv")
predictCard("psa9_10.jpg", "test.csv")
