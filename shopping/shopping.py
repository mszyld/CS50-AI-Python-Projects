import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Loads shopping data from a CSV file `filename` and converts into a list of
    evidence lists and a list of labels. Returns a tuple (evidence, labels).

    evidence is a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels is the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # Read data in from file
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)

        evidence = []
        labels = []
        for row in reader:
            
            # append row to evidence and modify data types as required
            evidence.append([cell for cell in row[:17]])

            # - Administrative, an integer
            evidence[-1][0] = int(evidence[-1][0])
            # - Administrative_Duration, a floating point number
            evidence[-1][1] = float(evidence[-1][1])
            # - Informational, an integer
            evidence[-1][2] = int(evidence[-1][2])
            # - Informational_Duration, a floating point number
            evidence[-1][3] = float(evidence[-1][3])
            # - ProductRelated, an integer
            evidence[-1][4] = int(evidence[-1][4])
            # - ProductRelated_Duration, a floating point number
            # - BounceRates, a floating point number
            # - ExitRates, a floating point number
            # - PageValues, a floating point number
            # - SpecialDay, a floating point number
            for i in range(5,10):
                evidence[-1][i] = float(evidence[-1][i])
            # - Month, an index from 0 (January) to 11 (December)
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            evidence[-1][10] = months.index(evidence[-1][10][:3])
            # - OperatingSystems, an integer
            # - Browser, an integer
            # - Region, an integer
            # - TrafficType, an integer
            for i in range(11,15):
                evidence[-1][i] = int(evidence[-1][i])
            # - VisitorType, an integer 0 (not returning) or 1 (returning)
            evidence[-1][15] = 1 if evidence[-1][15] == 'Returning_Visitor' else 0
            # - Weekend, an integer 0 (if false) or 1 (if true)
            evidence[-1][16] = 1 if evidence[-1][15] == 'TRUE' else 0

            labels.append(1 if row[-1] == "TRUE" else 0)
    return (evidence,labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, returns a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Each label is assumed either a 1 (positive) or 0 (negative).

    `sensitivity` is a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately accurately.

    `specificity` is a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    total_positives = 0
    positives_accurately_identified = 0
    total_negatives = 0
    negatives_accurately_identified = 0
    for label,prediction in zip(labels,predictions):
        if label == 1:
            total_positives += 1
            if prediction == 1:
                positives_accurately_identified += 1
        else:
            total_negatives += 1
            if prediction == 0:
                negatives_accurately_identified += 1

    sensitivity = positives_accurately_identified / total_positives
    specificity = negatives_accurately_identified / total_negatives
    return (sensitivity,specificity)


if __name__ == "__main__":
    main()
