from LR import LogisticRegression_
import numpy as np
import pandas as pd


def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df

if __name__ == "__main__":
    # load the data from the file
    data = load_data("data.txt", None)

    # X = feature values, all the columns except the last column
    X = data.iloc[:, :-1]

    # y = target values, last column of the data frame
    y = data.iloc[:, -1]

    # filter out the applicants that got admitted
    admitted = data.loc[y == 1]

    # filter out the applicants that din't get admission
    not_admitted = data.loc[y == 0]

    y = y.values
    X = X.values
    model = LogisticRegression_(alpha=0.5, max_iter=400)
    model.fit(X, y)

    prediction = model.predict(X)

    # class_ = np.argmax(probability)

    print("Train Accuracy:", sum(prediction == y), "%")