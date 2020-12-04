import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

C_ROUND = 5


def clean_train_data(df):
    """
    Clean/Filter dataset
    """
    survived = df['Survived']

    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked', 'Survived'],
            axis='columns',
            inplace=True)
    df.loc[:, 'Sex'].replace('male', 0, inplace=True)
    df.loc[:, 'Sex'].replace('female', 1, inplace=True)

    age_mean = df['Age'].mean()
    df.loc[:, 'Age'].fillna(age_mean, inplace=True)

    df['Survived'] = survived

    return df


def clean_test_data(df):
    """
    Clean/Filter dataset
    """
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'],
            axis='columns',
            inplace=True)
    df.loc[:, 'Sex'].replace('male', 0, inplace=True)
    df.loc[:, 'Sex'].replace('female', 1, inplace=True)

    age_mean = df['Age'].mean()
    df.loc[:, 'Age'].fillna(age_mean, inplace=True)
    df.loc[:, 'Fare'].fillna(0, inplace=True)

    return df


def train_dataset():
    """
    Train the model and save the best model
    """
    df_train = pd.read_csv("train.csv", header=0)
    df_train = clean_train_data(df_train)

    # Split the training data into 2 datasets
    X = df_train.to_numpy()[:, :-1]
    y = df_train.to_numpy()[:, -1]

    # Scale attributes down
    # X[:, 1:] = StandardScaler().fit_transform(X[:, 1:])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Multiple runs for several estimators to evaluate the best
    results = []
    estimators = [100, 125, 150, 175, 200, 225, 250, 275, 300]
    for estimator in estimators:
        adaBoost = AdaBoostClassifier(n_estimators=estimator)
        adaBoost.fit(X_train, y_train)
        y_pred = adaBoost.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        results.append({"Classifier": "AdaBoost",
                        "Estimators": estimator,
                        "Accuracy": round(score, C_ROUND)})

        gradBoost = GradientBoostingClassifier(n_estimators=estimator)
        gradBoost.fit(X_train, y_train)
        y_pred = gradBoost.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        results.append({"Classifier": "GradientBoost",
                        "Estimators": estimator,
                        "Accuracy": round(score, C_ROUND)})

    results.sort(key=lambda x: x['Accuracy'], reverse=True)
    for e in results:
        print("Method: " + str(e["Classifier"]) + "\t\t"
              "Estimators: " + str(e["Estimators"]) + "\t\t"
              "Score: " + str(e["Accuracy"]))

    best_result = results[0]
    if best_result["Classifier"] == "AdaBoost":
        adaBoost = AdaBoostClassifier(n_estimators=best_result["Estimators"])
        adaBoost.fit(X_train, y_train)
        return adaBoost
    else:
        gradBoost = GradientBoostingClassifier(
            n_estimators=best_result["Estimators"]
        )
        gradBoost.fit(X_train, y_train)
        return gradBoost


if __name__ == "__main__":
    model = train_dataset()
    df = pd.read_csv("test.csv", header=0)

    passengerId = df["PassengerId"]
    df = clean_test_data(df)

    # Split the training data into 2 datasets
    X = passengerId.to_numpy()
    y = model.predict(df.to_numpy())

    firstCol = pd.DataFrame(X, columns=['PassengerId'])
    secondCol = pd.DataFrame(y, columns=['Survived'])

    out_df = pd.concat([firstCol, secondCol], axis=1)
    out_df["Survived"] = out_df["Survived"].astype('int64')
    out_df.to_csv("result.csv", index=False)
