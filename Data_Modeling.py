from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.externals import joblib

class Data_Modeling(object):

    def __init__(self):
        self.data = pd.read_csv('cleanData.csv')

    def setData(self, newFileName):
        self.data = pd.read_csv(newFileName)

    def getData(self):
        return self.data

    def transformData(self, data):
        #Select the relevant features
        #print prep.columns
        relevantFeatures = ['Martial_Status', 'Occupation','Relationship', 'Race', 'Sex',
            'Age', 'Education_Num','Capital_Gain', 'Capital_Loss', 'Hours_Per_Week']

        #Construct big matrix X and array y
        X = data[relevantFeatures].values
        y = data['Income'].values

        #split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

        #Transform categorical variables into integer labels
        martial_le = LabelEncoder()
        occupation_le = LabelEncoder()
        relationship_le = LabelEncoder()
        race_le = LabelEncoder()
        sex_le = LabelEncoder()
        transformers = [martial_le, occupation_le, relationship_le, race_le, sex_le]

        for i in range(len(transformers)):
            X_train[:, i] = transformers[i].fit_transform(X_train[:,i])
            X_test[:,i] = transformers[i].transform(X_test[:,i])
        #print X_train.shape
        #print X_train[0,:]

        #Dummy code categorical variables
        dummy_code = OneHotEncoder(categorical_features = range(5))
        X_train = dummy_code.fit_transform(X_train).toarray()
        X_test = dummy_code.transform(X_test).toarray()
        #print X_train.shape
        #print X_train[0,:]

        #Normalize all features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        #print X_train[0,:]

        #Encode y
        class_le = LabelEncoder()
        y_train = class_le.fit_transform(y_train)
        y_test = class_le.transform(y_test)
        #print class_le.transform(["<=50K", ">50K"])

        return X_train, X_test, y_train, y_test


    #Logistic Regression
    def buildLogisticRegression(self, X_train, X_test, y_train, cv = 5, save = False):
        lr = LogisticRegression()
        #Tune the model
        param_grid = {
            'C':[10**-5, 10**-4, 0.001,0.01,0.1,1,10,100]
        }

        lr_optimized = GridSearchCV(
            estimator = lr,
            param_grid = param_grid,
            scoring= "f1",
            cv=cv
        )

        lr_optimized.fit(X_train, y_train)
        if save == True:
            joblib.dump(value =lr_optimized, filename='lr_optimized.pkl', compress=1)

        print "Best parameter: %s"  %lr_optimized.best_params_
        print "Best average cross validated F1 score: %0.4f" %lr_optimized.best_score_
        print "--------------------------------------------"
        print lr_optimized.best_estimator_.coef_

        #predictions
        predicted_y_train = lr_optimized.predict(X_train)
        predicted_y_test = lr_optimized.predict(X_test)

        return predicted_y_train, predicted_y_test


    #Random Forest
    def buildRandomForest(self, X_train, X_test, y_train, cv = 3, n_iter = 5, save = False):
        rf = RandomForestClassifier(random_state = 9)
        #Tune the model
        param_distributions = {
            'n_estimators': range(1,50,1),
            'max_depth': range(1,70,1),
            'max_features': range(6,15,1),
            'min_samples_split':[2,3,4],
            'min_samples_leaf':[1,2,3,4],
            'n_jobs':[-1]
        }

        rf_optimized = RandomizedSearchCV(
            estimator = rf,
            param_distributions = param_distributions,
            n_iter= n_iter,
            scoring = 'f1',
            cv = cv,
            random_state = 1
        )

        rf_optimized.fit(X_train, y_train)
        if save == True:
            joblib.dump(value = rf_optimized, filename = "rf_optimized.pkl", compress=1)

        print "Best parameter: %s"  %rf_optimized.best_params_
        print "Best average cross validated F1 score: %0.4f" %rf_optimized.best_score_
        print "--------------------------------------------"

        #predictions
        predicted_y_train = rf_optimized.predict(X_train)
        predicted_y_test = rf_optimized.predict(X_test)

        return predicted_y_train, predicted_y_test


    #Evaluate model performance
    def evaluatePerformance(self, actual, prediction, title):
        print title
        print "Accuracy is %.4f" % accuracy_score(actual, prediction)
        print "F1 Score is %0.4f" %f1_score(actual, prediction)
        print classification_report(actual, prediction, target_names = ["<=50K", ">50K"])
        matrix = confusion_matrix(actual, prediction)
        print "Confusion Matrix:"
        print matrix
        print "----------------------------------------------------"
        plt.figure(1)
        sns.heatmap(
            data= matrix,
            annot=True,
            fmt="d",
            xticklabels = ["<=50K", ">50K"],
            yticklabels = ["<=50K", ">50K"],
            square= True
        )
        plt.title(title)
        plt.xlabel("Prediction")
        plt.ylabel("Actual")
        plt.show()