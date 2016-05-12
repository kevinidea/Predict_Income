import pandas as pd
from Data_Preparation import Data_Preparation
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

class Prediction(object):

    def __init__(self):
        self.originalDataFileName = 'cleanData.csv'

    def setOriginalDataFileName(self, newName):
        self.originalDataFileName = newName

    def getOriginalDataFileName(self):
        return self.originalDataFileName

    def prepareTestData(self, fileName):
        #Prepare test data
        prep = Data_Preparation()
        columnNames = prep.getColumns()
        test_data = pd.read_csv(
            fileName,
            names = columnNames,
            sep = r"\s*,\s*",
            na_values = "?",
            engine = "python",
        )

        #Fill in missing values
        #print test_data.count()
        test_data['Workclass'].fillna(value = 'Private', inplace=True)
        test_data['Occupation'].fillna(value= 'Prof-specialty', inplace=True)
        test_data['Country'].fillna(value = 'United-States', inplace = True)
        #print test_data.count()
        return test_data


    def __selectRelevantFeatures(self, df):
        #Select relevant features
        relevantFeatures = ['Martial_Status', 'Occupation','Relationship', 'Race', 'Sex',
            'Age', 'Education_Num','Capital_Gain', 'Capital_Loss', 'Hours_Per_Week']

        #Construct X and y and turn them in numpy arrays
        X = df[relevantFeatures].values
        y = df['Income'].values

        return X, y


    def transformTestData(self, train_data, test_data):
        #Select the right features for both training and testing data
        X_train, y_train = self.__selectRelevantFeatures(train_data)
        X_test, y_test = self.__selectRelevantFeatures(test_data)

        #Transform categorical variables into integer labels
        martial_le = LabelEncoder()
        occupation_le = LabelEncoder()
        relationship_le = LabelEncoder()
        race_le = LabelEncoder()
        sex_le = LabelEncoder()
        transformers = [martial_le, occupation_le, relationship_le, race_le, sex_le]

        for i in range(len(transformers)):
            X_train[:,i] = transformers[i].fit_transform(X_train[:,i])
            X_test[:,i] = transformers[i].transform(X_test[:,i])

        #Dummy code categorical variables
        dummy_code = OneHotEncoder(categorical_features = range(5))
        X_train = dummy_code.fit_transform(X_train).toarray()
        X_test = dummy_code.transform(X_test).toarray()

        #Normalize all features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        #Encode y
        class_le = LabelEncoder()
        y_train = class_le.fit_transform(y_train)
        y_test = class_le.transform(y_test)
        #print class_le.transform(["<=50K", ">50K"])

        return X_train, X_test, y_train, y_test


    def predictWithLR(self, test_data, saveModel = False):
        #Get data
        train_data = pd.read_csv(self.originalDataFileName)
        test_data = test_data
        X_train, X_test, y_train, y_test = self.transformTestData(train_data, test_data)

        #Retrain the model using full original dataset
        finalLogisticRegression = LogisticRegression (C =0.01)
        finalLogisticRegression.fit(X_train, y_train)

        if saveModel == True:
            joblib.dump(finalLogisticRegression, "Final_Logistic_Regression.pkl", compress=1)

        #Make Predictions
        predictions = finalLogisticRegression.predict(X_test)

        return predictions


    def predictWithRF(self, test_data, saveModel = False):
        #Get data
        train_data = pd.read_csv(self.originalDataFileName)
        test_data = test_data
        X_train, X_test, y_train, y_test = self.transformTestData(train_data, test_data)

        #Retrain the model using full original dataset
        finalRandomForest = RandomForestClassifier (
            min_samples_leaf= 3,
            n_estimators = 21,
            max_features= 12,
            min_samples_split= 4,
            max_depth= 67,
            random_state = 101
        )
        finalRandomForest.fit(X_train, y_train)
        if saveModel == True:
            joblib.dump(finalRandomForest, "Final_Random_Forest.pkl", compress=1)

        #Make predictions
        predictions = finalRandomForest.predict(X_test)

        return predictions