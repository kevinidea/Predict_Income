import pandas as pd

class Data_Preparation(object):

    def __init__(self):
        self.fileName = "adult.csv"
        self.columns = [ "Age", "Workclass", "fnlwgt", "Education", "Education_Num", "Martial_Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital_Gain", "Capital_Loss",
         "Hours_Per_Week", "Country", "Income"]
        self.categoricalVariables = [ "Workclass", "Education", "Martial_Status",
            "Occupation", "Relationship", "Race", "Sex", "Country", "Income" ]
        self.numericalVariables = ["Age", "fnlwgt", "Education_Num","Capital_Gain", "Capital_Loss","Hours_Per_Week" ]

    def setFileName(self,newFileName):
        self.fileName = newFileName

    def getFileName(self):
        return self.fileName

    def getColumns(self):
        return self.columns

    def getCategoricalVariables(self):
        return self.categoricalVariables

    def getNumericalVariables(self):
        return self.numericalVariables

    def storeCleanData(self, save = True):
        #Read original data
        data = pd.read_csv(
            self.fileName,
            names = self.columns,
            sep = r"\s*,\s*",
            na_values = "?",
            engine = 'python'
        )

        print "Original Data Shape: " + str(data.shape)

        #Delete rows that have missing values
        cleanData = data.dropna(
            axis = 0,
            how='any',
            inplace = False
        )

        print "Clean Data Shape " + str(cleanData.shape)

        #Save clean data into another CSV to import into SQLite3 database
        if save == True:
            cleanData.to_csv("cleanData.csv", index=False)

        return cleanData


