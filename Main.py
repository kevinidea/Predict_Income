from Data_Preparation import Data_Preparation
from Data_Exploration import Data_Exploration
from Data_Modeling import Data_Modeling
from Prediction import Prediction

while True:
    whatToDo = raw_input("\n Do you want to \"prepare data\", \"explore data\", \"model data\", \"make prediction\" or \"quit\"?\n")

    #####################Data Preparation
    if "prepare data" in whatToDo:
        prep = Data_Preparation()
        prep.setFileName('adult.csv')
        data = prep.storeCleanData(save=False)

    #####################Data Exploration
    elif "explore data" in whatToDo:
        prep = Data_Preparation()
        explore  = Data_Exploration()
        #get all the variable names
        numericalVariables = prep.getNumericalVariables()
        print "Numerical variables:" + str(numericalVariables)
        categoricalVariables = prep.getCategoricalVariables()
        print "Categorical variables:" + str(categoricalVariables)

        #Explore 1 variable at a time
        #Numerical variables
        explore.createHistograms(numericalVariables, 2,3,1)
        #Categorical variables
        explore.createCountPlots(categoricalVariables,3,3,2)

        #Explore relationships between income and many categorical variables
        explore.incomeByVariables(categoricalVariables[0:4], 2,2,3)
        explore.incomeByVariables(categoricalVariables[4:8], 2,2,4)

        #Explore correlations among all variables
        explore.createHeatmapOfCorrelation(5)

    #####################Data Modeling
    elif "model data" in whatToDo:
        whichModel = raw_input("Do you want logistic regression or random forest model?\n")
        modeling = Data_Modeling()
        data = modeling.getData()
        X_train, X_test, y_train, y_test = modeling.transformData(data)

        if "logistic" in whichModel:
            predicted_y_train, predicted_y_test = modeling.buildLogisticRegression(X_train, X_test, y_train, cv =10)
            modeling.evaluatePerformance(y_train, predicted_y_train, "TRAINING Performance for Logistic Regression Model")
            modeling.evaluatePerformance(y_test, predicted_y_test, "TESTING Performance for Logistic Regression Model")

        elif "random" in whichModel:
            predicted_y_train, predicted_y_test = modeling.buildRandomForest(X_train, X_test, y_train, cv=5, n_iter=2000)
            modeling.evaluatePerformance(y_train, predicted_y_train, "TRAINING Performance for Random Forest Model")
            modeling.evaluatePerformance(y_test, predicted_y_test, "TESTING Performance for Random Forest Model")

        else:
            print "We don't have this model!"

    #####################Prediction
    elif "prediction" in whatToDo:
        modeling = Data_Modeling()
        prediction = Prediction()

        #Get the data
        train_data = modeling.getData()
        test_data = prediction.prepareTestData('adult.test.txt')

        #Transform the data
        X_train, X_test, y_train, y_test = prediction.transformTestData(train_data, test_data)

        #Make predictions
        whichModel = raw_input("Do you want to make predictions using logistic regression or random forest model?\n")
        if "logistic" in whichModel:
            predictions = prediction.predictWithLR(test_data)

        elif "random" in whichModel:
            predictions = prediction.predictWithRF(test_data)

        else:
            print "We don't have this model"

        #Evaluate prediction performance
        modeling.evaluatePerformance(y_test, predictions, "Final Prediction Performance")

    else:
        break