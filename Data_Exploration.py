import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import numpy as np

class Data_Exploration(object):

    def __init__(self):
        #Connect to SQLite3 database and clean dataset
        self.database = sqlite3.connect('cleanData.sqlite')
        self.FileName = 'cleanData.csv'
        self.data = pd.read_csv(self.FileName)

    def getFileName(self):
        return self.FileName

    def setFilename(self, newFileName):
        self.FileName = newFileName

    #Exploring 1 variable at a time
    '''
    createHistogram function
    Inputs:
    1. numericalVariables: List of numerical variable names from the database (list of strings)
    2. nRows: Number of rows of subplots (int)
    3. mCols: Number of columns of subplots (int)
    4. figureNumer: Figure number (int)
    Output:
    Generate one figure containing many histograms of all numerical variables (histograms)
    '''
    def createHistograms(self, numericalVariables, nRows, mCols, figureNumber):
        plt.figure(figureNumber)
        #Visualize many numerical variables distribution all at once
        for i in range(len(numericalVariables)):
            #Write SQL query to get a list of all values from a variable
            sql = "SELECT " + numericalVariables[i] + " FROM cleanData"
            #Save the result into dataframe
            df = pd.read_sql(sql =sql, con= self.database)

            #Visualize the results with histograms
            plt.subplot(str(nRows) + str(mCols) + str(i))
            sns.distplot(a = df, hist=True, kde = False, bins = 10)
            plt.title(numericalVariables[i] + " Histogram")
            plt.xlabel(numericalVariables[i])
            plt.ylabel("Frequency")
        plt.show()


    '''
    createCountPlots function
    Inputs:
    1. numericalVariables: List of categorical variable names from the database (list of strings)
    2. nRows: Number of rows of subplots (int)
    3. mCols: Number of columns of subplots (int)
    4. figureNumer: Figure number (int)
    output:
    Generate one figure containing many count plots of all categorical variables (count plots)
    '''
    def createCountPlots(self, categoricalVariables, nRows, mCols, figureNumber):
        plt.figure(figureNumber)
        #Visualize many categorical variable distributions all at once
        for i in range(len(categoricalVariables)):
            #Write SQL query to calculate the frequency of a variable
            sql = "SELECT " + categoricalVariables[i] + "," \
            + "COUNT(" + categoricalVariables[i] + ") AS Frequency FROM cleanData " \
            + "GROUP BY " + categoricalVariables[i]
            #Read the result into dataframe
            df = pd.read_sql(sql = sql, con = self.database)
            #Sort the result with highest frequency from top to bottom
            df.sort_values(by = "Frequency", ascending = False, inplace = True)

            #Visualize the results with many subplots
            plt.subplot(str(nRows) + str(mCols) + str(i))
            sns.barplot(x = "Frequency", y = categoricalVariables[i], data = df, palette = "Blues_d")
            plt.title(categoricalVariables[i] + " Distribution")
            plt.xlabel("Frequency")
            plt.ylabel(categoricalVariables[i])
        plt.show()


    #Exploring relationship between income and many other variables
    '''
    incomeByVariables function
    Inputs:
    1. variables: List of categorical variable names from the database (list of strings)
    2. nRows: Number of rows of subplots (int)
    3. mCols: Number of columns of subplots (int)
    4. figureNumer: Figure number (int)
    output:
    Generate one figure containing many income distributions by each categorical variable
    '''
    def incomeByVariables(self, variables, nRows, mCols, figureNumber):
        plt.figure(figureNumber)
        #Create one subplot per each variable
        for i in range(len(variables)):
            #Write SQL to calculate the number of population with <=50K and >50K income per group
            sql = "SELECT " + variables[i] + ", " \
            + "COUNT(CASE WHEN Income =\"<=50K\" THEN Income ELSE NULL END) AS LessThanOrEqualTo50K, " \
            + "COUNT(CASE WHEN Income =\">50K\" THEN Income ELSE NULL END) AS MoreThan50K " \
            + "FROM cleanData " \
            + "GROUP BY " + variables[i]
            #Read the SQL into Pandas dataframe
            df = pd.read_sql(sql=sql, con=self.database)
            #Calculate total population per group
            df['Total_Population'] = df['LessThanOrEqualTo50K'] + df['MoreThan50K']
            #Calcualte the proprotion of MoreThan50K per group
            df['MoreThan50KProportion'] = df.MoreThan50K / df.Total_Population
            #Sort the data with highest MoreThan50KProportion from top to bottom
            df.sort_values(by = 'MoreThan50KProportion', inplace= True, ascending = False)

            #Visualize the results
            plt.subplot(str(nRows) + str(mCols) + str(i))
            sns.barplot(y = variables[i], x='Total_Population', data = df, color = 'Red')
            sns.barplot(y = variables[i], x ='MoreThan50K', data = df, color = 'Blue')
            topBar =plt.Rectangle((0,0), 1,1, fc='Red', edgecolor = None)
            bottomBar = plt.Rectangle((0,0), 1,1, fc='Blue', edgecolor = None)
            plt.legend([topBar, bottomBar], ['<=50K', '>50K'], loc=1, ncol = 2, prop = {'size':16})
            plt.title('Income Distribution by ' + variables[i])
            plt.xlabel('Frequency')
            plt.ylabel(variables[i])
        plt.show()

    #Explore correlations among all variables
    '''
    createHeatmapOfCorrelation function
    Inputs: figureNumber (int)
    output: Generate a heatmap to visualize correlations among all variables
    '''
    def createHeatmapOfCorrelation(self, figureNumber):
        #Transform only categorical variables into integers
        encodedData = self.data.copy()
        encoders = {}
        for column in encodedData.columns:
            if encodedData.dtypes[column] == np.object:
                encoders[column] = LabelEncoder()
                encodedData[column] = encoders[column].fit_transform(encodedData[column])
        #Find the correlations among all variables
        correlation = encodedData.corr()

        #Plot the correlation with heatmap
        plt.figure(figureNumber)
        #Shorten some of the column names for readiability on the graph
        names = [ "Age", "Workclass", "fnlwgt", "Edu", "Edu_Num", "Martial",
        "Job", "Relationship", "Race", "Sex", "Gain", "Loss",
         "Hours", "Country", "Income"]
        sns.heatmap(correlation, xticklabels = names, yticklabels = names)
        plt.show()

