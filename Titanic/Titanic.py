import pandas as pd #data manipulation
import numpy as np #linear algebra
import seaborn as sns #data visualization
import missingno as mn #visualizing missing data
import matplotlib.pyplot as plt #data visualization
from CategoricalData import *
from NumericalData import *
from scipy import stats #statistics

# plt.style.use('bmh')
sns.set_style({'axes.grid':False})

from IPython.display import Markdown, display
def bold(string):
    display(Markdown(string))

#### Reading Data
TrainDataLoc = './Dataset/train.csv'
TestDataLoc = './Dataset/test.csv'
train = pd.read_csv(TrainDataLoc)
test = pd.read_csv(TestDataLoc)

display(train.head(2))

#The first step will be analysing the data and modifying it to suit our model

merged = pd.concat([train, test], sort = False)

display(merged.shape) #1309 rows of data, 12 variables

display(merged.columns)
display(merged.dtypes)
'''Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')'''


###First step will be analyzing each variable on its own, we know that there are categorical data and numerical data

###Note that cabin and Ticket have too many categories, so we will have to group them some how
'''
absolute_and_relative_freq(merged.Survived)
absolute_and_relative_freq(merged.Sex)
absolute_and_relative_freq(merged.Pclass)
absolute_and_relative_freq(merged.Sex)
absolute_and_relative_freq(merged.SibSp)
absolute_and_relative_freq(merged.Embarked)
absolute_and_relative_freq(merged.Parch)
'''
'''
histogram(merged.Fare)
density_plot(merged.Fare)
bold('**Summary Stats of Fare:**')
summary_stats(merged.Fare)

histogram(merged.Age)
density_plot(merged.Age)
bold('**Summary Stats of Age:**')
summary_stats(merged.Age)
'''

#Now we should be concerned about missing values, and feature engineering in general
#for instance Cabin has a lot of empty data

bold('**Missing Values in Cabin:**')
display(merged.Cabin.isnull().sum()) #1000 data
merged.Cabin.fillna(value = 'X', inplace = True)
merged.Cabin = merged.Cabin.apply( lambda x : x[0])
display(merged.Cabin.value_counts())

#In name we can obtain the honorifics
merged['Title'] = merged.Name.str.extract('([A-Za-z]+)\.')
display(merged.Title.value_counts())
