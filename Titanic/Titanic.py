import pandas as pd #data manipulation
import numpy as np #linear algebra
import seaborn as sns #data visualization
import missingno as mn #visualizing missing data
import matplotlib.pyplot as plt #data visualization
from CategoricalData import *
from NumericalData import *
from BoxPlots import *
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
summary_stats(merged.Fare)

histogram(merged.Age)
density_plot(merged.Age)
summary_stats(merged.Age)
'''

#Now we should be concerned about missing values, and feature engineering in general
#for instance Cabin has a lot of empty data

display(merged.Cabin.isnull().sum()) #1000 data
merged.Cabin.fillna(value = 'X', inplace = True)
merged.Cabin = merged.Cabin.apply( lambda x : x[0])
display(merged.Cabin.value_counts())

#In name we can obtain the honorifics
merged['Title'] = merged.Name.str.extract('([A-Za-z]+)\.')
display(merged.Title.value_counts())

merged.Title.replace(to_replace = ['Dr', 'Rev', 'Col', 'Major', 'Capt'], value = 'Officer', inplace = True)
merged.Title.replace(to_replace = ['Dona', 'Jonkheer', 'Countess', 'Sir', 'Lady', 'Don'], value = 'Aristocrat', inplace = True)
merged.Title.replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'}, inplace = True)
# absolute_and_relative_freq(merged.Title)

merged['Family_size'] = merged.SibSp + merged.Parch + 1  # Adding 1 for single person
display(merged.Family_size.value_counts())
merged.Family_size.replace(to_replace = [1], value = 'single', inplace = True)
merged.Family_size.replace(to_replace = [2,3], value = 'small', inplace = True)
merged.Family_size.replace(to_replace = [4,5], value = 'medium', inplace = True)
merged.Family_size.replace(to_replace = [6, 7, 8, 11], value = 'large', inplace = True)
# absolute_and_relative_freq(merged.Family_size)

ticket = []
for x in list(merged.Ticket):
    if x.isdigit():
        ticket.append('N')
    else:
        ticket.append(x.replace('.','').replace('/','').strip().split(' ')[0])

merged.Ticket = ticket


'''Count the categories in Ticket.'''
display(merged.Ticket.value_counts())
merged.Ticket = merged.Ticket.apply(lambda x : x[0])
display(merged.Ticket.value_counts())

'''After processing, visualise and count the absolute and relative frequency of updated Ticket.'''
# absolute_and_relative_freq(merged.Ticket)

# outliers(merged.Age)
# outliers(merged.Fare)

# mn.matrix(merged)
# plt.show()
# sns.heatmap(merged.isnull(), cbar=False)
# plt.show()

display(merged.isnull().sum())

merged.Embarked.fillna(value = 'S', inplace = True)
merged.Fare.fillna(value = merged.Fare.median(), inplace = True)

correlation = merged.loc[:, ['Sex', 'Pclass', 'Embarked', 'Title', 'Family_size', 'Parch', 'SibSp', 'Cabin', 'Ticket']]
# fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = (15,10))
# for ax, column in zip(axes.flatten(), correlation.columns):
#     sns.boxplot(x = correlation[column], y =  merged.Age, ax = ax)
#     ax.set_title(column, fontsize = 13)
#     ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
#     ax.tick_params(axis = 'both', which = 'minor', labelsize = 10)
#     ax.set_ylabel('Age', fontsize = 10)
#     ax.set_xlabel('')
# fig.suptitle('Variables Associated with Age', fontsize = 30)
# fig.tight_layout(rect = [0, 0.03, 1, 0.95])
# plt.show()


# from sklearn.preprocessing import LabelEncoder
# correlation = correlation.agg(LabelEncoder().fit_transform)
# correlation['Age'] = merged.Age # Inserting Age in variable correlation.
# correlation = correlation.set_index('Age').reset_index() # Move Age at index 0.

# '''Now create the heatmap correlation.'''
# plt.figure(figsize = (20,7))
# sns.heatmap(correlation.corr(), cmap ='BrBG', annot = True)
# plt.title('Variables Correlated with Age', fontsize = 18)
# plt.show()

merged.Age = merged.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))

#Skipping bivariate/multivariate analysis  for now

#We will bin our data to prevent overfitting
label_names = ['infant','child','teenager','young_adult','adult','aged']

cut_points = [0,5,12,18,35,60,81]
merged['Age_binned'] = pd.cut(merged.Age, cut_points, labels = label_names)

groups = ['low','medium','high','very_high']
cut_points = [-1, 130, 260, 390, 520]
merged['Fare_binned'] = pd.cut(merged.Fare, cut_points, labels = groups)


merged.drop(columns = ['Name', 'Age', 'Fare'], inplace = True, axis = 1)

display(merged.columns)

display(merged.dtypes)

merged.loc[:, ['Pclass', 'Sex', 'Embarked', 'Cabin', 'Title', 'Family_size', 'Ticket']] = merged.loc[:, ['Pclass', 'Sex', 'Embarked', 'Cabin', 'Title', 'Family_size', 'Ticket']].astype('category')

merged.Survived = merged.Survived.dropna().astype('int')#Converting without dropping NaN throws an error.

merged = pd.get_dummies(merged)
display(merged.head(2))




# TESTING TRAINING

seed = 43

df_train = merged.iloc[:891, :]
df_test  = merged.iloc[891:, :]

df_train = df_train.drop(columns = ['PassengerId'], axis = 1)
df_test = df_test.drop(columns = ['Survived'], axis = 1)

X_train = df_train.drop(columns = ['Survived'], axis = 1)
y_train = df_train['Survived']

X_test  = df_test.drop("PassengerId", axis = 1).copy()

"""Building machine learning models:
We will try 10 different classifiers to find the best classifier after tunning model's hyperparameters that will best generalize the unseen(test) data."""

'''Now initialize all the classifiers object.'''
'''#1.Logistic Regression'''
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

'''#2.Support Vector Machines'''
from sklearn.svm import SVC
svc = SVC(gamma = 'auto')

'''#3.Random Forest Classifier'''
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = seed, n_estimators = 100)

'''#4.KNN'''
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

'''#5.Gaussian Naive Bayes'''
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

'''#6.Decision Tree Classifier'''
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state = seed)

'''#7.Gradient Boosting Classifier'''
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(random_state = seed)

'''#8.Adaboost Classifier'''
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(random_state = seed)

'''#9.ExtraTrees Classifier'''
from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier(random_state = seed)

'''#10.Extreme Gradient Boosting'''
from xgboost import XGBClassifier
xgbc = XGBClassifier(random_state = seed)

'''Create a function that returns train accuracy of different models.'''
def train_accuracy(model):
    model.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    train_accuracy = np.round(train_accuracy*100, 2)
    return train_accuracy


train_accuracy = pd.DataFrame({'Train_accuracy(%)':[train_accuracy(lr), train_accuracy(svc), train_accuracy(rf), train_accuracy(knn), train_accuracy(gnb), train_accuracy(dt), train_accuracy(gbc), train_accuracy(abc), train_accuracy(etc), train_accuracy(xgbc)]})
train_accuracy.index = ['LR', 'SVC', 'RF', 'KNN', 'GNB', 'DT', 'GBC', 'ABC', 'ETC', 'XGBC']
sorted_train_accuracy = train_accuracy.sort_values(by = 'Train_accuracy(%)', ascending = False)
display(sorted_train_accuracy)

def x_val_score(model):
    from sklearn.model_selection import cross_val_score
    x_val_score = cross_val_score(model, X_train, y_train, cv = 10, scoring = 'accuracy').mean()
    x_val_score = np.round(x_val_score*100, 2)
    return x_val_score

x_val_score = pd.DataFrame({'X_val_score(%)':[x_val_score(lr), x_val_score(svc), x_val_score(rf), x_val_score(knn), x_val_score(gnb), x_val_score(dt), x_val_score(gbc), x_val_score(abc), x_val_score(etc), x_val_score(xgbc)]})
x_val_score.index = ['LR', 'SVC', 'RF', 'KNN', 'GNB', 'DT', 'GBC', 'ABC', 'ETC', 'XGBC']
sorted_x_val_score = x_val_score.sort_values(by = 'X_val_score(%)', ascending = False)
display(sorted_x_val_score)

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": rf.predict(X_test)})
submission.to_csv('submission_rf.csv', index = False)
