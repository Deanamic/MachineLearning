import matplotlib.pyplot as plt
import seaborn as sns #data visualization
import pandas as pd

def outliers(variable, title_size = 15, font_size = 10, grid=False):
    # global filtered
    # Calculate 1st, 3rd quartiles and iqr.
    q1, q3 = variable.quantile(0.25), variable.quantile(0.75)
    iqr = q3 - q1

    # Calculate lower fence and upper fence for outliers
    l_fence, u_fence = q1 - 1.5*iqr , q3 + 1.5*iqr   # Any values less than l_fence and greater than u_fence are outliers.

    # Observations that are outliers
    outliers = variable[(variable<l_fence) | (variable>u_fence)]
    print('Total Outliers of', variable.name,':', outliers.count())

    # Drop obsevations that are outliers
    filtered = variable.drop(outliers.index, axis = 0)

    # Create subplots
    curV = variable.to_frame(variable.name)
    curF = filtered.to_frame(variable.name)
    curV['DataType'] = 'Raw'
    curF['DataType'] = 'Without Outliers'
    newSeries = pd.concat([curV, curF])
    sns.set_style('darkgrid')
    bplot = sns.boxplot(y=variable.name, x='DataType',data=newSeries,width=0.5)
    plt.show(block = False)
    plt.figure(2)
    bplot = sns.boxplot(y=variable.name, x='DataType',data=curF,width=0.5)
    plt.show();
