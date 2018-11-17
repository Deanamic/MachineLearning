import pandas as pd
import matplotlib.pyplot as plt #data visualization
from CategoricalData import *
from IPython.display import Markdown, display

#Histogram
def histogram(variable, font_size = 15, fig_size = (18,7), title_size = 18, block = True):
    ax = variable.plot.hist(figsize = fig_size)
    plt.xlabel('%s' %variable.name, fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plt.title('%s' %variable.name + ' Distribution with Histogram', fontsize = title_size)
    abs_bar_labels(ax)
    plt.show(block = block)

#Density
def density_plot(variable, font_size = 15, fig_size = (18,7), title_size = 18, block = True):
    plt.figure(figsize = fig_size)
    variable.plot.hist(density = True)
    variable.plot.kde(style = 'k--')
    plt.xlabel('%s'%variable.name, fontsize = font_size)
    plt.ylabel('Density', fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.title('%s ' %variable.name + 'Distribution with Density Plot & Histogram', fontsize = title_size)
    plt.show(block = block)

'''#3.Calculate descriptive statistics.'''
def summary_stats(variable):
    stats = variable.describe()
    skew = pd.Series(variable.skew(), index = ['skewness'])
    df_stats = pd.DataFrame(pd.concat([skew, stats], sort = False), columns = [variable.name])
    df_stats.index.name = 'Stats'
    display(df_stats)
