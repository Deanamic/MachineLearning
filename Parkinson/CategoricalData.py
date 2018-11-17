import matplotlib.pyplot as plt #data visualization
import pandas as pd
from IPython.display import Markdown, display

#Bar Graph Absolute Scale
def abs_bar_labels(ax, font_size=15):
    plt.ylabel('Absolute Frequency', fontsize = font_size)
    plt.xticks(rotation = 0, fontsize = font_size)
    plt.yticks([])

    # Set individual bar lebels in absolute number
    for x in ax.patches:
        ax.annotate(x.get_height(),
        (x.get_x() + x.get_width()/2., x.get_height()), ha = 'center', va = 'center', xytext = (0, 7),
        textcoords = 'offset points', fontsize = font_size, color = 'black')

#Bar graph in Relative scale
def pct_bar_labels(ax, font_size = 15):
    plt.ylabel('Relative Frequency (%)', fontsize = font_size)
    plt.xticks(rotation = 0, fontsize = font_size)
    plt.yticks([])

    # Set individual bar lebels in proportional scale
    for x in ax.patches:
        ax.annotate(str(x.get_height()) + '%',
        (x.get_x() + x.get_width()/2., x.get_height()), ha = 'center', va = 'center', xytext = (0, 7),
        textcoords = 'offset points', fontsize = font_size, color = 'black')

'''#3.Function to create a dataframe of absolute and relative frequency of each variable. And plot absolute and relative frequency.'''
def absolute_and_relative_freq(variable):
    # Dataframe of absolute and relative frequency
    absolute_frequency = variable.value_counts()
    relative_frequency = round(variable.value_counts(normalize = True)*100, 2)
    # Was multiplied by 100 and rounded to 2 decimal points for percentage.
    df = pd.DataFrame({'Absolute Frequency':absolute_frequency, 'Relative Frequency(%)':relative_frequency})
    print('Absolute & Relative Frequency of',variable.name,':')
    display(df)

    # This portion plots absolute frequency with bar labeled.
    fig_size = (18,5)
    font_size = 15
    title_size = 18
    ax =  absolute_frequency.plot.bar(title = 'Absolute Frequency of %s' %variable.name, figsize = fig_size)
    ax.title.set_size(title_size)
    abs_bar_labels(ax)  # Displays bar labels in abs scale.
    plt.show(block = False)
    plt.figure(2)
    # This portion plots relative frequency with bar labeled.
    ax1 = relative_frequency.plot.bar(title = 'Relative Frequency of %s' %variable.name, figsize = fig_size)
    ax1.title.set_size(title_size)
    pct_bar_labels(ax1) # Displays bar labels in relative scale.
    plt.show()
