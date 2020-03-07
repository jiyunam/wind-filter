'''
Visualize the dataset
'''

import numpy as np
import pandas as pd


def visualize_data(dataset):
    good_data = dataset['good']
    bad_data = dataset['bad']

    # Let's print the statistics of the data between good vs. bad
    stats_good = good_data.describe(include=[np.number])
    stats_bad = bad_data.describe(include=[np.number])

    # Let's make a pie chart of each of the categorical data
    projects_good = good_data['Project Name'].value_counts()
    projects_bad = bad_data['Project Name'].value_counts()
    projects_both = pd.concat((projects_good, projects_bad), axis=1, sort=True)
    projects_both.columns = ['Projects from Good Dataset', 'Projects from Bad Dataset']

    return stats_good, stats_bad, projects_both

