import numpy as np
import pandas as pd
import datetime
import ipywidgets as widgets
import matplotlib.pyplot as plt

#The interactive plots is made:
def _plot_timeseries(dataframes, variable, sex): #making the timeseries graph
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(1,1,1)

    if sex == 'Men':
        dataframe= dataframes[0]
    else:
        dataframe= dataframes[1]
    
    I = dataframe["EDUCATION LEVEL"] == variable #choseing the variable, that is the interactive element
    x = dataframe.loc[I,"YEAR"] 
    y = dataframe.loc[I,"UNEMPLOYED"]
    ax.plot(x,y, label = f'SEX: {sex}, EDUCATION: {variable}')
    ax.legend(frameon=True, loc = 'center left', bbox_to_anchor=(1, 0.5)) 
    
def plot_timeseries(dataframes): #making the interactive element
    widgets.interact(_plot_timeseries, 
    dataframes = widgets.fixed(dataframes), #choosing the dataset, used in the grafh
    variable = widgets.Dropdown( #Chooses what variabel is used in the interactive plot
        description='EDUCATION LEVEL', #namine the interactive element
        options=['H40 Short cycle higher education','H50 Vocational bachelors educations','H70 Masters programs'], #Defining which value 
        #to choose form in the interactive element
        value='H40 Short cycle higher education'),  #defining what the model chooses as the first variable, when showing the graph 
    sex = widgets.Dropdown(
        description='SEX',
        options=['Men','Women'],
        value='Men'),          
); 