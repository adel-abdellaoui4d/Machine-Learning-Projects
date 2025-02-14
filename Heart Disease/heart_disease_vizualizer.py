import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

def plotting():
    sns.set(style='whitegrid',palette='muted',font_scale=1.5)
    rcParams['figure.figsize'] = 7, 4
    data = pd.read_csv("heart.csv")

    # Plot Heart disease presense distribution
    f = sns.countplot(x='target',data=data)
    f.set_title("Heart disease presense distribution")
    f.set_xticklabels(['No Heart disease', 'Heart Disease'])
    plt.xlabel("")
    plt.show()

    # Plot Heart disease presence by gender
    f = sns.countplot(x='target', data=data, hue='sex')
    plt.legend(['Female','Male'])
    f.set_title("Heart disease presence by gender")
    f.set_xticklabels(['No Heart disease', 'Heart Disease'])
    plt.xlabel("")
    plt.show()

    # scatter plot 
    plt.scatter(x=data.age[data.target==1], y=data.thallach[(data.target==1)],c="red",s=60)
    plt.scatter(x=data.age[data.target==0],y=data.thallach[(data.target==0)],s=60)
    plt.legend(["Disease","No Disease"])
    plt.xlabel("Age")
    plt.ylabel("Maximum Heart Rate")
    plt.show()

    f = sns.countplot(x='cp',data=data, hue='target')
    f.set_xticklabels(['Typical Angina','Atypical Angina', 'Non-anginal Pain','Asymptomatic'])
    f.set_title('Disease presence by chest pain type')
    plt.ylabel('Chest Pain Type')
    plt.xlabel('')
    plt.legend(['No disease','Disease'])
    plt.show()







