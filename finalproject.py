import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st
import matplotlib.pyplot as plt

def cleandata(data):
    data = data.drop(labels=0, axis=0)

    data = data.set_index('Variable')

    data = data.drop(data.loc[:,'GP_16':].columns, axis=1)

    zeroGP =[]
    for index, row in data.iterrows():
        if row['GP_17'] == '0':
            zeroGP.append(index)

    data = data.drop(labels=zeroGP, axis=0)

    columnstokeep = ['Position', 'Shots_On_Target17', 'Offsides17', 'Passes17', 'Fouls17']
    columnstoremove = []

    for column in data.columns:
        if column not in columnstokeep:
            columnstoremove.append(column)

    data = data.drop(labels=columnstoremove, axis=1)

    positiondict = {'GK':0, 'D':1, 'M':2, 'F':3}

    data['PositionValue']= data['Position'].map(positiondict)
    data = data.drop(labels='Position', axis=1)

    for rowIndex, row in data.iterrows():
        for columnIndex, value in row.items():
            value = int(value)
    return data

def choice(option, data):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    if option == 'Position and Fouls':
        #correlation between position and fouls

        position = np.array(data.PositionValue).reshape((-1,1)).astype(int)
        fouls = np.array(data.Fouls17).astype(int)
        model = LinearRegression().fit(position, fouls)
        r_sq = model.score(position, fouls)
        intercept = model.intercept_
        slope = float(model.coef_)
        st.write('Coefficient of Determination:', r_sq)
        st.write('Intercept:', intercept)
        st.write('Slope:', slope)
        plt.scatter(position, fouls)
        plt.plot(position, slope * position + intercept)
        plt.xticks([0,1,2,3],['GK','D','M','F'])
        plt.xlabel('Positions')
        plt.ylabel('Number of Fouls in a Season')
        plt.title('Linear Regression Model of Player Position and Number of Fouls in the 2016-2017 Premier League Season')
        st.pyplot()

    if option == 'Position and Passes':
        #correlation between position and passes

        position = np.array(data.PositionValue).reshape((-1,1)).astype(int)
        passes = np.array(data.Passes17).astype(int)
        model = LinearRegression().fit(position, passes)
        r_sq = model.score(position, passes)
        intercept = model.intercept_
        slope = float(model.coef_)
        st.write('Coefficient of Determination:', r_sq)
        st.write('Intercept:', intercept)
        st.write('Slope:', slope)
        plt.scatter(position, passes)
        plt.plot(position, slope * position + intercept)
        plt.xticks([0,1,2,3],['GK','D','M','F'])
        plt.xlabel('Positions')
        plt.ylabel('Number of Passes in a Season')
        plt.title('Linear Regression Model of Player Position and Number of Passes in the 2016-2017 Premier League Season')
        st.pyplot()

    if option == 'Offsides and Shots on Target':
        #correlation between offsides and shots on target

        offsides = np.array(data.Offsides17).reshape((-1,1)).astype(int)
        shotsontarget = np.array(data.Shots_On_Target17).astype(int)
        model = LinearRegression().fit(offsides, shotsontarget)
        r_sq = model.score(offsides, shotsontarget)
        intercept = model.intercept_
        slope = float(model.coef_)
        st.write('Coefficient of Determination:', r_sq)
        st.write('Intercept:', intercept)
        st.write('Slope:', slope)
        plt.scatter(offsides, shotsontarget)
        plt.plot(offsides, slope * offsides + intercept)
        plt.xlabel('Number of Offsides in a Season')
        plt.ylabel('Number of Shots on Target in a Season')
        plt.title('Linear Regression Model of Number of Offsides and Number of Shots on Target in the 2016-2017 Premier '
                  'League Season')
        st.pyplot()

def main():
    data = pd.read_csv('PremierLeagueData2016-2017and2017-2018.csv')
    data = cleandata(data)
    st.header('EPL Dashboard of Player Statistics')
    correlations = ['Position and Fouls', 'Position and Passes', 'Offsides and Shots on Target']
    option = st.selectbox('Which linear regression would you like to see?', correlations)
    choice(option, data)

main()
