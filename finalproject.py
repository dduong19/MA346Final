
# We first must import the necessary libraries that will be used later on for cleaning the data and running statistical
# analyses.

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st
import matplotlib.pyplot as plt

# Before going to the main function, we defined a couple of other functions. The first function we defined was created to
# clean the data. This function takes the parameter 'data' which is the dataframe that is sent to the function for cleaning.

def cleandata(data):

# When looking at the full dataframe, we noticed that there was a row called description that with the index value 0. We
# felt that this row wasn't necessary for our analysis because it just had descriptions of each variable in the dataset.
# Therefore, we dropped this row using the .drop() function, defining our parameters with the index of 0 and the axis being
# 0 because it was a row.

    data = data.drop(labels=0, axis=0)

# Once this description row was removed, we wanted to set the player's name column as the index value. We wanted to do this
# so that every time we want to refer to a row, we can refer to it by the player's name, instead of the row's index number.
# To accomplish this, we used the .set_index() function, removing the column called 'Variable' which was the player's name
# column.

    data = data.set_index('Variable')

# Now that are dataset was indexed correctly and contained purely statistics, we wanted to drop any columns or rows that
# provided no other use for our analyses. The first columns we wanted to drop was any that didn't have anything to do
# with the 2017-2018 season. To do this, we used the drop.() function in combination with the .loc() function. We
# found that the first column that was irrelevant to the 2017-2018 season was called 'GP_16", so we located that column
# using .loc() and dropped all the columns from this column to the end of the dataframe using .drop().

    data = data.drop(data.loc[:,'GP_16':].columns, axis=1)

# Next, we wanted to remove any rows of players that had zero games played because we only wanted players that actually
# played during the season. To do this, we created an empty list called zeroGP. We iterated through the dataframe and
# found any rows that had a value of 0 for the column 'GP_17'. If a row met this criteria, it was added to the zeroGP
# list.

    zeroGP =[]
    for index, row in data.iterrows():
        if row['GP_17'] == '0':
            zeroGP.append(index)

# Once this list was complete, we used the .drop() function to drop the columns in the list from the dataset.

    data = data.drop(labels=zeroGP, axis=0)

# Next, we wanted to remove any columns that contain stats that we weren't  using. To do this we created a list with
# the columns we wanted to keep. This column contained the columns: 'Position', 'Shots_On_Target17', 'Offsides17',
# 'Passes17', 'Fouls17'. We then created an empty list for columns we wanted to remove. We used a for loop to loop
# through the columns in the dataset, and if the column name wasn't in the columns to keep list, it was added to the
# columns to remove list.

    columnstokeep = ['Position', 'Shots_On_Target17', 'Offsides17', 'Passes17', 'Fouls17']
    columnstoremove = []

    for column in data.columns:
        if column not in columnstokeep:
            columnstoremove.append(column)

# Once this columns to remove list was complete, we used the .drop() function to drop all the columns from the dataset
# that this list contained.

    data = data.drop(labels=columnstoremove, axis=1)

# Because we wanted to eventually run regressions on our variables, we wanted all our variables to be quantitative.
# However, our position variable had qualitative variables of GK, D, M, and F. Therefore, we wanted to convert these
# qualitative variables to quantitative variables. To do this, we created a dictionary with the qualitative position
# variables as keys and arbitrary integers as values. We made these integer values in a way that the higher the number,
# the higher up the field the player's position is.

    positiondict = {'GK':0, 'D':1, 'M':2, 'F':3}

# With this dictionary created, we used the .map() function to take the qualitative values in the position column and
# create a new column that has the respective position quantitative value using the dictionary's predefined values. This
# new column with the quantitative position values was called 'PositionValue'.

    data['PositionValue']= data['Position'].map(positiondict)

# Because we no longer needed the original position column with the qualitative values, we dropped it using the .drop()
# function.

    data = data.drop(labels='Position', axis=1)

# After all this cleaning, our data was finally ready for statistical analysis. Therefore, we return the fully-cleaned
# data back to the main function.

    return data

# The second function we defined was a choice function. This function takes the correlation that the user chose to inspect
# in the main function, and based on this option, the appropriate linear regression model and statistics will be shown on
# the dashboard. Therefore, this function takes two parameters. The first parameter is the correlation option that the user
# chooses, and the second function is the dataset that the data is being taken from.

def choice(option, data):

# We used the st.set_option to remove a warning message on our dashboard.

    st.set_option('deprecation.showPyplotGlobalUse', False)

# Below we used three if statements to portray the correct correlation based on the user's input. The first if statement
# checks if the option is 'Position and Fouls'.

    if option == 'Position and Fouls':

# If it is, then it takes the position value column of the dataset and turns it into an array using the np.array function.
# With this array, we used the .reshape() function to make the array the right shape for later analysis. We also used
# .astype() function we turned each element of the array into an integer. Once this array was set, we then converted the
# fouls column of the dataset into an array. Again, we used the .astype() function to turn each element into an integer.

        position = np.array(data.PositionValue).reshape((-1,1)).astype(int)
        fouls = np.array(data.Fouls17).astype(int)

# Now that our x and y variables are set in the form of arrays. We use the LinearRegression() function from the previously
# imported sk.linear_model library. This function fits a linear regression model for the x and y variables we set. We stored
# this model under the variable 'model'.

        model = LinearRegression().fit(position, fouls)

# With this model, we used the three functions, .score(), .intercept_, and .coef_, to obtain the coefficient of determination,
# the y-intercept, and the slope, respectively.

        r_sq = model.score(position, fouls)
        intercept = model.intercept_
        slope = float(model.coef_)

# With these obtained stats, we used st.write() to write the stats onto the dashboard.

        st.write('Coefficient of Determination:', r_sq)
        st.write('Intercept:', intercept)
        st.write('Slope:', slope)

# Finally, we plotted the x and y variables onto a scatter plot, with position and fouls on the x and y-axis, respectively.
# To plot to a scatterplot we used the .scatter function that came with the previously imported matplotlib.pyplot library.
# We also plotted a line of best fit by using the .plot() function. To do this, we used the parameters position as the
# x-values and we plotted the line using the formula y=mx+b. We used position as x, the slope as m, the y-intercept as b.

        plt.scatter(position, fouls)
        plt.plot(position, slope * position + intercept)

# With our plot created, we just added some customizations to the chart. We first rewrote the xticks using the .xticks()
# function. We did this so that we could change the numerical position values back to their qualitative values. We used
# the .xlabel() and the .ylabel() functions to create x-axis and y-axis labels. We used the .title() to give the plot a
# title. Finally, we used streamlit's .pyplot function to show the plot on our dashboard.

        plt.xticks([0,1,2,3],['GK','D','M','F'])
        plt.xlabel('Positions')
        plt.ylabel('Number of Fouls in a Season')
        plt.title('Linear Regression Model of Player Position and Number of Fouls in the 2016-2017 Premier League Season')
        st.pyplot()

# The second if statement checks if the option is 'Position and Passes'.

    if option == 'Position and Passes':

# The rest of the if statement does the same commands as the first if statement with the respective correlation. Refer
# to the first if statement to understand what the code does.

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

# The third if statement checks if the option is 'Offsides and Shots on Target'.

    if option == 'Offsides and Shots on Target':

# The rest of the if statement does the same commands as the first if statement with the respective correlation. Refer
# to the first if statement to understand what the code does.

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

# Finally we define the main function here.

def main():

# This line of code reads in the dataframe using the pandas funtion .read_csv() and stores the dataframe under the variable
# name 'data'.

    data = pd.read_csv('PremierLeagueData2016-2017and2017-2018.csv')

# We use the cleandata() function that we defined to clean our recently opened dataframe. We store the cleaned dataframe
# under the same variable name 'data'.

    data = cleandata(data)

# We imported the image function from the library PIL. We did this to post a picture to our dashboard. We opened an image
# with the .open() function and stored it under the variable name 'img'. Then we used streamlit's .image() function to
# post the picture to our dashboard with a width of 700 pixels.
    from PIL import Image
    img = Image.open("logo.jpeg")
    st.image(img, width=700)

# Here we are writing a header to our dashboard using streamlit's .header() function.

    st.header('EPL Dashboard of Player Statistics')

# Then we create a list called correlations that contains all the correlations that we are able to analyze. We then use
# streamlit's selectbox() funtion to create a drop down menu with the correlations from our list as the options. We store
# whatever option the user selects under the variable name 'option'.

    correlations = ['Position and Fouls', 'Position and Passes', 'Offsides and Shots on Target']
    option = st.selectbox('Which linear regression would you like to see?', correlations)

# Now that we have the user's selected correlation, we send that correlation and the dataframe up to the previously
# defined choice() function. When the function receives the these two parameters, the dashboard with show the linear
# regression model for the appropriate correlation.
    choice(option, data)

# Finally, we run our main function.
main()
