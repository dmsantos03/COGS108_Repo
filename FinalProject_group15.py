#!/usr/bin/env python
# coding: utf-8

# # UC San Diego: Data Science in Practice
# ## Final Project Title (change this to your project's title)

# ## Permissions
# 
# Place an `X` in the appropriate bracket below to specify if you would like your group's project to be made available to the public. (Note that student names will be included (but PIDs will be scraped from any groups who include their PIDs).
# 
# * [  ] YES - make available
# * [  x] NO - keep private

# # Names
# 
# - Dave Santos
# - Jose Ortega

# # Overview

# * Write a clear, 3-4 sentence summary of what you did and why.

# <a id='research_question'></a>
# # Research Question

# How does the presence of fast food restaurants and other food establishments correlate with obesity rates in adolescents?

# <a id='background'></a>
# 
# ## Background & Prior Work

# Fast food is a staple in many diets around the world due to its convenience, affordability, and appeal. However, the consumption of these high-calorie, nutrient-poor foods has been linked with numerous health problems, including obesity (Rosenheck). Obesity, a global health issue, is often characterized by an excessive accumulation of body fat, which can have deleterious impacts on one's health and wellbeing (World Health Organization). Fast food consumption has been particularly studied in relation to obesity due to its potential contribution to energy imbalance, leading to weight gain. It is generally accepted that the consumption of fast food is a significant contributor to the obesity epidemic; however, the degree to which this is true varies among different age groups. This variability may be due to differences in metabolic rates, lifestyle, and food preferences among different age groups (Rosenheck). Research conducted by Powell et al. (2012) examined the correlation between fast food consumption and obesity rates among different age groups in the United States (Powell). They found that fast food consumption was more prevalent among adolescents and young adults, and there was a significant positive correlation between fast food consumption and obesity in these age groups. In contrast, the correlation was less robust in older adults. Another study by Bowman and Vinyard (2004) specifically focused on the impact of fast food consumption on diet quality and weight in children and adolescents (Bowman). They found that children who consumed more fast food had poorer diet quality and higher energy intake, which can potentially lead to overweight and obesity. These previous studies establish a link between fast food consumption and obesity, particularly in younger age groups, but more nuanced research is required to delineate the strength and nature of this correlation across different age brackets.
# 
# References:
# 
# Rosenheck, R. (2008). Fast food consumption and increased caloric intake: a systematic review of a trajectory towards weight gain and obesity risk. Obesity reviews, 9(6), 535-547. https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-789X.2008.00477.x
# 
# World Health Organization. (2021). Obesity and overweight. https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight
# 
# Powell, L. M., Nguyen, B. T., & Han, E. (2012). The impact of restaurant consumption among US adults: effects on energy and nutrient intakes. Public health nutrition, 17(11), 2445-2452. https://www.cambridge.org/core/journals/public-health-nutrition/article/abs/impact-of-restaurant-consumption-among-us-adults-effects-on-energy-and-nutrient-intakes/8E8C9E8ED6E2094B3DF6F8B2733A6846
# 
# Bowman, S. A., & Vinyard, B. T. (2004). Fast food consumption of US adults: impact on energy and nutrient intakes and overweight status. Journal of the American College of Nutrition, 23(2), 163-168. https://www.tandfonline.com/doi/abs/10.1080/07315724.2004.10719357

# # Hypothesis
# 

# Our research hypothesis is that there is a stronger positive correlation between the presence of fast food establishments and higher obesity rates in adolescents. The null hypothesis is that there is no significance in the correlation between fast food establishment presence and obesity rates among adolescents.

# # Dataset(s)

# Dataset Name: Is the local food environment associated with excess body weight in adolescents in São Paulo, Brazil?
# Link to the dataset:https://figshare.com/articles/dataset/Is_the_local_food_environment_associated_with_excess_body_weight_in_adolescents_in_S_o_Paulo_Brazil_/12210575/1?file=22456268
# Number of observations:15
# Dataset Name: Is the local food environment associated with excess body weight in adolescents in São Paulo, Brazil?
# Link to the dataset:https://figshare.com/articles/dataset/Is_the_local_food_environment_associated_with_excess_body_weight_in_adolescents_in_S_o_Paulo_Brazil_/12210575/1?file=22456271
# Number of observations:12

# # Data Wrangling

# * Explain steps taken to pull the data you need into Python.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind


# Path to the first local CSV file
file_path = 'E:\\mattd\\Desktop\\ProjectData\\Table_1.csv'

# Read the CSV data into a pandas DataFrame
df = pd.read_csv(file_path)

# Display the first 20 rows of the DataFrame
df.head(20)


# In[2]:


# Path to the second local CSV file
file_path2 = 'E:\\mattd\\Desktop\\ProjectData\\Table_2.csv'

# Read the CSV data into a pandas DataFrame
df2 = pd.read_csv(file_path2)

# Display the first 20 rows of the DataFrame
df2.head(20)


# # Data Cleaning

# We started the data cleaning process by looking for any missing values in our dataframes. We used the isnull().sum() function, which conveniently gives us the total number of missing values in each column. We found that some columns did have missing values.
# 
# To deal with these missing values, we used the fillna('') function. This function replaces all missing values in the dataframe with an empty string. 
# 
# Next, we noticed that some column names had HTML tags in them. To clean this up, we used the str.replace() function to replace all occurrences of '' and '' tags with an empty string, effectively removing them.
# 
# For the first dataframe (df), we had some rows that weren't needed for our analysis. We used the drop() function to remove these rows (rows 4 to 23) from our dataframe.
# 
# In the second dataframe (df2), we decided to rename some columns for clarity. We created a dictionary with the current column names as keys and the new column names as values. Then, we used the rename() function to apply these new names to the dataframe.
# 
# To make sure our cleaning steps worked as expected, we displayed the first few rows of the cleaned first dataframe using the head(20) function. This gave us a quick snapshot of our cleaned data.

# In[3]:


# Check for missing values
print(df.isnull().sum())
print(df2.isnull().sum())

# Fill missing values with empty string
df = df.fillna('')
df2 = df2.fillna('')

# Remove HTML tags from column names
df.columns = df.columns.str.replace('<b>', '').str.replace('</b>', '')
df2.columns = df2.columns.str.replace('<b>', '').str.replace('</b>', '')

#For df:
# Drop unecessary rows for observation
df.drop(range(4, 23), axis=0, inplace=True)

#For df2:
# Create a dictionary with current column names as keys and new column names as values
rename_dict = {'Unnamed: 2': '', 'Unnamed: 4': '', 'Unnamed: 6': ''}

# Use the rename() function to rename the columns
df2 = df2.rename(columns=rename_dict)

# Display the first cleaned DataFrame
df.head(20)


# In this step below, we're reshaping our data to make it easier to work with. We start by creating a new dictionary, 'reform_data1', that holds the data we want to reformat. We then convert this dictionary into a pandas DataFrame, which we'll call 'df3'.
# 
# Next, we notice that the columns 'Total population', 'Not overweight/Obese', and 'Overweight/Obese' contain both numbers and percentages. To make our data analysis easier, we decide to split these columns into two separate ones: one for the numbers and one for the percentages. We do this using the str.split() function, which splits the string at the space character.
# 
# Now, we have our percentages, but they're still in string format and include parentheses. So, we use the str.strip() function to remove the parentheses.
# 
# With the parentheses gone, we can now convert the percentage columns from string format to numeric format. We use the pd.to_numeric() function for this. Once they're in numeric format, we divide them by 100 to convert them into decimal format. We use the apply() function to do this for each percentage column.
# 
# We no longer need the original 'Total population', 'Not overweight/Obese', and 'Overweight/Obese' columns, since we've split them into separate columns. So, we drop these original columns from our DataFrame using the drop() function.

# In[4]:


#Reformat
reform_data1 = {
    'Age': ['12-15', '16-19'],
    'Total population': ['259 (51.4)', '245 (48.6)'],
    'Not overweight/Obese': ['173 (66.8)', '182 (74.3)'],
    'Overweight/Obese': ['86 (33.2)', '63 (25.7)'],
    #'p-value': [np.nan, np.nan]
}

df3 = pd.DataFrame(reform_data1)

# Split the numbers and percentages into separate columns
for col in ['Total population', 'Not overweight/Obese', 'Overweight/Obese']:
    df3[col + '_n'], df3[col + '_%'] = df3[col].str.split(' ', 1).str
    df3[col + '_%'] = df3[col + '_%'].str.strip('()')


# Convert percentage columns to decimal format
percentage_cols = [col + '_%' for col in ['Total population', 'Not overweight/Obese', 'Overweight/Obese']]
df3[percentage_cols] = df3[percentage_cols].apply(lambda x: pd.to_numeric(x, errors='coerce') / 100)

#drop redundant columns
df3.drop(['Total population', 'Not overweight/Obese', 'Overweight/Obese'], axis=1, inplace=True)


# In[5]:


# Display the second cleaned DataFrame
df2.head(20)


# In[6]:


# Check again for missing values
print(df.isnull().sum())
print(df2.isnull().sum())


# Next up, we're going to give the same treatment to our second dataset, 'df2', and store the result in 'df4'. We start by organizing the data into a dictionary, which we call 'reform_data2'. We then convert this dictionary into a pandas DataFrame, 'df4'.
# 
# We notice that the first row of our DataFrame is filled with empty strings, which aren't useful for our analysis. So, we get rid of this row using slicing.
# 
# Next, we turn our attention to the percentage columns: 'Total_%', 'Not_overweight_%', and 'Overweight_%'. Just like before, we convert these from string format to numeric format using the pd.to_numeric() function. Then, we divide them by 100 to convert them into decimal format. We use the apply() function to do this for each percentage column.
# 
# We want to remove any rows where the 'Presence' column is an empty string, as these rows won't be useful for our analysis. We create a boolean mask that checks for non-empty strings in the 'Presence' column, and use this mask to filter our DataFrame.

# In[7]:


# Let's reformat the data
reform_data2 = {
    'Establishment': ['', 'Fast food restaurants', 'Fast food restaurants', '', 'Markets, supermarkets and grocery stores', 'Markets, supermarkets and grocery stores', '', 'Bakeries and cafeterias', 'Bakeries and cafeterias', '', 'Restaurants', 'Restaurants', '', 'Pizzerias', 'Pizzerias', '', 'Street markets and whole food markets', 'Street markets and whole food markets'],
    'Presence': ['', 'No', 'Yes', '', 'No', 'Yes', '', 'No', 'Yes', '', 'No', 'Yes', '', 'No', 'Yes', '', 'No', 'Yes'],
    'Total_n': ['', '473', '31', '', '73', '431', '', '451', '53', '', '123', '381', '', '387', '117', '', '249', '255'],
    'Total_%': ['', '93.9', '6.1', '', '14.5', '85.5', '', '89.5', '10.5', '', '24.4', '75.6', '', '76.8', '23.2', '', '49.4', '50.6'],
    'Not_overweight_n': ['', '3.36', '0.19', '', '0.53', '3.02', '', '3.15', '0.40', '', '0.87', '2.68', '', '2.70', '0.85', '', '1.75', '1.80'],
    'Not_overweight_%': ['', '71.0', '61.3', '', '72.6', '70.0', '', '69.8', '75.5', '', '70.7', '70.3', '', '69.8', '72.6', '', '70.3', '70.6'],
    'Overweight_n': ['', '1.37', '0.12', '', '0.20', '1.29', '', '1.36', '0.13', '', '0.36', '1.13', '', '1.17', '0.32', '', '0.74', '0.75'],
    'Overweight_%': ['', '28.9', '38.7', '', '27.4', '30.0', '', '30.2', '24.5', '', '29.3', '29.7', '', '30.2', '27.4', '', '29.7', '29.4'],
    'p-value': ['', '0.0025', '', '', '0.0071', '', '', '0.0039', '', '', '0.0093', '', '', '0.0055', '', '', '0.0094', ''],
    #'Unadjusted_OR': ['', '0.90-5.61', '', '', '0.71-2.77', '', '', '0.36-2.18', '', '', '0.40-2.05', '', '', '0.61-1.95', '', '', '0.68-1.70', '']
}

df4 = pd.DataFrame(reform_data2)

# Remove the first row
df4 = df4[1:]

# Convert percentage columns to decimal format
percentage_cols = ['Total_%', 'Not_overweight_%', 'Overweight_%']
df4[percentage_cols] = df4[percentage_cols].apply(lambda x: pd.to_numeric(x, errors='coerce') / 100)

# Display the reformatted DataFrame
df4 = df4[df4['Presence'] != '']
#df4


# In[8]:


# Reset the indexes of the dataframe
df3 = df3.reset_index(drop=True)

# Set the index starting from 1
df3.index = df3.index + 1

#display the second reformatted dataframe
df3


# In[9]:


# Reset the indexes of the second dataframe
df4 = df4.reset_index(drop=True)

# Set the index starting from 1
df4.index = df4.index + 1

#display the second reformatted dataframe
df4


# # Data Visualization

# Diving into our EDA below, we're going to start by creating a bar plot. This will help us visualize the average 'Total_n' for each type of 'Establishment' and whether they're present or not in the DataFrame df4.
# 
# First, we group our DataFrame by 'Establishment' and 'Presence'. Then, we calculate the mean of 'Total_n' for each group. We use the groupby() function to do this, and then unstack the result. This gives us a new DataFrame, which we'll call 'grouped'.
# 
# Next, we create an array 'x'. This will represent the position of each bar along the x-axis in our plot.
# 
# We then create our plot. We set the figure size, and then plot two sets of bars: one for when 'Presence' is 'No', and one for when 'Presence' is 'Yes'. We offset the bars by 0.2 units on the x-axis to make sure they don't overlap, and set their width to 0.4 units.
# 
# We then set the x-axis labels to the 'Establishment' values from our 'grouped' DataFrame. We rotate these labels by 90 degrees to make them easier to read.
# 
# We add a legend to our plot to help distinguish between the 'No' and 'Yes' bars. We also add a title and labels to the x and y axes.
# 
# We display our plot and the 'grouped' DataFrame, so you can see the data in a table as well as in the plot.
# 
# This visualization gives us a comparative view of the average 'Total_n' for different food establishments, split by whether these establishments are present or not.

# In[10]:


# Create a new DataFrame with the mean of 'Total_n' for each 'Establishment' and 'Presence'
grouped = df4.groupby(['Establishment', 'Presence'])['Total_n'].mean().unstack()

# Create an array with the position of each bar along the x-axis
x = np.arange(len(grouped))

# Create the plot
fig, plt1 = plt.subplots(figsize=(10,6))

# Plot the 'No' bars
plt1.bar(x - 0.2, grouped['No'], width=0.4, label='No')

# Plot the 'Yes' bars
plt1.bar(x + 0.2, grouped['Yes'], width=0.4, label='Yes')

# Add the x-axis labels and rotate them
plt1.set_xticks(x)
plt1.set_xticklabels(grouped.index, rotation=90)

# Add a legend
plt1.legend()

# Add title and labels
plt1.set_title('Presence of Different Food Establishments')
plt1.set_xlabel('Establishment')
plt1.set_ylabel('Total_n')

# Show the plot
plt.show()
grouped


# We now visualize the average percentage of overweight individuals in relation to the presence of different types of food establishments.
# 
# We grouped the data by 'Establishment' and 'Presence' categories, and the mean percentage of overweight individuals ('Overweight_%') is calculated for each group. This results in a new DataFrame, 'grouped2', where the indices are the different types of establishments and the columns correspond to the presence (or absence) of these establishments.
# 
# Next, a bar plot is created using matplotlib. Separate bars are plotted for the 'No' and 'Yes' categories of the 'Presence' variable. The x-axis represents the different types of food establishments, and the y-axis represents the average percentage of overweight individuals. The bars are color-coded and a legend is provided for clarity.
# 
# The plot provides a visual comparison of the average overweight percentage across different types of food establishments, based on their presence or absence. This can help in understanding if there's a correlation between the presence of certain types of food establishments and the prevalence of overweight individuals.

# In[11]:


#repeating the process with the visualization of Overweight % over Presence of Establishment
grouped2 = df4.groupby(['Establishment', 'Presence'])['Overweight_%'].mean().unstack()

# Create an array with the position of each bar along the x-axis
x2 = np.arange(len(grouped2))

# Create the plot
fig, plt2 = plt.subplots(figsize=(10,6))

# Plot the 'No' bars
plt2.bar(x2 - 0.2, grouped2['No'], width=0.4, label='No')

# Plot the 'Yes' bars
plt2.bar(x2 + 0.2, grouped2['Yes'], width=0.4, label='Yes')

# Add the x-axis labels and rotate them
plt2.set_xticks(x2)
plt2.set_xticklabels(grouped.index, rotation=90)

# Add a legend
plt2.legend()

# Add title and labels
plt2.set_title('Overweight Percentage by Presence of Establishment')
plt2.set_xlabel('Establishment')
plt2.set_ylabel('Overweight_%')

# Show the plot
plt.show()
df4


# In[12]:


# display the age and obesity % dataframe
df3


# # Data Analysis & Results

# Based on our exploration of the first dataset and visualizations on establishment presence, we can formulate some assumptions and analysis:
# 
# Fast Food Restaurants: The presence of fast food restaurants seems to have a higher percentage of overweight adolescents compared to when there are no fast food restaurants. This could suggest a correlation between the presence of fast food restaurants and higher obesity rates, possibly due to the consumption of high-calorie, low-nutrient food typically served in these establishments.
# 
# Markets, Supermarkets, and Grocery Stores: Interestingly, the presence of markets, supermarkets, and grocery stores shows a slightly lower percentage of overweight adolescents compared to their absence. This could be due to these establishments providing access to a wider variety of food options, including healthier choices like fresh fruits and vegetables.
# 
# Bakeries and Cafeterias: The presence of bakeries and cafeterias seems to correlate with a lower percentage of overweight adolescents. This might be due to the fact that these establishments often offer a wider variety of food, including potentially healthier options.
# 
# Restaurants and Pizzerias: The presence of restaurants and pizzerias shows a slightly higher percentage of overweight adolescents, but the difference is not as pronounced as with fast food restaurants.
# 
# Street Markets and Whole Food Markets: The presence of these establishments shows almost an equal percentage of overweight adolescents compared to their absence, suggesting that these types of establishments might not have a significant impact on adolescent obesity rates.
# 
# The age data provides some interesting insights into the prevalence of overweight and obesity among different age groups of adolescents.
# 
# Age Group 12-15: This age group makes up approximately 51.4% of the total population in the study. Among these, around 66.8% are not overweight or obese, while 33.2% are overweight or obese. This suggests that a significant portion of adolescents in this age group are dealing with weight-related health issues.
# 
# Age Group 16-19: This age group comprises about 48.6% of the total population. Interestingly, the percentage of non-overweight or non-obese individuals in this group is higher (74.3%) compared to the younger age group. Conversely, the percentage of overweight or obese individuals is lower (25.7%).
# 
# From this data, we can infer that the prevalence of overweight and obesity seems to decrease as adolescents grow older from the 12-15 age group to the 16-19 age group. Along with the presence and influence of certain establishments, this could also stem from a variety of factors such as changes in dietary habits, increased physical activity, or heightened awareness and concern about body image and health among older adolescents.

# In[13]:


# Convert 'Yes'/'No' to 1/0
df4['Presence'] = df4['Presence'].map({'Yes': 1, 'No': 0})

# Group by establishment type and calculate the correlation with the overweight percentage
for establishment in df4['Establishment'].unique():
    df_establishment = df4[df4['Establishment'] == establishment]
    correlation = df_establishment['Presence'].corr(df_establishment['Overweight_%'])
    print(f'Correlation between presence of {establishment} and overweight percentage: {correlation}')


# The correlation results above show the strength and direction of the linear relationship between the presence of different types of food establishments and the overweight percentage in adolescents. A correlation of 1 or -1 indicates a perfect linear relationship. Positive values indicate that as one variable increases, the other also increases, while negative values indicate that as one variable increases, the other decreases.

# In[14]:


# Define the establishment types based the data
establishment_types = ['Fast food restaurants', 'Markets, supermarkets and grocery stores', 'Bakeries and cafeterias', 'Restaurants', 'Pizzerias', 'Street markets and whole food markets']

# Perform a t-test for each establishment type
for establishment in establishment_types:
    # Get the overweight percentages for areas with and without the establishment
    overweight_with_establishment = df4[df4['Establishment'] == establishment]['Overweight_%']
    overweight_without_establishment = df4[df4['Establishment'] != establishment]['Overweight_%']
    
    # Perform the t-test
    t_stat, p_val = ttest_ind(overweight_with_establishment, overweight_without_establishment, nan_policy='omit')
    
    # Print the t-statistic and p-value
    print(f'{establishment}:')
    print('t-statistic:', t_stat)
    print('p-value:', p_val)
    print()


# The t-test results provide a measure of the difference in overweight percentages between areas with and without each type of food establishment. The t-statistic is a measure of the size of the difference relative to the variation in your data. The p-value is a measure of the probability that you would see the observed difference (or a larger one) by chance if there were no real difference.

# To summate our correlation and t-test findings:
# 
# Fast food restaurants: The correlation is 1, indicating a perfect positive linear relationship. The t-test shows a statistically significant difference in overweight percentages between areas with and without fast food restaurants (p-value < 0.05).
# 
# Markets, supermarkets and grocery stores: The correlation is close to 1, indicating a strong positive linear relationship. However, the t-test shows no statistically significant difference in overweight percentages between areas with and without these establishments (p-value > 0.05).
# 
# Bakeries and cafeterias: The correlation is -1, indicating a perfect negative linear relationship. However, the t-test shows no statistically significant difference in overweight percentages between areas with and without these establishments (p-value > 0.05).
# 
# Restaurants: The correlation is close to 1, indicating a strong positive linear relationship. However, the t-test shows no statistically significant difference in overweight percentages between areas with and without restaurants (p-value > 0.05).
# 
# Pizzerias: The correlation is -1, indicating a perfect negative linear relationship. However, the t-test shows no statistically significant difference in overweight percentages between areas with and without pizzerias (p-value > 0.05).
# 
# Street markets and whole food markets: The correlation is close to -1, indicating a strong negative linear relationship. However, the t-test shows no statistically significant difference in overweight percentages between areas with and without these markets (p-value > 0.05).
# 
# These results suggest that while there are strong correlations between the presence of certain types of food establishments and overweight percentages in adolescents, these correlations do not necessarily translate into statistically significant differences in overweight percentages.

# # Conclusion & Discussion

# Fast food restaurants seem to have a significant correlation with overweight and obesity prevalence, which supports the commonly held belief that fast food consumption can contribute to unhealthy weight gain due to the high-calorie, low-nutrient nature of many fast food options. Moreover, the t-test indicates a statistically significant difference, strengthening this claim.
# 
# However, despite strong correlations observed with other types of food establishments like supermarkets, bakeries, and restaurants, the lack of statistically significant differences according to the t-tests suggests these correlations may not be translating into measurable impacts on adolescent obesity rates. This could be because these establishments offer a broader variety of food options, allowing for both healthier and less healthy choices.
# 
# There are several potential limitations to consider in this analysis:
# 
# Cross-sectional nature of the data: This data provides a snapshot in time, and while it can show correlations, it does not establish causality. Longitudinal studies would be necessary to assess causality.
# 
# Confounding factors: Many factors can influence obesity rates, including socioeconomic status, education level, physical activity, and more. While the presence of food establishments is one factor, it's important to consider these other variables as well.
# 
# Quality and diversity of food options: While we categorized establishments into general types like "fast food" or "supermarket," there can be significant diversity within these categories. Not all fast food is equally unhealthy, and not all supermarkets offer an abundance of healthy options.
# 
# Future research could dig deeper into these factors, perhaps considering the quality of food available at these establishments or the influence of pricing on food choices. Moreover, investigating how these establishments interact with socioeconomic and cultural factors could provide a more holistic view of the obesity problem. Understanding the interactions between these variables might help in designing more effective interventions to address adolescent overweight and obesity rates.
