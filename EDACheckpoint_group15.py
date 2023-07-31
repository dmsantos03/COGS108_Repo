#!/usr/bin/env python
# coding: utf-8

# # UC San Diego: Data Science in Practice - EDA Checkpoint
# ### Summer Session I 2023 | Instructor : C. Alex Simpkins Ph.D.
# 
# ## Draft project title if you have one (can be changed later)

# (This checkpoint helps you to perform your EDA on your data for your project. You can remove this text description. Consider this the next step in your final project. See the project readme for bullet points to check off in terms of details to include beyond the main section heading content below.)

# # Names
# 
# - Dave Santos
# - Jose Ortega

# <a id='research_question'></a>
# # Research Question

# How does the presence of fast food restaurants and other food establishments correlate with obesity rates in adolescents?

# # Setup

# In[1]:


## YOUR CODE HERE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
# To deal with these missing values, we used the fillna('') function. This function replaces all missing values in the dataframe with an empty string. It's worth mentioning that this approach might not be the best fit for every scenario. If we're dealing with numerical data, replacing missing values with an empty string could skew our data. In such cases, it might be better to fill in missing values with the mean, median, or mode, or use data imputation techniques.
# 
# Next, we noticed that some column names had HTML tags in them. To clean this up, we used the str.replace() function to replace all occurrences of '' and '' tags with an empty string, effectively removing them.
# 
# For the first dataframe (df), we had some rows that weren't needed for our analysis. We used the drop() function to remove these rows (rows 4 to 23) from our dataframe.
# 
# In the second dataframe (df2), we decided to rename some columns for clarity. We created a dictionary with the current column names as keys and the new column names as values. Then, we used the rename() function to apply these new names to the dataframe.
# 
# Finally, to make sure our cleaning steps worked as expected, we displayed the first few rows of the cleaned first dataframe using the head(20) function. This gave us a quick snapshot of our cleaned data.

# In[3]:


## YOUR CODE HERE
## FEEL FREE TO ADD MULTIPLE CELLS PER SECTION

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


# In this step, we're reshaping our data to make it easier to work with. We start by creating a new dictionary, 'reform_data1', that holds the data we want to reformat. We then convert this dictionary into a pandas DataFrame, which we'll call 'df3'.
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


df3


# In[9]:


df4


# Now we have two dataframes properly cleaned and organized for readability and ready for EDA.

# # Data Analysis & Results (EDA)

# Carry out EDA on your dataset(s); Describe in this section

# In[10]:


## YOUR CODE HERE
## FEEL FREE TO ADD MULTIPLE CELLS PER SECTION


# Diving into our EDA, we're going to start by creating a bar plot. This will help us visualize the average 'Total_n' for each type of 'Establishment' and whether they're present or not in the DataFrame df4.
# 
# First, we group our DataFrame by 'Establishment' and 'Presence'. Then, we calculate the mean of 'Total_n' for each group. We use the groupby() function to do this, and then unstack the result. This gives us a new DataFrame, which we'll call 'grouped'.
# 
# Next, we create an array 'x'. This will represent the position of each bar along the x-axis in our plot.
# 
# Now, it's time to create our plot. We set the figure size, and then plot two sets of bars: one for when 'Presence' is 'No', and one for when 'Presence' is 'Yes'. We offset the bars by 0.2 units on the x-axis to make sure they don't overlap, and set their width to 0.4 units.
# 
# We then set the x-axis labels to the 'Establishment' values from our 'grouped' DataFrame. We rotate these labels by 90 degrees to make them easier to read.
# 
# We add a legend to our plot to help distinguish between the 'No' and 'Yes' bars. We also add a title and labels to the x and y axes.
# 
# We display our plot and the 'grouped' DataFrame, so you can see the data in a table as well as in the plot.
# 
# This visualization gives us a comparative view of the average 'Total_n' for different food establishments, split by whether these establishments are present or not.

# In[11]:


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

# In[12]:


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


# In[13]:


df3


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
# It's important to note that these observations are based on the provided data and visualizations, and while they suggest certain trends, they do not establish causation. Other factors such as socioeconomic status, physical activity levels, and dietary habits can also significantly influence obesity rates among adolescents. Therefore, further research and more comprehensive data would be needed to draw definitive conclusions.

# The age data provides some interesting insights into the prevalence of overweight and obesity among different age groups of adolescents.
# 
# Age Group 12-15: This age group makes up approximately 51.4% of the total population in the study. Among these, around 66.8% are not overweight or obese, while 33.2% are overweight or obese. This suggests that a significant portion of adolescents in this age group are dealing with weight-related health issues.
# 
# Age Group 16-19: This age group comprises about 48.6% of the total population. Interestingly, the percentage of non-overweight or non-obese individuals in this group is higher (74.3%) compared to the younger age group. Conversely, the percentage of overweight or obese individuals is lower (25.7%).
# 
# From this data, we can infer that the prevalence of overweight and obesity seems to decrease as adolescents grow older (from the 12-15 age group to the 16-19 age group). This could be due to a variety of factors such as changes in dietary habits, increased physical activity, or heightened awareness and concern about body image and health among older adolescents.

# In[ ]:





# In[ ]:




