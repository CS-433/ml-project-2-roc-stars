import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Directory where the images will be saved
path = "plots/ethics/"


# Load dataset
df = pd.read_csv('data/Data_Patients.csv', sep=";", header=0)

# Plot histogram of participants Age
plt.hist(df['age'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age distribution across participants')
plt.savefig(path + "age.png")
plt.show()

#Plot Pie Chart of participants Gender
# Map 0, 1 to female and male 
df['sex'] = df['sex'].map({0: 'Female', 1: 'Male'})
#ATTENTION IL FAUT DEMANDER SI O = HOMME OUR FEMME
plt.figure(figsize=(8, 8))
age_counts = df['sex'].value_counts()
plt.pie(age_counts, labels=age_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title('Participants Gender')
plt.savefig(path + "gender.png")
plt.show()

#Plot Histogram of participants number of years of education

# Replace commas with decimal points and convert to numeric
df['years_of_education'] = pd.to_numeric(df['years_of_education'].str.replace(',', '.'), errors='coerce')
# Plot Histogram
plt.hist(df['years_of_education'] , bins=5, color='skyblue', edgecolor='black')
plt.xlabel('Years of eductation')
plt.ylabel('Count')
plt.title('Participants years of education')
plt.savefig(path + "education.png")
plt.show()

# Plot bar plot of countries occurence in patient's origin

# Concatenate values from specified columns into a new column
origins = ['origin_grandfather_m', 'origin_grandmother_m', 'origin_grandmother_f', 'origin_grandfather_f']
origin_df = df[origins]

# Count occurrences in the entire DataFrame
category_counts = origin_df.stack().value_counts()

# Display the counts
counts =category_counts.values
countries = category_counts.index.tolist()
# Plotting

plt.figure(figsize=(12, 8))
plt.barh(countries, counts, color='skyblue')
plt.xlabel('Counts')
plt.ylabel('Countries')
plt.title('Number of Occurrences of patients origins by Country')
#plt.yticks(countries[::2]) Optional makes it clear but not all countries are displayed
plt.savefig(path + "origins.png")
plt.show()