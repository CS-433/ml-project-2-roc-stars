import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Directory where the images will be saved
path = "plots/ethics/"

# Set figures dimensions
im_size = (8,6)

# Set figures fontisze
font_size = 15

# Load dataset
df = pd.read_csv('data/Data_Patients.csv', sep=";", header=0)

# Plot histogram of participants Age
plt.figure(figsize=im_size) 
plt.hist(df['age'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Age', fontsize = font_size)
plt.ylabel('Count', fontsize = font_size)
#plt.title('Age distribution across participants')
plt.savefig(path + "age.png")
plt.show()

# Plot Pie Chart of participants Gender
# Map 0, 1 to female and male 
df['sex'] = df['sex'].map({0: 'Female', 1: 'Male'})
#ATTENTION IL FAUT DEMANDER SI O = HOMME OUR FEMME
plt.figure(figsize=(8, 8))
age_counts = df['sex'].value_counts()
plt.pie(age_counts, labels=age_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors, textprops={'fontsize': 18})
plt.savefig(path + "gender.png")
plt.show()

#Plot Histogram of participants number of years of education

# Replace commas with decimal points and convert to numeric
df['years_of_education'] = pd.to_numeric(df['years_of_education'].str.replace(',', '.'), errors='coerce')
# Plot Histogram
plt.figure(figsize=(8, 8))
plt.hist(df['years_of_education'] , bins=5, color='skyblue', edgecolor='black')
plt.xlabel('Years of eductation', fontsize = font_size)
plt.ylabel('Count', fontsize = font_size)
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

plt.figure(figsize=(8, 8))
plt.barh(countries, counts, color='skyblue')
plt.xlabel('Counts', fontsize = font_size)
plt.ylabel('Countries', fontsize = font_size)
#plt.yticks(countries[::2]) Optional makes it clear but not all countries are displayed
plt.savefig(path + "origins.png")
plt.show()