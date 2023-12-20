# < ----------------------------------Import libraries----------------------------------- >
import pandas as pd
import matplotlib.pyplot as plt

# < -----------------------------Define path to save images------------------------------ >
path = "plots/ethics/"

# < -------------------------------------Load Dataset------------------------------------ >
df = pd.read_csv('Datasets/Data_Patients.csv', sep=";", header=0)

# < ----------------------------------------Age------------------------------------------ >
# Define age dataframe
ages = df['age']

# Define bin edges and labels
bins = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60] 
bin_labels = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59']

# Plot histogram
plt.figure(figsize=(10, 15)) 
plt.hist(ages, bins=bins, color='skyblue', edgecolor='black')
plt.xlabel('Age', fontsize=30)
plt.ylabel('Count', fontsize=30)
plt.xticks([(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)], bin_labels, fontsize=18)
plt.yticks(fontsize=25)
plt.title('Age distribution across participants', fontsize=30)

# Save and show the plot
plt.savefig(path + "age.png")
plt.show()

# < ----------------------------------------Gender--------------------------------------- >
# Plot Pie Chart of participants Gender
# Map 0, 1 to female and male 
df['sex'] = df['sex'].map({0: 'Female', 1: 'Male'})
custom_colors = ['skyblue', 'cornflowerblue']
# Plot pie chart
plt.figure(figsize=(10, 10))
sex_counts = df['sex'].value_counts()
plt.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%', startangle=90, colors=custom_colors, textprops={'fontsize': 30})
plt.title('Gender Distribution', fontsize=30)  

# Save and show the plot
plt.savefig(path + "gender.png")
plt.show()

# < ----------------------------------Years of Education--------------------------------- >
#Plot Histogram of participants number of years of education
# Replace commas with decimal points and convert to numeric
df['years_of_education'] = pd.to_numeric(df['years_of_education'].str.replace(',', '.'), errors='coerce')
# Plot Histogram
years = df['years_of_education']
bins = [10, 15, 20, 25, 30] 
bin_labels = ['10-14', '15-19', '20-24', '25-29']

plt.figure(figsize=(10, 10))
plt.hist(years, bins=bins, color='skyblue', edgecolor='black')
plt.xlabel('Years of eductation', fontsize = 30)
plt.ylabel('Count', fontsize = 30)
plt.xticks([(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)], bin_labels, fontsize=18)
plt.yticks(fontsize=25)
plt.title('Education', fontsize=30)
plt.savefig(path + "education.png")
plt.show()

# < ----------------------------------------Origins-------------------------------------- >
# Plot bar plot of countries occurence in patient's origin
# Concatenate values from specified columns into a new column
origins = ['origin_grandfather_m', 'origin_grandmother_m', 'origin_grandmother_f', 'origin_grandfather_f']
origin_df = df[origins]

# Count occurrences in the entire DataFrame
category_counts = origin_df.stack().value_counts()

# Display the counts
counts =category_counts.values
countries = category_counts.index.tolist()

# Plot
plt.figure(figsize=(10, 15))
plt.barh(countries, counts, color='skyblue')
plt.xlabel('Counts', fontsize = 30)
plt.ylabel('Countries', fontsize = 30)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('Origins', fontsize=30)
plt.savefig(path + "origins.png")
plt.show()