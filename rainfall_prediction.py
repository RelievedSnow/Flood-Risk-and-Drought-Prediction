import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load

# Load the provided CSV file to inspect its contents
file_path = 'All-India-Rainfall-Act_Dep_1901_to_2019_0.csv'
rainfall_data = pd.read_csv(file_path)

# Display basic information  
# print(rainfall_data.info()) 

# print first 5 rows of the dataset
# print(rainfall_data.head())

# Feature Engeneering

#FEATURE ENGINEERING 

# Feature Engineering: Defining thresholds for flood and drought
# We'll set arbitrary thresholds for now (can be fine-tuned):
# - Drought: JUN-SEP rainfall < 600 mm
# - Flood: JUN-SEP rainfall > 850 mm
# - Normal: Between 600 mm and 850 mm

def label_condition(row):
    if row['JUN-SEP'] < 600:
        return 'Drought'
    elif row['JUN-SEP'] > 850:
        return 'Flood'
    else:
        return 'Normal'

# Apply the function to create a new 'Condition' column
rainfall_data['Condition'] = rainfall_data.apply(label_condition, axis=1)

# Check the first few rows to see the new label
# print(rainfall_data.head())


# Prepare the data for modeling
# Features: Monthly rainfall (JUN, JUL, AUG, SEP)
X = rainfall_data[['JUN', 'JUL', 'AUG', 'SEP']]
# Target: Condition (Flood, Drought, Normal)
y = rainfall_data['Condition']


# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# print(accuracy)

# Create a color palette for different conditions
palette = {'Flood': 'red', 'Drought': 'blue', 'Normal': 'green'}

# Plot the data
plt.figure(figsize=(16, 6))
sns.barplot(x='YEAR', y='JUN-SEP', hue='Condition', data=rainfall_data, palette=palette)

# Add titles and labels
plt.title('Flood, Drought, and Normal Conditions Over the Years', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Total Rainfall (June - September)', fontsize=12)
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.legend(title='Condition')

# Show the plot
plt.tight_layout()
# plt.show()

# Save the trained model to a file
model_filename = 'rainfall_prediction_model.joblib'
# dump(clf, model_filename)
clf_in = load(model_filename)

# Example input for testing (you can modify this with real values)
new_sample = {
    'JUN': 120,
    'JUL': 250,
    'AUG': 260,
    'SEP': 130
}
# 1.5	2	2	7.1 // 120.0, 250.0, 260.0, 130.0

# Convert to DataFrame
input_df = pd.DataFrame([new_sample])

prediction = clf.predict(input_df)

print(f"The predicted condition is: {prediction[0]}")
