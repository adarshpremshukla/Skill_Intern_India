import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Load the dataset

data = pd.read_csv(r'C:\Users\adars\OneDrive\Desktop\code\internship\Skill Intern\student_scores.csv')

# Step 2: Prepare the data
X = data['Hours'].values.reshape(-1, 1)
y = data['Scores'].values

# Step 3: Data Visualization
plt.scatter(X, y)
plt.title('Study Hours vs Percentage Scores')
plt.xlabel('Study Hours')
plt.ylabel('Percentage Scores')
plt.show()

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
hours = 9.25
predicted_score = model.predict([[hours]])

print(f'Predicted score for studying {hours} hours/day: {predicted_score[0]}')
