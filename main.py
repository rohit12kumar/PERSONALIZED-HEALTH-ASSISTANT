import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Function to load custom CSS styles
def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load custom styles
load_css()
data = {
    'age': [25, 30, 22, 28, 35, 40],
    'weight': [60, 70, 65, 68, 80, 90],
    'hours_worked': [5, 8, 4, 6, 10, 12],
    'hydration': [1.5, 2.0, 1.2, 1.8, 2.5, 2.8]  # Water intake in liters
}

# Create a DataFrame
df = pd.DataFrame(data)
X = df[['age', 'weight', 'hours_worked']]
y = df['hydration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Test the model by predicting water intake for the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Output predictions and actual values
print("Predictions: ", y_pred)
print("Actual Values: ", y_test.values)


st.title("Personalized Health Assistant")
age = st.number_input("Enter your age", min_value=0, max_value=100)
weight = st.number_input("Enter your weight (kg)", min_value=0.0)
hours_worked = st.number_input("Hours worked today", min_value=0, max_value=24)
mood = st.selectbox("How do you feel today?", ["Happy", "Neutral", "streTired", "Stressed"])
st.write("Your inputs:")
st.write(f"Age: {age}")
st.write(f"Weight: {weight} kg")
st.write(f"Hours worked: {hours_worked}")
st.write(f"Mood: {mood}")
if st.button("Get Recommendations"):
    # Predict recommended water intake
    recommended_water = model.predict([[age, weight, hours_worked]])[0]
    st.write("You are supposed to,")

if hours_worked > 4:
    st.write("take a break for a while.")
    
if weight > 60 and hours_worked > 6:
    st.write(" Drink 2 liters of Water .")

if age > 40 and hours_worked > 5:
    st.write("Go for a walk for 15minutes to stay active.")

if hours_worked > 6:
    st.write("Take a break!")
    if mood == "Tired" or mood == "Stressed":
        st.write("Do Some relaxation exercises to de-stress.")
    
st.subheader("Hydration vs. Hours Worked")
plt.figure(figsize=(10, 5))
sns.scatterplot(x='hours_worked', y='hydration', data=df)
plt.xlabel('Hours Worked')
plt.ylabel('Hydration (Liters)')
plt.title('Hydration Needs Based on Hours Worked')
st.pyplot(plt)

    # New Bar Chart: Average working hours based on age
st.subheader("Average Working Hours by Age")
plt.figure(figsize=(10, 5))
    
    # Aggregate data: Calculate average hours worked for each age group
age_hours_avg = df.groupby('age')['hours_worked'].mean().reset_index()
    
    # Create a barplot
sns.barplot(x='age', y='hours_worked', data=age_hours_avg)
plt.xlabel('Age')
plt.ylabel('Average Hours Worked')
plt.title('Average Working Hours Based on Age')
st.pyplot(plt)

    
    
    