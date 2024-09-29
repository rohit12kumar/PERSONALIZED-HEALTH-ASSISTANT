import streamlit as st

st.title("Personalized Health Assistant")
age = st.number_input("Enter your age", min_value=0, max_value=100)
weight = st.number_input("Enter your weight (kg)", min_value=0.0)
hours_worked = st.number_input("Hours worked today", min_value=0, max_value=24)
mood = st.selectbox("How do you feel today?", ["Happy", "Neutral", "Tired", "Stressed"])
st.write("Your inputs are:", age, weight, hours_worked, mood)
