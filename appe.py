import pandas as pd
import numpy as np
import streamlit as st
import joblib

model = joblib.load('GB_model.pkl')

st.title('Hotel Booking Prediction')

st.divider()

st.write('Enter booking details to predict cancellation rate')

lead_time = st.slider('Lead Time', 0, 500, 50)
avg_price = st.number_input('Avg Price par room', 0.0, 500.0, 100.0)
special_requests = st.slider('Number of Special Guests', 0, 10, 1)
total_guests = st.slider('Total Guests', 1, 5, 2)
total_nights = st.slider('Total Nights', 1, 20, 3)
repeated_guest = st.selectbox('Repeated Guest', [0, 1])

if st.button('Predict'):
    # ترتيب الميزات يطابق تماماً ترتيب تدريب النموذج
    input_data = np.array([[
        lead_time,
        avg_price,
        repeated_guest,
        special_requests,
        total_guests,
        total_nights
    ]])

    # تم تصحيح Predict إلى predict (بحرف صغير)
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f'Prediction not Cancelled with a probability of {prob:.2f}')
    else:
        st.error(f'Prediction cancelled probability: {1-prob:.2f}')