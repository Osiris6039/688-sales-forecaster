
import streamlit as st
import pandas as pd
from prophet import Prophet
import base64

# === LOGIN SECTION ===
def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state["authenticated"] = True
        else:
            st.error("Invalid username or password")

# === FORECASTING SECTION ===
def forecast_app():
    st.title("📈 Sales & Customer Forecasting AI App")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data")
        st.dataframe(data.head())

        forecast_data = {}
        for target in ['sales', 'customers']:
            df = data[['date', target]].rename(columns={"date": "ds", target: "y"})
            df['ds'] = pd.to_datetime(df['ds'])
            model = Prophet()
            model.fit(df)
            future = model.make_future_dataframe(periods=7)
            forecast = model.predict(future)
            forecast_data[target] = forecast[['ds', 'yhat']].rename(columns={'yhat': f'{target}_forecast'})
            st.line_chart(forecast.set_index('ds')['yhat'])

        result = pd.merge(forecast_data['sales'], forecast_data['customers'], on='ds')
        st.subheader("📥 Download Forecast")
        csv = result.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="forecast.csv">📥 Download CSV</a>', unsafe_allow_html=True)

# === ROUTING ===
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if st.session_state["authenticated"]:
    forecast_app()
else:
    login()
