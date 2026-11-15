
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

st.title("My Week 4 Sales Forecasting App")

data = pd.read_csv("daily_sales.csv")
data["date"] = pd.to_datetime(data["date"])

st.write("Historical sales")
st.line_chart(data.set_index("date")["sales"])

days = st.slider("Forecast days", 7, 90, 30)

if st.button("Make forecast"):
    df = data.rename(
        columns={"date": "ds", "sales": "y"}
    )

    model = Prophet(changepoint_prior_scale=0.10)
    model.add_country_holidays(country_name="US")
    model.fit(df)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    result = forecast.tail(days)

    st.line_chart(
        result.set_index("ds")["yhat"]
    )

    st.dataframe(
        result[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    )

    st.download_button(
        "Download forecast",
        result.to_csv(index=False),
        "forecast.csv"
    )
