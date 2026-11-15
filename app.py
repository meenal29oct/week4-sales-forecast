
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet

st.set_page_config(
    page_title="Week 4 Sales Forecasting",
    layout="wide"
)

st.title("Week 4 | Sales Forecasting Dashboard")
st.caption(
    "Prophet forecasting with US holidays | "
    "Based on my Week 4 time-series homework"
)

# Load the professor's synthetic daily-sales dataset.
@st.cache_data
def load_data():
    df = pd.read_csv("daily_sales.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = pd.to_numeric(df["sales"])
    return df.sort_values("date").dropna()


data = load_data()

st.subheader("1. Historical sales")
st.line_chart(data.set_index("date")["sales"])

st.subheader("2. Forecast settings")

model_choice = st.selectbox(
    "Select forecasting model",
    [
        "My tuned Prophet (0.10)",
        "Professor's baseline Prophet (0.05)"
    ]
)

days = st.slider(
    "Number of days to forecast",
    min_value=7,
    max_value=90,
    value=30
)

prior_scale = (
    0.10 if model_choice == "My tuned Prophet (0.10)"
    else 0.05
)

st.info(
    f"Selected changepoint_prior_scale = {prior_scale}. "
    "Both models include US holidays."
)

st.subheader("3. Historical model evaluation")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Baseline Prophet test MAE",
        "11.240"
    )

with col2:
    st.metric(
        "Tuned Prophet test MAE",
        "11.228"
    )

st.caption(
    "These MAEs come from the historical 60-day test "
    "in the homework notebook. They do not measure "
    "accuracy on future dates."
)

# Forecast with the selected model.
@st.cache_data(show_spinner=False)
def generate_forecast(sales_data, scale, horizon):
    prophet_data = sales_data.rename(
        columns={"date": "ds", "sales": "y"}
    )[["ds", "y"]]

    model = Prophet(changepoint_prior_scale=scale)
    model.add_country_holidays(country_name="US")
    model.fit(prophet_data)

    future = model.make_future_dataframe(
        periods=horizon,
        freq="D"
    )

    prediction = model.predict(future)

    return prediction.tail(horizon).copy()


st.subheader("4. Generate forecast")

if st.button("Make forecast", type="primary"):

    with st.spinner("Training Prophet and forecasting..."):
        forecast = generate_forecast(
            data,
            prior_scale,
            days
        )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data["date"],
        y=data["sales"],
        mode="lines",
        name="Historical sales",
        line=dict(color="#475569")
    ))

    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_lower"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(14,165,233,0.18)",
        line=dict(width=0),
        name="Uncertainty interval"
    ))

    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat"],
        mode="lines",
        name="Predicted sales",
        line=dict(color="#0284c7", width=3)
    ))

    fig.update_layout(
        title=f"{days}-Day Sales Forecast",
        xaxis_title="Date",
        yaxis_title="Sales units",
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    result = forecast[
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].copy()

    result.columns = [
        "Date",
        "Predicted sales",
        "Lower bound",
        "Upper bound"
    ]

    result["Date"] = result["Date"].dt.date

    for col in [
        "Predicted sales",
        "Lower bound",
        "Upper bound"
    ]:
        result[col] = result[col].round(0).astype(int)

    st.subheader("5. Forecast results")
    st.dataframe(result, hide_index=True)

    st.download_button(
        "Download forecast CSV",
        data=result.to_csv(index=False),
        file_name="week4_sales_forecast.csv",
        mime="text/csv"
    )

    st.subheader("6. Business interpretation")

    st.write(
        "When stockout costs exceed leftover inventory costs, "
        "ordering exactly the point forecast may not be optimal. "
        "A higher demand quantile can be more appropriate, "
        "depending on shortage costs, holding costs, lead times, "
        "and inventory availability."
    )

    st.caption(
        "Forecast intervals are estimates, not guarantees. "
        "This app uses synthetic historical sales data."
    )
