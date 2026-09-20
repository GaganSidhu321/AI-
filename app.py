# ============================================================
# CRISIS WHISPER NETWORK
# AI-Based Economic Early Warning System
# FINAL A to Z WORKING STREAMLIT CODE
# ============================================================

# --------------------
# 1. IMPORT LIBRARIES
# --------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fredapi import Fred
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --------------------
# 2. PAGE CONFIG
# --------------------
st.set_page_config(
    page_title="Crisis Whisper Network",
    layout="wide"
)

st.title("📉 Crisis Whisper Network")
st.subheader("AI-based Economic Early Warning System (Live FRED Data)")

# --------------------
# 3. FRED API SETUP
# --------------------
FRED_API_KEY = "f9f063e4e865d01c042ceb4f3ce0de64"
fred = Fred(api_key=FRED_API_KEY)

# --------------------
# 4. LOAD FRED DATA
# --------------------
@st.cache_data
def load_data():
    long_rate = fred.get_series("GS10")     # 10Y Treasury
    short_rate = fred.get_series("GS2")     # 2Y Treasury
    credit = fred.get_series("BAA10Y")      # Credit Spread
    capital = fred.get_series("TCMDO")      # Capital Flow (TIC proxy)

    df = pd.concat([long_rate, short_rate, credit, capital], axis=1)
    df.columns = ["Long_Rate", "Short_Rate", "Credit_Spread", "Capital_Flow"]
    df.index = pd.to_datetime(df.index)

    return df

df = load_data()

# --------------------
# 5. SMART NaN FILL FUNCTION (NO NaN GUARANTEE)
# --------------------
def fill_safe(series, default=0.0):
    series = series.interpolate(method="time")

    if series.isna().any():
        series = series.fillna(series.median())

    if series.isna().any():
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        series = series.fillna((q1 + q3) / 2)

    return series.fillna(default)

# --------------------
# 6. APPLY FILLING (ALL COLUMNS)
# --------------------
for col in ["Long_Rate", "Short_Rate", "Credit_Spread", "Capital_Flow"]:
    df[col] = fill_safe(df[col], default=0.0)

# --------------------
# 7. FEATURE ENGINEERING
# --------------------
df["Term_Spread"] = df["Long_Rate"] - df["Short_Rate"]
df["Term_Spread"] = fill_safe(df["Term_Spread"], default=0.0)

# Crisis logic (economic rules)
df["Crisis"] = (
    (df["Term_Spread"] < 0) |
    (df["Credit_Spread"] > 2) |
    (df["Capital_Flow"] < 0)
).astype(int)

# --------------------
# 8. MODEL TRAINING (SAFE)
# --------------------
X = df[["Term_Spread", "Credit_Spread", "Capital_Flow"]]
y = df["Crisis"]

# Final safety check
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0.0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

df["Crisis_Probability"] = model.predict_proba(X_scaled)[:, 1]

# --------------------
# 9. RISK LABEL FUNCTION
# --------------------
def risk_label(p):
    if p >= 0.6:
        return "HIGH RISK 🔴"
    elif p >= 0.4:
        return "LOW RISK 🟠"
    else:
        return "SAFE 🟢"

df["Risk_Level"] = df["Crisis_Probability"].apply(risk_label)

# --------------------
# 10. YEAR INPUT
# --------------------
year = st.sidebar.number_input(
    "Select Year",
    min_value=int(df.index.year.min()),
    max_value=int(df.index.year.max()),
    value=2025,
    step=1
)

year_data = df[df.index.year == year]

if year_data.empty:
    st.error("No data available for selected year")
    st.stop()

latest = year_data.iloc[-1]

# --------------------
# 11. METRICS DISPLAY
# --------------------
c1, c2, c3, c4 = st.columns(4)

c1.metric("Term Spread", round(latest["Term_Spread"], 2))
c2.metric("Credit Spread", round(latest["Credit_Spread"], 2))
c3.metric("Capital Flow", round(latest["Capital_Flow"], 2))
c4.metric(
    "Final Risk Level",
    round(latest["Crisis_Probability"], 2),
    latest["Risk_Level"]
)

# =========================================================
# DAY 9 – WEBSITE GRAPHS + TABLE
# =========================================================

# --------------------
# LINE GRAPH
# Crisis Probability vs Year
# --------------------
st.subheader("📈 Crisis Probability Over Time")

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    df.index,
    df["Crisis_Probability"],
    linewidth=2
)

# Threshold lines
ax.axhline(0.4, linestyle="--")
ax.axhline(0.6, linestyle="--")

ax.set_xlabel("Year")
ax.set_ylabel("Crisis Probability")

st.pyplot(fig)

# --------------------
# BAR GRAPH
# Capital Flow
# --------------------
st.subheader("💰 Capital Flow Over Time")

fig2, ax2 = plt.subplots(figsize=(10, 4))

ax2.bar(
    df.index.year,
    df["Capital_Flow"]
)

ax2.set_xlabel("Year")
ax2.set_ylabel("Capital Flow")

st.pyplot(fig2)

# --------------------
# TABLE
# Last 10 Predictions
# --------------------
st.subheader("📋 Last 10 Predictions")

last_10 = df[["Crisis_Probability", "Risk_Level"]].tail(10)

# Date ko column banana
last_10 = last_10.reset_index()
last_10.columns = ["Date", "Crisis Probability", "Risk Level"]
st.dataframe(last_10)


# --------------------
# 12. GRAPH – FULL HISTORY
# --------------------
st.markdown("## 📊 Crisis Probability Over Time")

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df.index, df["Crisis_Probability"])
ax1.axhline(0.6, linestyle="--")
ax1.axhline(0.4, linestyle="--")
ax1.set_ylabel("Crisis Probability")
st.pyplot(fig1)

# --------------------
# 13. GRAPH – SELECTED YEAR
# --------------------
st.markdown(f"## 📈 Crisis Probability in {year}")

fig2, ax2 = plt.subplots(figsize=(10, 3))
ax2.plot(
    year_data.index,
    year_data["Crisis_Probability"],
    marker="o"
)
ax2.axhline(0.6, linestyle="--")
ax2.axhline(0.4, linestyle="--")
ax2.set_ylabel("Crisis Probability")
st.pyplot(fig2)

# --------------------


# --------------------
# 14. FINAL MESSAGE
# --------------------
st.success(
    f"In {year}, the economic condition is classified as: {latest['Risk_Level']}"
)





import streamlit as st
import pandas as pd
import numpy as np

from fredapi import Fred
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import mysql.connector
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="babamaheshdasji8872315127",
    database="crisis_db"
)

cursor = conn.cursor()

for index, row in df.iterrows():
    cursor.execute(
        """
        INSERT INTO crisis_data
        (date, term_spread, credit_spread, capital_flow, crisis_probability, risk_level)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            index.date(),
            row["Term_Spread"],
            row["Credit_Spread"],
            row["Capital_Flow"],
            row["Crisis_Probability"],
            row["Risk_Level"]
        )
    )

conn.commit()


st.subheader("Latest Prediction")

latest = df.iloc[-1]

st.metric("Term Spread", round(latest["Term_Spread"], 2))
st.metric("Credit Spread", round(latest["Credit_Spread"], 2))
st.metric("Capital Flow", round(latest["Capital_Flow"], 2))
st.metric("Risk Level", latest["Risk_Level"])


# ==============================
# READ DATA FROM MYSQL
# ==============================
cursor.execute("""
    SELECT 
        date,
        term_spread,
        credit_spread,
        capital_flow,
        risk_level
    FROM crisis_data
    ORDER BY date DESC
    LIMIT 10
""")


sql_data = cursor.fetchall()
import pandas as pd

df_sql = pd.DataFrame(
    sql_data,
    columns=[
        "Date",
        "Term Spread",
        "Credit Spread",
        "Capital Flow",
        "Risk Level"
    ]
)

st.markdown("---")
st.subheader("📊 Recent Records Stored in SQL Database")

st.dataframe(df_sql, use_container_width=True)
