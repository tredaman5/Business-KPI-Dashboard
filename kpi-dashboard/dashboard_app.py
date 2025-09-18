# dashboard_app.py
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from datetime import datetime
import joblib
from sqlalchemy import create_engine

DB_PATH = "db/kpi_data.db"
MODELS_DIR = "models"

@st.cache_data
def load_table(name):
    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    df = pd.read_sql_table(name, engine)
    return df

st.set_page_config(layout="wide", page_title="Business KPI Dashboard + Analytics")

st.title("Business KPI Dashboard — Enhanced (Segmentation & Churn)")

# Load data
daily = load_table("daily_revenue") if "daily_revenue" in pd.io.sql.read_sql_table.__qualname__ else pd.read_sql_query("SELECT * FROM daily_revenue", sqlite3.connect(DB_PATH))
monthly = load_table("monthly_kpis")
orders = load_table("orders")
customers = load_table("customers")
scored = load_table("customer_scored") if "customer_scored" in pd.io.sql.read_sql_table.__qualname__ or True else None

# Fallback safe loads (if read_sql_table path above mis-guards)
conn = sqlite3.connect(DB_PATH)
try:
    daily = pd.read_sql_query("SELECT * FROM daily_revenue", conn, parse_dates=['date'])
except Exception:
    daily = pd.DataFrame()
monthly = pd.read_sql_query("SELECT * FROM monthly_kpis", conn, parse_dates=['month'])
orders = pd.read_sql_query("SELECT * FROM orders", conn, parse_dates=['order_date'])
customers = pd.read_sql_query("SELECT * FROM customers", conn, parse_dates=['signup_date'])
try:
    scored = pd.read_sql_query("SELECT * FROM customer_scored", conn, parse_dates=['signup_date'])
except Exception:
    scored = pd.DataFrame()
conn.close()

# Parse dates
if not daily.empty:
    daily['date'] = pd.to_datetime(daily['date'])
monthly['month'] = pd.to_datetime(monthly['month'])
customers['signup_date'] = pd.to_datetime(customers['signup_date'])
orders['order_date'] = pd.to_datetime(orders['order_date'])
if not scored.empty and 'churn_prob' in scored.columns:
    scored['signup_date'] = pd.to_datetime(scored['signup_date'])

# Sidebar controls
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["KPIs", "Segments", "Churn"])

st.sidebar.header("Filters")
if not daily.empty:
    min_date = daily['date'].min()
    max_date = daily['date'].max()
else:
    min_date = orders['order_date'].min()
    max_date = orders['order_date'].max()
date_range = st.sidebar.date_input("Date range", [min_date, max_date])

region_options = list(customers['region'].unique())
region_sel = st.sidebar.multiselect("Region", region_options, default=region_options)

# apply region/date filters to orders & scored
cust_ids_region = customers[customers['region'].isin(region_sel)]['customer_id'].unique()
orders_f = orders[(orders['order_date'] >= pd.to_datetime(date_range[0])) & (orders['order_date'] <= pd.to_datetime(date_range[1])) & (orders['customer_id'].isin(cust_ids_region))]

# KPIs page
if page == "KPIs":
    st.header("KPIs")
    total_revenue = orders_f['revenue'].sum()
    total_orders = orders_f['order_id'].nunique()
    aov = (total_revenue / total_orders) if total_orders > 0 else 0
    total_spend = monthly[(monthly['month'] >= pd.to_datetime(date_range[0])) & (monthly['month'] <= pd.to_datetime(date_range[1]))]['total_spend'].sum() if 'total_spend' in monthly.columns else 0
    new_customers = customers[(customers['signup_date'] >= pd.to_datetime(date_range[0])) & (customers['signup_date'] <= pd.to_datetime(date_range[1])) & (customers['region'].isin(region_sel))]['customer_id'].nunique()
    cac = (total_spend / new_customers) if new_customers > 0 else None
    marketing_roi = (total_revenue / total_spend) if total_spend > 0 else None
    churn_rate = customers[customers['churn_flag'] & customers['region'].isin(region_sel)].shape[0] / customers[customers['region'].isin(region_sel)].shape[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"${total_revenue:,.0f}")
    c2.metric("AOV", f"${aov:,.2f}")
    c3.metric("CAC", f"${cac:,.2f}" if cac else "N/A")
    c4.metric("Marketing ROI", f"{marketing_roi:.2f}x" if marketing_roi else "N/A")

    st.markdown("---")
    st.subheader("Revenue Trend")
    if not daily.empty:
        daily_f = daily[(daily['date'] >= pd.to_datetime(date_range[0])) & (daily['date'] <= pd.to_datetime(date_range[1]))]
        fig = px.line(daily_f, x='date', y='daily_revenue', title='Daily Revenue')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No daily revenue data available.")

# Segments page
elif page == "Segments":
    st.header("Customer Segmentation (KMeans)")
    if scored.empty:
        st.write("No scored customer table found. Run `train_models.py` first.")
    else:
        seg_counts = scored.groupby('segment').agg(customers=('customer_id','count'), avg_revenue=('total_revenue','mean')).reset_index()
        st.dataframe(seg_counts.sort_values('customers', ascending=False))
        fig = px.bar(seg_counts, x='segment', y='customers', hover_data=['avg_revenue'], title='Customers per Segment')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Segment explorer")
        seg_sel = st.selectbox("Select segment", sorted(scored['segment'].unique()))
        seg_df = scored[scored['segment'] == seg_sel]
        st.markdown(f"Showing **{len(seg_df)}** customers in segment {seg_sel}")
        st.dataframe(seg_df[['customer_id','total_revenue','orders_count','avg_order_value','churn_prob']].sort_values('total_revenue', ascending=False).head(200))

# Churn page
elif page == "Churn":
    st.header("Churn prediction & risk")
    if scored.empty:
        st.write("No scored customer table found. Run `train_models.py` first.")
    else:
        st.subheader("High risk customers")
        high_risk = scored[(scored['churn_prob']>=0.6) & (scored['region'].isin(region_sel))]
        st.dataframe(high_risk[['customer_id','region','total_revenue','orders_count','churn_prob']].sort_values('churn_prob', ascending=False).head(300))

        st.subheader("Lookup single customer")
        cust_input = st.text_input("Enter customer_id (e.g., C100123):")
        if cust_input:
            c = scored[scored['customer_id']==cust_input]
            if c.empty:
                st.write("Customer not found.")
            else:
                st.write(c.T)
                fig = px.bar(c.melt(id_vars=['customer_id'], value_vars=['total_revenue','orders_count','avg_order_value','recency_days','tenure_days']), x='variable', y='value', title=f"Customer {cust_input} profile")
                st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.write("Notes: To refresh scored customers after re-training, run `python train_models.py` which writes `customer_scored` to the local SQLite DB.")