# etl.py
import os
import pandas as pd
from sqlalchemy import create_engine

os.makedirs("db", exist_ok=True)
DATA_DIR = "data"
DB_PATH = "db/kpi_data.db"

def load_csv(name, parse_dates=None):
    return pd.read_csv(os.path.join(DATA_DIR, name), parse_dates=parse_dates)

def preprocess():
    # Load CSVs
    customers = load_csv("customers.csv", parse_dates=["signup_date"])
    orders = load_csv("orders.csv", parse_dates=["order_date"])
    marketing = load_csv("marketing_spend.csv", parse_dates=["date"])

    # Basic cleaning
    customers['signup_date'] = pd.to_datetime(customers['signup_date'])
    customers['churn_flag'] = customers['churn_flag'].astype(bool)

    orders['order_date'] = pd.to_datetime(orders['order_date'])
    orders = orders[orders['revenue'] >= 0]

    marketing['date'] = pd.to_datetime(marketing['date'])

    # Derived fields
    orders['order_month'] = orders['order_date'].dt.to_period('M').dt.to_timestamp()
    customers['signup_month'] = customers['signup_date'].dt.to_period('M').dt.to_timestamp()
    marketing['spend_month'] = marketing['date'].dt.to_period('M').dt.to_timestamp()

    # Aggregates
    daily_revenue = orders.groupby(orders['order_date'].dt.date).agg(daily_revenue=('revenue','sum'),
                                                                    orders=('order_id','count')).reset_index().rename(columns={'order_date':'date'})
    monthly_revenue = orders.groupby('order_month').agg(month_revenue=('revenue','sum'),
                                                       orders=('order_id','count')).reset_index().rename(columns={'order_month':'month'})

    # Customer-level aggregates (features)
    cust_agg = orders.groupby('customer_id').agg(
        total_revenue=('revenue','sum'),
        orders_count=('order_id','count'),
        first_order=('order_date','min'),
        last_order=('order_date','max'),
        avg_order_value=('revenue','mean')
    ).reset_index()

    cust_agg['recency_days'] = (pd.to_datetime(orders['order_date'].max()) - pd.to_datetime(cust_agg['last_order'])).dt.days
    cust_agg['tenure_days'] = (pd.to_datetime(cust_agg['last_order']) - pd.to_datetime(cust_agg['first_order'])).dt.days.fillna(0)
    cust_agg['avg_order_value'] = cust_agg['avg_order_value'].fillna(0)

    # Merge churn flag and region
    cust_features = pd.merge(customers[['customer_id','signup_date','churn_flag','region']], cust_agg, on='customer_id', how='left')
    cust_features[['total_revenue','orders_count','avg_order_value','recency_days','tenure_days']] = cust_features[['total_revenue','orders_count','avg_order_value','recency_days','tenure_days']].fillna(0)

    # Marketing monthly spend by channel and total
    marketing_monthly = marketing.groupby(['spend_month','channel']).agg(month_spend=('spend','sum')).reset_index().rename(columns={'spend_month':'month'})
    marketing_total_monthly = marketing_monthly.groupby('month').agg(total_spend=('month_spend','sum')).reset_index()

    # Merge monthly revenue with marketing & new customers for KPI calculations
    new_customers = customers.groupby('signup_month').agg(new_customers=('customer_id','count')).reset_index().rename(columns={'signup_month':'month'})
    monthly = pd.merge(monthly_revenue, marketing_total_monthly, how='left', left_on='month', right_on='month')
    monthly = pd.merge(monthly, new_customers, how='left', left_on='month', right_on='month')
    monthly['total_spend'] = monthly['total_spend'].fillna(0)
    monthly['new_customers'] = monthly['new_customers'].fillna(0)

    monthly['aov'] = monthly['month_revenue'] / monthly['orders'].replace(0, pd.NA)
    monthly['cac'] = monthly['total_spend'] / monthly['new_customers'].replace(0, pd.NA)
    monthly['marketing_roi'] = monthly['month_revenue'] / monthly['total_spend'].replace(0, pd.NA)

    # Save to sqlite
    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    daily_revenue.to_sql("daily_revenue", engine, if_exists="replace", index=False)
    monthly.to_sql("monthly_kpis", engine, if_exists="replace", index=False)
    cust_features.to_sql("customer_features", engine, if_exists="replace", index=False)
    customers.to_sql("customers", engine, if_exists="replace", index=False)
    orders.to_sql("orders", engine, if_exists="replace", index=False)
    marketing_monthly.to_sql("marketing_monthly_channel", engine, if_exists="replace", index=False)
    marketing_total_monthly.to_sql("marketing_monthly_total", engine, if_exists="replace", index=False)

    print("ETL complete — tables written to", DB_PATH)

if __name__ == "__main__":
    preprocess()
