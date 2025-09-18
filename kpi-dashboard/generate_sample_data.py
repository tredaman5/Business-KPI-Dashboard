# generate_sample_data.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

os.makedirs("data", exist_ok=True)
np.random.seed(42)

### CONFIG: increase these to scale dataset
n_customers = 50000        # 50k customers (tweak as needed)
n_orders = 700000          # 700k orders (tweak as needed)
start_date = datetime(2023, 1, 1)
end_date = datetime(2025, 9, 1)

print(f"Generating {n_customers} customers and {n_orders} orders...")

# Customers
customer_ids = [f"C{100000+i}" for i in range(n_customers)]
# Signup roughly over first 18 months
signup_dates = [start_date + timedelta(days=int(np.random.exponential(scale=300))) for _ in range(n_customers)]
# Simulate churn probability based on signup age and randomness
signup_ages_days = [(end_date - sd).days for sd in signup_dates]
churn_probs = np.clip(np.array(signup_ages_days) / 2000 + np.random.beta(1.2,6,size=n_customers)*0.2, 0, 1)
churn_flag = np.random.rand(n_customers) < churn_probs
regions = np.random.choice(["North", "South", "East", "West"], size=n_customers, p=[0.3,0.25,0.25,0.2])

customers = pd.DataFrame({
    "customer_id": customer_ids,
    "signup_date": pd.to_datetime(signup_dates).date,
    "churn_flag": churn_flag,
    "region": regions
})
customers.to_csv("data/customers.csv", index=False)

# Orders
date_range_days = (end_date - start_date).days
order_dates = [start_date + timedelta(days=int(np.random.exponential(scale=date_range_days/2))) for _ in range(n_orders)]
order_customer = np.random.choice(customer_ids, size=n_orders)
# revenue distribution: many small, few large
order_amount = np.round(np.random.gamma(2.2, 40.0, size=n_orders) + np.random.choice([0,10,20,50,100], size=n_orders, p=[0.6,0.15,0.12,0.08,0.05]), 2)
product_cat = np.random.choice(["A","B","C","D"], size=n_orders, p=[0.45,0.25,0.2,0.1])

orders = pd.DataFrame({
    "order_id": [f"O{200000+i}" for i in range(n_orders)],
    "customer_id": order_customer,
    "order_date": pd.to_datetime(order_dates).date,
    "revenue": order_amount,
    "product_category": product_cat
})
orders.to_csv("data/orders.csv", index=False)

# Marketing spend (daily per channel)
channels = ["Paid Search", "Social", "Email", "Affiliate"]
dates = pd.date_range(start_date, end_date)
rows = []
for d in dates:
    base = np.random.uniform(500,5000)
    split = np.random.dirichlet([2,1.5,1.2,0.5])
    for ch, s in zip(channels, split):
        rows.append({"date": d.date(), "channel": ch, "spend": float(round(base * s + np.random.normal(0,50),2))})

marketing = pd.DataFrame(rows)
marketing.to_csv("data/marketing_spend.csv", index=False)

print("Sample data generated in ./data/: customers.csv, orders.csv, marketing_spend.csv")
