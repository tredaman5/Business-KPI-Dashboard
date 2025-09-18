# Business KPI Dashboard + Analytics

An interactive **Streamlit dashboard** for tracking business KPIs, analyzing customer segments, and predicting churn.  
The system uses **synthetic sample data**, runs through an ETL pipeline, trains machine learning models, and outputs results to a local SQLite database.

---

## 🚀 Features

- **KPIs Dashboard**  
  - Total Revenue, AOV, CAC, Marketing ROI, Churn Rate  
  - Daily revenue trend charts  

- **Customer Segmentation (KMeans)**  
  - Groups customers by behavior & revenue  
  - Explore top customers in each segment  
  - Shows both **Customer ID** and **Customer Name**  

- **Churn Prediction (Logistic Regression)**  
  - Identify high-risk customers  
  - Lookup individual customer profiles  

- **Data Pipeline**  
  - `generate_sample_data.py` → generates sample customers, orders, and revenue  
  - `etl.py` → loads data into a SQLite database (`db/kpi_data.db`)  
  - `train_models.py` → trains models & writes results to `customer_scored` table  

---

## 📂 Project Structure

Business-KPI-Dashboard/
│
├── db/ # SQLite database lives here
│ └── kpi_data.db
│
├── models/ # Trained ML models stored here
│
├── generate_sample_data.py # Generate synthetic sample data
├── etl.py # ETL pipeline into SQLite DB
├── train_models.py # Train segmentation + churn models
├── dashboard_app.py # Streamlit dashboard
│
└── requirements.txt # Python dependencies

yaml
Copy code

---

## ⚙️ Installation

1. **Clone repository**
   ```bash
   git clone https://github.com/yourusername/Business-KPI-Dashboard.git
   cd Business-KPI-Dashboard
Create a virtual environment

bash
Copy code
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
Install dependencies

bash
Copy code
pip install --upgrade pip
pip install -r requirements.txt
📊 Workflow
Generate sample data

bash
Copy code
python generate_sample_data.py
Creates CSV files with customers, orders, and revenue data.

Run ETL pipeline

bash
Copy code
python etl.py
Loads the CSVs into db/kpi_data.db.

Train models

bash
Copy code
python train_models.py
Trains KMeans for segmentation

Trains Logistic Regression for churn prediction

Saves models to models/

Writes scored customers into the database

Run dashboard

bash
Copy code
streamlit run dashboard_app.py
Open the provided URL (usually http://localhost:8501) in your browser.

🖥️ Dashboard Usage
KPIs Page

Filter by date range and region (sidebar)

View revenue trends and KPIs

Segments Page

See customer segments and average revenue

Explore top customers in a selected segment

Now shows both Customer ID and Customer Name

Churn Page

Lists high-risk customers with churn probability

Lookup individual customer profiles by Customer ID

🛠️ Tech Stack
Python (pandas, numpy, scikit-learn, sqlalchemy, sqlite3)

Streamlit for the dashboard

Plotly for interactive charts

SQLite for local database storage

📌 Notes
All data is synthetic (generated randomly by generate_sample_data.py).

To refresh customers after retraining, simply re-run:

bash
Copy code
python train_models.py
The dashboard automatically reflects new data and models.

