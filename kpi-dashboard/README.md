# 📊 Business KPI Dashboard with Segmentation & Churn Prediction

An interactive **Business KPI Dashboard** built with **Streamlit**, **SQLite**, and **scikit-learn**.  
It tracks revenue, customer KPIs, customer segmentation (KMeans), and churn prediction (RandomForest).  

---

## 🚀 How to Operate the Program

Follow these steps in order:

### 1. Generate Sample Data 
Creates fake customers, orders, and revenue history.

```bash
python generate_sample_data.py

### 2. Run the ETL Pipeline 

- Loads the generated data into the SQLite database (db/kpi_data.db).

python etl.py

### 3. Train the ML Models

Customer segmentation (KMeans)

Churn prediction (RandomForest)

This saves trained models and writes scored customers into the database.

python train_models.py

### 4. Launch the Dashboard

Runs the Streamlit app to explore KPIs, segments, and churn risk.

streamlit run dashboard_app.py