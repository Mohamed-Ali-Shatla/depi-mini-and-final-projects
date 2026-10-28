# 🛒 Brazilian E-Commerce Sales Analysis

An end-to-end Data Science project analyzing Brazilian e-commerce transactions, customer reviews, and sales performance to extract business insights, build an interactive dashboard, and train a review prediction model.

---

## 📁 Repository Structure

```text
mini-projects/
│
├── Dashboard/
│   └── 5_Dashboard(Habiba Sameh).ipynb                 # Interactive dashboard
│
├── data/
│   ├── raw/
│   │   ├── raw_data_link.md                            # Link to download the original Olist dataset
│   │   └── Schema.png                                  # ERD Diagram / Database Schema
│   ├── processed/
│   │   └── brazilian_ecommerce_data.csv                # Master dataset exported from the SQL query
│   ├── cleaned/
│   │   └── cleaned_ecommerce_data.csv                  # Dataset after data cleaning
│   └── preprocessed/
│       └── preprocessed.csv                            # Dataset after preprocessing, ready for modeling
│
├── Model/
│   └── 6_Review_Prediction_Model_(Habiba Sameh).ipynb  # Review prediction model
│
├── notebooks/
│   ├── 1_data_querying_and_export_(mohamed_shatla).ipynb   # SQL data extraction & master dataset export
│   ├── 2_cleaned_data_(salma mamdouh).ipynb                # Data cleaning
│   ├── 3_preprocessed(Kareem_Elawaadi).ipynb               # Data preprocessing
│   ├── 4_EDA(Omar_tamer).ipynb                             # Exploratory Data Analysis
│   ├── 5_Dashboard(Habiba Sameh).ipynb                     # Dashboard
│   └── 6_Review_Prediction_Model_(Habiba Sameh).ipynb      # Review prediction model
│
├── sql/
│   └── master_query.sql                                # SQL query used to build the master dataset
│
├── requirements.txt                                    # Project dependencies
└── README.md                                           # Project documentation
```

---

## 🔄 Data Pipeline

```text
raw (Olist DB) ──► processed ──► cleaned ──► preprocessed ──► EDA / Dashboard / Model
   SQL query        Notebook 1    Notebook 2    Notebook 3      Notebooks 4, 5, 6
```

| Stage | Folder | File | Produced by |
|-------|--------|------|-------------|
| Raw source | `data/raw/` | Link in `raw_data_link.md` + `Schema.png` | — |
| Master dataset | `data/processed/` | `brazilian_ecommerce_data.csv` | `sql/master_query.sql` + Notebook 1 |
| Cleaned data | `data/cleaned/` | `cleaned_ecommerce_data.csv` | Notebook 2 |
| Preprocessed data | `data/preprocessed/` | `preprocessed.csv` | Notebook 3 |

> The dashboard (Notebook 5) reads the master dataset from `data/processed/brazilian_ecommerce_data.csv` and applies its own light cleaning (see *Interactive Dashboard*).

---

## 📓 Notebooks

| # | Notebook | Description | Author |
|---|----------|-------------|--------|
| 1 | `1_data_querying_and_export_(mohamed_shatla).ipynb` | SQL data extraction & master dataset export | Mohamed Shatla |
| 2 | `2_cleaned_data_(salma mamdouh).ipynb` | Data cleaning | Salma Mamdouh |
| 3 | `3_preprocessed(Kareem_Elawaadi).ipynb` | Data preprocessing | Kareem Elawaadi |
| 4 | `4_EDA(Omar_tamer).ipynb` | Exploratory Data Analysis | Omar Tamer |
| 5 | `5_Dashboard(Habiba Sameh).ipynb` | Professional interactive Plotly Dash dashboard (KPIs, insights, 7 charts, filters) | Habiba Sameh |
| 6 | `6_Review_Prediction_Model_(Habiba Sameh).ipynb` | Review prediction model | Habiba Sameh |

---

## 📊 Interactive Dashboard

A professional, multi-view **Plotly Dash** dashboard (`Dashboard/5_Dashboard(Habiba Sameh).ipynb`) built on the master dataset in `data/processed/brazilian_ecommerce_data.csv`.

### Features

| Section | What it shows |
|---------|---------------|
| **KPI strip** | Total sales, orders, average order value, average review score, average delivery time, on-time delivery % |
| **Key insights** | Auto-generated business takeaways that update with every filter (top category, peak month, category concentration, late-delivery impact on reviews, freight share) |
| **Sales trend** | Monthly sales, order volume and a 3-month moving average |
| **Category analysis** | Top 10 categories by sales + a bubble matrix (average price vs. average review, sized by sales) |
| **Customer behaviour** | Weekday × hour heatmap of order activity |
| **Customer experience** | Review score distribution, delivery time by review score, monthly on-time delivery rate |

**Interactivity:** multi-select filters for **year** and **product category**, plus a one-click **reset**. Every KPI, insight and chart is driven by the same filtered slice, so all views stay consistent.

### Metric definitions

| Metric | Definition |
|--------|------------|
| Sales | Sum of `price` over order items (freight excluded) |
| Orders | Unique `order_id` count |
| Average order value | Mean of per-order total `price` |
| Avg review | Mean `review_score` per unique order |
| Delivery time | `order_delivered_customer_date − order_purchase_timestamp` (delivered orders only) |
| On-time % | Delivered orders with delivery date ≤ `order_estimated_delivery_date` |

### Data rules applied in the dashboard

- Cancelled and unavailable orders are excluded (no realised revenue).
- Months at the edges of the dataset with very few records (< 500 order lines, e.g. Sep–Dec 2016 and Sep 2018) are dropped so they don't distort trends.
- Reviews and delivery metrics are computed per unique order, since the dataset has one row per order item.
- Items without a category are labelled *Unknown* and hidden from category charts.

### Run the dashboard

```bash
pip install -r requirements.txt
jupyter notebook Dashboard/"5_Dashboard(Habiba Sameh).ipynb"
```

Run all cells, then open **http://127.0.0.1:8070**.
The notebook looks for the data file automatically (absolute local path → `../data/processed/` → `data/processed/` → next to the notebook). If your file lives elsewhere, edit `DATA_CANDIDATES` in the *Configuration* cell.

> 📸 Add a screenshot of the dashboard to the repo (e.g. `Dashboard/dashboard_preview.png`) and reference it here:
> `![Dashboard preview](Dashboard/dashboard_preview.png)`

---

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mohamed-Ali-Shatla/depi-mini-and-final-projects.git
   cd depi-mini-and-final-projects/mini-projects
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the raw data**  
   The raw database is not stored in the repo. Use the link inside `data/raw/raw_data_link.md`.

4. **Run the notebooks in order** (1 → 2 → 3 → 4 → 5 → 6) from the `notebooks/` folder.

---

## 🗺️ Roadmap

- [x] SQL querying & master dataset export
- [x] Data cleaning
- [x] Data preprocessing
- [x] Exploratory Data Analysis (EDA)
- [x] Professional interactive dashboard (`Dashboard/`) — KPIs, auto-insights, filters, 7 charts
- [x] Review prediction model (`Model/`)
