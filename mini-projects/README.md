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

---

## 📓 Notebooks

| # | Notebook | Description | Author |
|---|----------|-------------|--------|
| 1 | `1_data_querying_and_export_(mohamed_shatla).ipynb` | SQL data extraction & master dataset export | Mohamed Shatla |
| 2 | `2_cleaned_data_(salma mamdouh).ipynb` | Data cleaning | Salma Mamdouh |
| 3 | `3_preprocessed(Kareem_Elawaadi).ipynb` | Data preprocessing | Kareem Elawaadi |
| 4 | `4_EDA(Omar_tamer).ipynb` | Exploratory Data Analysis | Omar Tamer |
| 5 | `5_Dashboard(Habiba Sameh).ipynb` | Interactive dashboard | Habiba Sameh |
| 6 | `6_Review_Prediction_Model_(Habiba Sameh).ipynb` | Review prediction model | Habiba Sameh |

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
- [x] Interactive dashboard (`Dashboard/`)
- [x] Review prediction model (`Model/`)
