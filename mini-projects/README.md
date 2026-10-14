# 🛒 Brazilian E-Commerce Sales Analysis

An end-to-end Data Science project analyzing Brazilian e-commerce transactions, customer reviews, and sales performance to extract business insights and build a predictive sales model.

---

## 📁 Repository Structure

```text
mini-projects/
│
├── dashboard/
│   └── .gitkeep                                        # Placeholder for future Streamlit/Plotly dashboard
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
├── notebooks/
│   ├── 1_data_querying_and_export_(mohamed_shatla).ipynb   # SQL data extraction & master dataset export
│   ├── 2_cleaned_data_(salma mamdouh).ipynb                # Data cleaning
│   └── 3_preprocessed(Kareem_Elawaadi).ipynb               # Data preprocessing
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
raw (Olist DB) ──► processed ──► cleaned ──► preprocessed
   SQL query        Notebook 1    Notebook 2    Notebook 3
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

4. **Run the notebooks in order** (1 → 2 → 3) from the `notebooks/` folder.

---

## 🗺️ Roadmap

- [x] SQL querying & master dataset export
- [x] Data cleaning
- [x] Data preprocessing
- [ ] Exploratory Data Analysis (EDA)
- [ ] Predictive sales model
- [ ] Interactive dashboard (`dashboard/`)
