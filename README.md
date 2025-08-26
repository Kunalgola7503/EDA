# 🏘️ King County House Sales EDA

## 📌 Overview
This project performs Exploratory Data Analysis (EDA) on the **King County House Sales dataset (Seattle, USA)**.  
The goal is to understand patterns, correlations, and key factors affecting house prices.  

## 🔧 Project Workflow
1. **Data Import & Setup**
   - Loaded dataset from local CSV file.
   - Configured Pandas display options and Seaborn styling.

2. **Initial Exploration**
   - Inspected first few rows, dataset shape, and column details.
   - Generated summary statistics (numerical + categorical).

3. **Data Quality Checks**
   - Checked for missing values and their percentage.
   - Counted duplicate rows.

4. **Univariate Analysis**
   - Distribution of numerical variables (histograms, boxplots).
   - Frequency distributions of categorical variables (count plots).

5. **Bivariate Analysis**
   - Correlation heatmap for numerical variables.
   - Boxplots comparing categorical vs numerical features.

6. **Outlier Detection**
   - Boxplots for numerical columns to spot extreme values.

7. **Final Insights**
   - Dataset size and structure.
   - Missing values summary.
   - Duplicate rows count.
   - Imbalanced categorical features.
   - Key correlations between numeric features.
   - Presence of outliers.

## 📊 Tools & Libraries
- Python 🐍
- Pandas
- NumPy
- Matplotlib
- Seaborn

## 📂 Project Structure
KingCounty_HouseSales_EDA/
│── Dataset 
│── Complete code with EDA steps
│── images/ # Plots and visualizations
│── README.md # Project documentation


## ⚠️ Dataset Access
The dataset is **not uploaded in this repository** to respect licensing and size restrictions.  
You can download it from Kaggle here:  
👉 [House Sales in King County, USA (Kaggle)](https://www.kaggle.com/harlfoxem/housesalesprediction)  

After downloading, place the file (`kc_house_data.csv`) inside the `/data` folder before running the notebook.

