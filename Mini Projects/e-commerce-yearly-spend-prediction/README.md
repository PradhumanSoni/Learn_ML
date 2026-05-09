🛒 E-Commerce Customer Yearly Spending Prediction

A mini machine learning project that uses **Linear Regression** to predict the yearly amount spent by customers of an e-commerce platform based on their behavioral metrics (app usage, website usage, session length, and membership duration).

---

## 📌 Project Overview

This project analyzes an e-commerce dataset to determine which customer behaviors most strongly influence yearly spending. The goal is to help the business decide whether to invest more in their **mobile app** or **website** experience.

---

## 📂 Dataset

The dataset (`ecommerce.csv`) contains the following features:

| Column | Description |
|--------|-------------|
| `Avg. Session Length` | Average duration of in-store style advice sessions |
| `Time on App` | Average time spent on the mobile app (in minutes) |
| `Time on Website` | Average time spent on the website (in minutes) |
| `Length of Membership` | Number of years the customer has been a member |
| `Yearly Amount Spent` | **Target** — Total amount spent yearly by the customer |

---

## 🛠️ Tech Stack / Libraries Used

- **pandas** — Data manipulation
- **numpy** — Numerical computations
- **matplotlib & seaborn** — Data visualization
- **scikit-learn** — Machine learning (Linear Regression, train/test split, metrics, cross-validation)
- **scipy** — Statistical tests (Shapiro-Wilk, Omnibus, Q-Q plot)

---

## 🔄 Project Workflow

### 1️⃣ Data Inspection & Cleaning
- Loaded the dataset and inspected shape, info, and summary statistics.
- Verified there were **no null values** and the data was already clean.
- Dropped non-numeric / non-essential columns for analysis.

### 2️⃣ Assumptions of Linear Regression
Checked the three key assumptions:
1. **Linearity** between features and target.
2. **Normality of residuals**.
3. **Homoscedasticity** (constant variance of errors).

### 3️⃣ Exploratory Data Analysis (EDA)

#### Univariate Analysis
- Plotted histograms with KDE for all features.
- Checked **skewness** and **kurtosis** — values close to 0 indicate near-normal distribution.
- Performed **Shapiro-Wilk test** → All features are approximately normally distributed.

#### Bivariate Analysis
- Used `sns.pairplot()` to inspect relationships with the target.
- Built a **correlation heatmap**:
  - `Length of Membership` ↔ `Yearly Amount Spent` → **0.81** (strong)
  - `Time on App` ↔ `Yearly Amount Spent` → **0.50** (moderate)
  - `Avg. Session Length` ↔ `Yearly Amount Spent` → **0.36** (weak)
  - `Time on Website` ↔ `Yearly Amount Spent` → ~0 (no correlation)

#### Multicollinearity Check
- No significant correlation between features → safe to fit Linear Regression.

### 4️⃣ Model Building
- Split the data into **80% training / 20% testing** using `train_test_split`.
- Trained a **Linear Regression** model on the training set.
- Predicted on the test set.

### 5️⃣ Model Evaluation

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error |
| **RMSE** | Root Mean Squared Error |
| **R² (Train)** | Goodness of fit on training data |
| **R² (Test)** | Goodness of fit on testing data |

- Compared **Train vs Test R²** — absolute difference < 0.05 → **Model is perfectly fitted (no overfit/underfit)**.

### 6️⃣ Residual Analysis
- Plotted **Residuals vs Predicted Values** → randomly scattered around 0 (homoscedasticity holds).
- Plotted **Residual histogram** → bell-shaped curve.
- **Shapiro-Wilk** & **Omnibus** tests → p-value > 0.05 → Residuals are normally distributed.
- **Q-Q plot** confirmed normality of residuals.

### 7️⃣ Cross Validation
- Performed **5-Fold Cross Validation** to verify model stability.
- Reported mean R² with standard deviation.

---

## 📊 Key Insights

- **Length of Membership** is by far the strongest predictor of yearly spending.
- **Time on App** has a stronger impact on revenue than **Time on Website**.
- 💡 **Business Recommendation:** Investing further in the **mobile app experience** is likely to yield higher returns than the website. Additionally, focusing on **customer retention** (longer memberships) directly boosts revenue.

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
