import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Research Question:
# Does the rating determine the movie revenue?

# ------------------- Functions -----------------------------------

def dropMissing(cleaned):
    if cleaned.isna().any().any():
        return cleaned.dropna()
    else:
        return cleaned


def clean_df(revenue_df):
    cleaned = revenue_df.copy()

    cleaned['Rating'] = pd.to_numeric(
        cleaned["Rating"].astype(str).str.replace('/10', '', regex=False),
        errors='coerce'
    )
    cleaned['$Worldwide'] = pd.to_numeric(cleaned['$Worldwide'], errors='coerce')
    cleaned['Vote_Count'] = pd.to_numeric(cleaned['Vote_Count'], errors='coerce')
    cleaned['Year'] = pd.to_numeric(cleaned['Year'], errors='coerce')

    cleaned = dropMissing(cleaned)

    cleaned = cleaned[cleaned['$Worldwide'] > 0]
    cleaned = cleaned[cleaned['Rating'] > 0]
    cleaned = cleaned[cleaned['Vote_Count'] > 0]

    return cleaned


def statisticsRevenue(cleaned_df):
    rev = cleaned_df['$Worldwide']
    rev_mean = round(float(rev.mean()), 3)
    rev_median = round(float(rev.median()), 3)
    rev_var = round(float(rev.var()), 3)
    rev_std = round(float(rev.std()), 3)
    rev_q1 = round(float(rev.quantile(.25)), 3)
    rev_q3 = round(float(rev.quantile(.75)), 3)
    rev_iqr = round(float(rev_q3 - rev_q1), 3)

    rev_stats = {
        "Revenue Mean": rev_mean,
        "Revenue Median": rev_median,
        "Revenue Variance": rev_var,
        "Revenue Standard Deviation": rev_std,
        "Revenue Quarter One": rev_q1,
        "Revenue Quarter Three": rev_q3,
        "Revenue IQR": rev_iqr
    }
    return rev_stats


def statisticsRatings(cleaned_df):
    rating = cleaned_df['Rating']
    rating_mean = round(float(rating.mean()), 3)
    rating_median = round(float(rating.median()), 3)
    rating_var = round(float(rating.var()), 3)
    rating_std = round(float(rating.std()), 3)
    rating_q1 = round(float(rating.quantile(.25)), 3)
    rating_q3 = round(float(rating.quantile(.75)), 3)
    rating_iqr = round(float(rating_q3 - rating_q1), 3)

    rating_stats = {
        "Rating Mean": rating_mean,
        "Rating Median": rating_median,
        "Rating Variance": rating_var,
        "Rating Standard Deviation": rating_std,
        "Rating Quarter One": rating_q1,
        "Rating Quarter Three": rating_q3,
        "Rating IQR": rating_iqr
    }
    return rating_stats


def genreRevenue(cleaned_df):
    genre_df = cleaned_df.copy()
    genre_df["Genres"] = genre_df["Genres"].astype(str)
    genre_df = genre_df.assign(Genres=genre_df["Genres"].str.split(","))
    genre_df = genre_df.explode("Genres")
    genre_df["Genres"] = genre_df["Genres"].str.strip()

    genre_stats = genre_df.groupby("Genres")["$Worldwide"].mean().sort_values(ascending=False)
    return genre_stats


# ---------------- Set up DF -------------------------------------------------------

movie_df = pd.read_csv("enhanced_box_office_data(2000-2024)u.csv")

revenue_df = movie_df[['Release Group', '$Worldwide', 'Rating', 'Vote_Count', 'Year', 'Genres']]

cleaned_df = clean_df(revenue_df)

if len(cleaned_df) > 5000:
    cleaned_df = cleaned_df.sample(5000, random_state=42)

print("Cleaned Data Shape:", cleaned_df.shape)
print(cleaned_df.head())

# ------------- Analyze DF -------------------------------

rev_stats = statisticsRevenue(cleaned_df)
rating_stats = statisticsRatings(cleaned_df)

print("\nRevenue Statistics")
for rev, stats in rev_stats.items():
    print(rev, stats)

print("\nRating Statistics")
for rating, stats in rating_stats.items():
    print(rating, stats)

# --------- Data Visualization -----------------

genre_stats = genreRevenue(cleaned_df)

plt.figure(figsize=(12, 6))
genre_stats.head(10).plot(kind='bar')
plt.title("Top 10 Genres by Average Worldwide Revenue")
plt.xlabel("Genre")
plt.ylabel("Average Worldwide Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(cleaned_df['Rating'], bins=20)
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(cleaned_df['Rating'], cleaned_df['$Worldwide'], alpha=0.5)
plt.title("Movie Rating vs Worldwide Revenue")
plt.xlabel("Rating")
plt.ylabel("Worldwide Revenue")
plt.tight_layout()
plt.show()

# --------- Linear Regression -----------------

X = cleaned_df[['Rating', 'Vote_Count', 'Year']]
y = cleaned_df['$Worldwide']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

y_pred = reg_model.predict(X_test)

print("\nLinear Regression Results")
print("Inputs (X): Rating, Vote_Count, Year")
print("Output (y): $Worldwide")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": reg_model.coef_
})
print("\nRegression Coefficients")
print(coef_df)

# --------- Machine Learning Model -----------------

median_revenue = cleaned_df['$Worldwide'].median()
cleaned_df['High_Revenue'] = (cleaned_df['$Worldwide'] >= median_revenue).astype(int)

X_ml = cleaned_df[['Rating', 'Vote_Count', 'Year']]
y_ml = cleaned_df['High_Revenue']

X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(
    X_ml, y_ml, test_size=0.2, random_state=42, stratify=y_ml
)

scaler = StandardScaler()
X_train_ml = scaler.fit_transform(X_train_ml)
X_test_ml = scaler.transform(X_test_ml)

ml_model = LogisticRegression(max_iter=1000)
ml_model.fit(X_train_ml, y_train_ml)

y_pred_ml = ml_model.predict(X_test_ml)

print("\nMachine Learning Model Results")
print("Model: Logistic Regression")
print("What is being predicted: High revenue movie or low revenue movie")
print("Accuracy:", accuracy_score(y_test_ml, y_pred_ml))

print("\nClassification Report")
print(classification_report(y_test_ml, y_pred_ml))

print("Confusion Matrix")
print(confusion_matrix(y_test_ml, y_pred_ml))
