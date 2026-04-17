import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Reserch Question: Does the rating determine the movie revenue?

#-------------------Functions -----------------------------------

def dropMissing(cleaned):
    if cleaned.isna().any().any():
        return cleaned.dropna()#drops na if there are any
    else:
        return cleaned#return same list if not
    
        

def clean_df(revenue_df):
    cleaned =revenue_df.copy()#create copy of df to work wiht
    cleaned['Rating'] = pd.to_numeric(cleaned["Rating"].str.strip('/10'),errors='coerce')#strips the "/10" out
    cleaned = dropMissing(cleaned)#drops missing
    return cleaned

def statisticsRevenue(cleaned_df):
    #Revenue
    rev = cleaned_df['$Worldwide']
    rev_mean = round(rev.mean(),3)
    rev_median = round(rev.median(),3)
    rev_var = round(rev.var(),3)
    rev_std = round(rev.std(),3)
    rev_q1 = round(rev.quantile(.25),3)
    rev_q3 = round(rev.quantile(.75),3)
    rev_iqr = round(rev_q3-rev_q1,3)

    rev_stats = {"Revenue Mean":rev_mean,"Revenue Median" : rev_median, "Revenue Variance" : rev_var, "Revenue Standard Deviation": rev_std, "Revenue Quarter One": rev_q1, "Revenue Quarter Three":rev_q3, "Revenue IQR":rev_iqr}
    return rev_stats

def statisticsRatings(cleaned_df):
    #Rating
    rating = cleaned_df['Rating']
    rating_mean = round(rating.mean(),3)
    rating_median = round(rating.median(),3)
    rating_var = round(rating.var(),3)
    rating_std = round(rating.std(),3)
    rating_q1 = round(rating.quantile(.25),3)
    rating_q3 = round(rating.quantile(.75),3)
    rating_iqr = round(rating_q3-rating_q1,3)

    rating_stats = {"Rating Mean":rating_mean,"Rating Median" : rating_median, "Rating Variance" : rating_var, "Rating Standard Deviation": rating_std, "Rating Quarter One": rating_q1, "Rating Quarter Three":rating_q3, "Rating IQR":rating_iqr}
    return rating_stats

        
    

#----------------Set up DF-------------------------------------------------------

movie_df = pd.read_csv("enhanced_box_office_data(2000-2024)u.csv")#loads csv


revenue_df = movie_df[['Release Group', '$Worldwide','Rating']]#focuses on main research categories

cleaned_df = clean_df(revenue_df)

#-------------Analyze DF-------------------------------

rev_stats = statisticsRevenue(cleaned_df)
rating_stats = statisticsRatings(cleaned_df)

for rev, stats in rev_stats.items():
    print(rev,stats)

#---------Data Visualization-----------------














