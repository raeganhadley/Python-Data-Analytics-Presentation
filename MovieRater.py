import pandas as pd
import matplotlib.pyplot as plt

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



        
    

#----------------Set up DF-------------------------------------------------------

movie_df = pd.read_csv("enhanced_box_office_data(2000-2024)u.csv")#loads csv


revenue_df = movie_df[['Release Group', '$Worldwide','Rating']]#focuses on main research categories

cleaned_rev_df = clean_df(revenue_df)

#-------------Analyze DF-------------------------------














