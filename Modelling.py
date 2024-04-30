import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

event_df = pd.read_csv('events.csv')
category_tree = pd.read_csv('category_tree.csv')
item_properties_part1 = pd.read_csv('item_properties_part1.csv')
item_properties_part2 = pd.read_csv('item_properties_part2.csv')

visitors = event_df.visitorid.unique()
windowShoppers = event_df[event_df['event'] == "addtocart"].visitorid.unique()
buyers = event_df[event_df.transactionid.notnull()].visitorid.unique()

    # Sort arrays by visitorID
visitors.sort()
windowShoppers.sort()
buyers.sort()

    # Amount of visitors, without transaction
visit_only = list(set(visitors) - set(buyers))

    # Amount of visitors, without addtocart
addtocart_only = list(set(visitors) - set(windowShoppers))

def create_dataframe(visitor_list):
        array_for_df = []

        for index in visitor_list:
            # Filter the DataFrame for the current visitor
            v_df = event_df[event_df['visitorid'] == index]

            # Initialize a list to store the data for the current visitor
            temp = [index]

            # Add the total number of unique products viewed
            num_items_viewed = v_df[v_df['event'] == 'view']['itemid'].nunique()
            temp.append(num_items_viewed)

            # Add the total number of views regardless of product type
            view_count = v_df[v_df['event'] == 'view'].shape[0]
            temp.append(view_count)

            # Add the total number of unique products added to cart
            num_items_added = v_df[v_df['event'] == 'addtocart']['itemid'].nunique()
            temp.append(num_items_added)

            # Add the total number of added to cart regardless of product type
            items_added = v_df[v_df['event'] == 'addtocart'].shape[0]
            temp.append(items_added)

            # Add the total number of purchases
            bought_count = v_df[v_df['event'] == 'transaction'].shape[0]
            temp.append(bought_count)

            # Add 1 if the visitor made a purchase, otherwise add 0
            purchased = 1 if bought_count > 0 else 0
            temp.append(purchased)

            # Append the data for the current visitor to the main list
            array_for_df.append(temp)

        # Create a DataFrame from the list of visitor data
        return pd.DataFrame(array_for_df, columns=['visitorid', 'num_items_viewed', 'view_count', 'num_items_added', 'items_added', 'bought_count', 'purchased'])

buying_visitors_df = create_dataframe(buyers)
random.shuffle(visit_only)
visit_only_df = create_dataframe(visit_only[:35157])
model_df = pd.concat([buying_visitors_df, visit_only_df], ignore_index=True)

X = model_df.drop(['purchased', 'visitorid', 'bought_count'], axis=1)
y = model_df.purchased


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train,y_train)

rf_clf = RandomForestClassifier()
rf_clf.fit(X_train,y_train)

joblib.dump(log_reg, "/Users/office2/Streamlit_project/model_log_reg.pkl")
joblib.dump(dt_clf, "/Users/office2/Streamlit_project/model_dt_clf.pkl")
joblib.dump(rf_clf, "/Users/office2/Streamlit_project/model_rf_clf.pkl")