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

@st.cache_data
def generate_random_value(x): 
  return random.uniform(0, x) 
a = generate_random_value(10) 
b = generate_random_value(20) 
st.write(a) 
st.write(b)

st.title("Predicting the purchasing behavior of users on e-commerce site")
st.sidebar.title("Table of contents")
pages=["Project overview","Pre-Processing", "Data Vizualization","Feature Engineering and Modelling","Interpretation of Results","Conclusion"]
page=st.sidebar.radio("Go to", pages)
# Names of the creators
creators = ["Julian Lowe"]

# Display the names in the sidebar
st.sidebar.write("Created by:")   
for creator in creators:
    st.sidebar.write(creator)
st.sidebar.write('April 2024')



base="dark"
secondaryBackgroundColor="#f2f8fc"
font="Helvetica"
event_df = pd.read_csv('events.csv')
category_tree = pd.read_csv('category_tree.csv')
item_properties_part1 = pd.read_csv('item_properties_part1.csv')
item_properties_part2 = pd.read_csv('item_properties_part2.csv')
if page == pages [0] : 
   # Display the image
    st.image("ConversionPredictionModel.jpg", width=None)

    # Header
    st.header("Conversion Prediction Model")
    st.subheader("Helping Ecommerce Operators improving their conversion rate")

    # Text content
    st.markdown("""
    When users engage with an e-commerce site, various information in regards to their online activity is tracked
    and stored as data. This data can be used to understand user behavior on an e-commerce site and help building models.
    Specifically, the focus lies on developing a **Conversion Prediction Model**. This model endeavors to discern patterns
    within the dataset that can accurately identify which visitors will likely proceed with a transaction. By leveraging
    advanced machine learning algorithms and predictive analytics, the aim is to forecast visitor behavior and anticipate
    conversion events with precision. This predictive capability holds immense potential for e-commerce operators,
    enabling them to tailor marketing strategies, optimize resource allocation, optimize website design, and enhance
    the overall efficiency of their platforms.

    ## Unlocking E-commerce Success: The Power of Conversion Prediction Models

    Creating a conversion prediction model for e-commerce operators is vital for maximizing sales numbers and optimizing
    business operations. Here is how it help:
                
    - It enables operators to allocate marketing resources more effectively by identifying the channels and campaigns likely to yield the highest conversions.
    This ensures a higher return on investment (ROI) and better utilization of resources.
    - Prediction models facilitate personalized customer experiences. By understanding each customer's likelihood
    to convert, operators can tailor product recommendations, promotions, and messaging accordingly. This personalization
    increases the probability of successful conversions and fosters customer loyalty.
    - The models aid in inventory management by forecasting demand for specific products. By maintaining
    optimal stock levels, e-commerce operators can avoid stockouts and excess inventory, capitalizing on potential sales
    opportunities and enhancing customer satisfaction.
    - They contribute to a smoother user experience by streamlining the checkout process based on
    past behavior and preferences. This reduces cart abandonment rates and increases conversion rates, resulting in
    improved overall customer satisfaction and retention.
    """)
  

if page == pages[1] :
  st.header("Pre-Processing") 
  st.subheader("Data Overview und Usage")
  st.write("""
    In total, there are 4 datasets available, which were taken from the website: [Kaggle RetailRocket eCommerce Dataset](https://www.kaggle.com/retailrocket/ecommerce-dataset/home) and are allowed to be shared and modified under the license: [CC BY-NC-SA 4.0 DEED](https://creativecommons.org/licenses/by-nc-sa/4.0/).
           
    Overview of Datasets:""")
  st.image("overview-datasets.png", width=None)       

  st.markdown("""The **item property** datasets were used in the Exploratory Analysis for visualizations concerning the price. The **category tree** dataset was not used as the link between category ID to parent ID did not provide any additional value for the project objective. The main dataset that was used was the **events** dataset, as it pertained to the user behavior and also provided information about item transactions.
  The dataset “events.csv” comprises 2.7 million rows of data documenting events generated by users. 
  
  Each row contains the following fields:
  - **Event**: Denotes the type of event that occurred. Events could include various user interactions.
  - **Timestamp**: Records the precise date and time when the event occurred. The timestamp provides granularity, allowing analysis of user behavior over time.
  - **Visitor ID**: Each user or visitor to the platform is assigned a unique identifier. This ID tracks individual user interactions and analyzes their behavior patterns across different events.
  - **Item ID**: Identifies the specific product or item associated with the event. For instance, if the event is a purchase, the item ID would correspond to the product bought by the user.
  - **Transaction ID**: Applicable only for events related to transactions, such as purchases. It records a unique identifier for each transaction, facilitating analysis of purchase behavior and revenue generation.
  - **Type of Event**: Categorizes the event into different types, providing insights into the nature of user interactions.
  - **Date of Event**: Complements the timestamp by providing the date (without the time component) on which the event occurred.
  - **Visitor ID Creating the Event**: Similar to the Visitor ID, this field records the unique identifier of the user who initiated the event. It helps understand user engagement and the role of specific users in driving platform activity.
  - **Product ID associated with the Event**: Mirrors the Item ID and specifies the product or item involved in the event.
    """)
  
  st.subheader('Preview Dataframes')

  st.write("The first step in the data analysis process involved importing the datasets required for the project. "
         "This included loading four separate datasets: events.csv (event_df), category_tree.csv (category_tree), "
         "and item_properties_1.csv, item_properties_2.csv (item_properties). By previewing these dataframes and "
         "understanding their dimensions and contents, the groundwork is laid for subsequent data preparation and "
         "cleaning steps essential for meaningful analysis.")
  
  st.write("**Events Dataframe:**")
  st.dataframe(event_df.head(3))
  st.write("**Category Dataframe:**")
  st.dataframe(category_tree.head(3))
  st.write("**Item Property Part 1:**")
  st.dataframe(item_properties_part1.head(3))
  st.write("**Item Property Part 2:**")
  st.dataframe(item_properties_part2.head(3))
  
  st.subheader('Management of Missing Values')


  st.write("Following data import, missing values were assessed across all datasets. Using the .isna().sum() method, "
         "missing value counts were determined. This process aids in identifying gaps in data completeness, enabling "
         "subsequent strategies for handling missing values. It was observed that the 'transactionid' column has a "
         "considerable number of missing values (2733184), which makes sense since the dataset includes all users and "
         "not only the converters. This finding underscores the importance of addressing missing data to ensure the "
         "reliability of subsequent analyses. In the category tree dataframe, the analysis revealed missing values in "
         "the 'parentid' column, totaling 25 instances. Upon examination of the item properties dataframe, it was "
         "determined that there are no missing values exists in any of the columns ('timestamp', 'itemid', 'property', "
         "'value'). This finding indicates that the dataset is complete regarding missing values for these attributes. "
         "As the main focus of our analysis was on the ‘events’ dataset, we decided to proceed with the missing values "
         "in the 'transactionid' column as followed: Since the transaction id was our target variable, our main concern "
         "was whether the user converted or not. So we considered the variable as a Boolean format, where if a transaction "
         "id is present, then add 1, else add 0.")

  st.subheader('Management of Duplicates')

  st.write("A check for duplicate entries within the dataframes was performed using the .duplicated().sum() method. "
         "The analysis revealed that no duplicate entries were found in the dataset. This indicates that each record "
         "within the dataframes are unique and eliminates the need for further processing to handle duplicate entries.")

  st.subheader('Hashed Values and Formatting')

  st.write("The 'timestamp' column within the events data frame underwent formatting operations to convert Unix timestamps "
         "into human-readable date and time formats. Additional datetime attributes, such as day of the week, hour, day, "
         "month, and week number, were derived to provide insights into temporal aspects of the data. Furthermore, the "
         "start and end dates of the dataset were determined to understand the temporal scope of the data collection. "
         "Further, in the item properties dataset, the price format in the value column had an ‘n’ character in the "
         "beginning and three digits precision after the decimal point (for example: 8 = n8.000). In order to ensure "
         "accurate numerical representation any extraneous characters were removed from the 'value' column by using the "
         "slicing method. Lastly, product information in the value column was hashed as well. It seems to be for privacy "
         "reasons.")

  st.subheader('Timeframe of Data')

  st.write("By converting the Unix timestamps, it is possible to determine the time period in which the data was collected. "
         "The data provided ranges from the date of 03.05.2015 to 18.09.2015.")

  


if page == pages[2] : 
  st.write("### Data Vizualization")

  st.subheader("Distribution of Events")
  st.write("Event Distribution: Initially, the count of each event type is calculated, followed by the calculation of event percentages to understand their relative distribution. The pie chart below illustrates the distribution of events, showcasing the proportion of each event type relative to the total number of events: “view” 96,68 %, “addtocart” 2,50 % and “transaction” 0.81 %.")

  event_count = event_df['event'].value_counts()
  labels = event_count.index
  sizes = event_count.values
  explode = (0, 0.15, 0.15)
  fig, ax = plt.subplots(figsize=(8, 8))
  ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=0)
  ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
  st.pyplot(fig)


st.set_option('deprecation.showPyplotGlobalUse', False) 

def new_func(model_df, selected_vars, include_hue):
    if selected_vars:
        st.subheader('Pair Plot')
        if include_hue:
            sns.pairplot(model_df, x_vars=selected_vars, y_vars=selected_vars, hue='purchased')
        else:
            sns.pairplot(model_df, x_vars=selected_vars, y_vars=selected_vars)
        st.pyplot()
    else: 
        st.write('Please select at least one variable for the pair plot.')

if page == pages[3]:
    st.write("### Feature Engineering & Modelling")

    st.subheader('Feature Engineering')

    st.write("In order to prepare the data for the modeling, it was first important to split up the ‘events’ data frame based "
         "on users who have visited the site (visitors), users who have added an item to their cart (window-shoppers), "
         "and people who completed a transaction (buyers). After, we used the method of Feature Engineering in order to "
         "create additional columns, which would be relevant for the model to use as part of their conversion prediction. "
         "These columns were added:")
    
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
    st.dataframe(buying_visitors_df.head())
    st.write("To have a 75% - 25% split for the training and test data we need a total sample size of 46.876 (35.157 + 11719). To achieve this size we take the visitor only list.")

    # To avoid a bias, the list will be shuffled before the data frame creation
    random.shuffle(visit_only)

    visit_only_df = create_dataframe(visit_only[:35157])

    st.dataframe(visit_only_df.head())

    st.write("# Merge of the two data frames")
    model_df = pd.read_csv("model_df.csv")
    st.dataframe(model_df.head())

    st.write("Before choosing the appropriate models, we initially created a pairplot in order to analyze the correlation between each variable from the data frame.")

    # Allow users to choose variables for pair plot
    selected_vars = st.multiselect('Select variables for pair plot', ['num_items_viewed', 'view_count', 'num_items_added', 'items_added', 'bought_count'])
    # Allow users to include hue based on the 'purchased' column
    include_hue = st.checkbox('Include hue based on "purchased" column')

    # Create pair plot
    new_func(model_df, selected_vars, include_hue)

    st.write("As expected, we saw a linear correlation between view counts and bought counts as well as the number of items added and the bought count. To be specific, this means that the higher the user engagement with the e-commerce site the more likely a user will complete a transaction. A linear relationship is identified and the nature of searching for a qualitative target variable (Will User X make a purchase?)  means that we are dealing with Classification problem. Therefore, three classification algorithms, namely Logistic Regression, Decision Tree Classifier, and Random Forest Classifier, are employed and evaluated for their performance in accurately predicting visitor behavior.")
    
    X = model_df.drop(['purchased', 'visitorid', 'bought_count'], axis=1)
    y = model_df.purchased

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

    def prediction(classifier):
        if classifier == 'Logistic Regression':
            clf = joblib.load("model_log_reg.pkl")
        elif classifier == 'Decision Tree':
            clf =  joblib.load("model_dt_clf.pkl")
        elif classifier == 'Random Forest':
            clf =  joblib.load("model_rf_clf.pkl")
        return clf

    def scores(clf, choice):
        if choice == 'Accuracy':
            return clf.score(X_test, y_test)
        elif choice == 'Confusion matrix':
            return confusion_matrix(y_test, clf.predict(X_test))
        
    st.subheader('Modelling')

    choice = ['Logistic Regression', 'Decision Tree', 'Random Forest']
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is :', option)

    clf = prediction(option)
    display = st.radio('What do you want to show ?', ('Accuracy', 'Confusion matrix'))
    if display == 'Accuracy':
        st.write(scores(clf, display))
    elif display == 'Confusion matrix':
        st.dataframe(scores(clf, display))

    st.image('/Users/office2/Streamlit_project/model-overview.png', caption='Optional caption')

    st.subheader('Model Choice and Optimization')

    st.write("The results of all three models are very good at predicting which customers are most likely to complete a "
         "transaction. The Random Forest Classifier is chosen as the primary model due to the slightly better results. "
         "The models are already performing very well. Nevertheless, an attempt was made to achieve an improvement by changing the random_state hyperparameter." 
         "For the logistic regression, the hyperparameter C was also used with different values."
         "A clear improvement to the already very good results could not be achieved, which is why this is not discussed further here."
         "For faster computing power with larger data sets, we recommend limiting the maximum depth of the tree models to 3 in order to still obtain very good results"
         "See also the feature importance section to determine the most relevant explanatory variables in advance")

if page == pages[4] : 
    st.write("### Interpretation of Results")
    
    st.write("The selected model's accuracy in predicting buying visitors is around 0.96 %. This means that the model is quite "
         "accurate at predicting the user behavior and therefore confirms its robustness in regards to new data. All models "
         "performed comparably well, yet the Random Forest Model slightly exceeded all models in terms of Performance. Further, "
         "the precision, recall and f1-score were all above 90 %, which can be interpreted as a confident result in predicting "
         "user behaviour. Overfitting was checked for each model and was subsequently ruled out. This means that the models will "
         "also perform well on other training sets.")
    
    st.subheader('Feature Importance')

    st.image('/Users/office2/Streamlit_project/FI.png', caption='Optional caption')

    st.write("In order to quantify the roles of each variable in the random forest model, we used an attribute called ‘Feature Importance’. "
         "Each feature was given a score in order to determine how effective it could predict the target variable. As can be seen in "
         "the graph, the feature ‘items_added’ was the largest contributor to predicting whether or not a user would purchase an item. "
         "Other relevant factors included the number of items added and the view count. This observation is in line with our overall "
         "hypothesis, which is that there is a higher chance that a user will convert if they already have an item in their cart. "
         "Interestingly, the type of item added to the cart was more important than the amount of items that were added. Overall, "
         "the interpretation of results highlights the value of predictive analytics in understanding and optimizing user behavior "
         "on e-commerce platforms. By leveraging machine learning models and feature importance analysis, businesses can identify "
         "key drivers of conversation, optimize marketing strategies, and enhance the overall user experience to drive business "
         "growth and profitability.")
    
if page == pages[5] : 
  st.write("### Conclusion")
  st.write("Here's the conclusion of our analysis.")
    
    # Insert the image using its URL
  image_path = r"/Users/office2/Streamlit_project/Conclusion.jpg"
  st.image(image_path, caption='Conclusion Image', use_column_width=True)
  st.write("""Limitation

  While the project achieved its objectives and provided valuable insights into user behavior on the e-commerce platform, several limitations should be acknowledged:

           Data Limitations: The available data constrained the analysis, which covered a specific timeframe. Insights drawn from this period may not fully represent long-term trends or seasonal variations in user behavior. 

           Imbalanced Data: The dataset exhibited an inherent class imbalance, with a significantly higher number of non–transaction events (view) than transaction events. Addressing this class imbalance during model training may affect the model’s performance and generalization. 

           Scope of Variables: The predictive models relied on limited variables derived from user events and interactions. Additional data, such as demographic information or user preferences, could enhance the accuracy and granularity of the models.

           Temporal Dynamics: The analysis focused on understanding user behavior within the provided timeframe. The analysis did not explicitly account for changes in market dynamics, consumer preferences, or platform features over time. 
        
           Model Interpretability: While the chosen machine learning models demonstrated high predictive accuracy, their interpretability may be limited. Understanding the underlying mechanisms driving predictions and user behavior may require additional techniques or model architectures. 

           Generalization: The models developed in this project are specific to the dataset and context of the e-commerce platform studied. Generalizing the findings to other platforms or industries may require further validation and adaption of the models. 

           Abnormal User Detection: The analysis did not explicitly address the identification and handling of abnormal user behavior, including fraudulent activities. Such abnormal users may distort patterns and trends in the data, leading to biased model predictions and inaccurate insights. 
                                                      

  Outlook
  
  Users tend to be price sensitive according to the type of product and the price range. The model used in this project could be even more realistic if the variable price is taken into account. Especially finding a correlation between price fluctuations and user demand could be an interesting avenue to explore in the future. For example, what kind of impact do discounts have on user behavior? This presumes that the data available is not in a hashed format.

  E-Commerce sites increasingly face cyber security concerns, including bot/spam traffic. This can have a negative impact on the integrity of internal data. Therefore, it is critical for companies to look for solutions on how to filter out this particular traffic from their data, as it does not add any sort of value. The abnormal traffic can be defined in many ways, though it is important to have an actual definition in order to find actionable counter-measurements then. For this project, a logical next step would be to look into the abnormal users, visualize the quantiles for user views per day in a boxplot, and then define a threshold for abnormality. After, the users above the threshold can be filtered out and run through the model again in order to analyze the differences between the model results with and without the spam traffic. 

  Lastly, when adding additional data into the datasets from other marketing touchpoints it would give a more complete picture of the customer journey and would paint a picture that would be closer to a real-life scenario. To give a better example, imagine the following scenario: User A generates one view on the website, whereas User B generates 2 views. In general, without having any additional information one would assume that User B is more engaged and therefore more likely to convert. However, what if User A got to the website by typing in the website in their search bar, whereas User B accidentally clicked on a display banner on a news site? Now the story has changed, and the intent is more clear. 
  """)
