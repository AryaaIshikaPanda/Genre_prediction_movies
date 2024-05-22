#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries 

# In[31]:



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Reading Data

# In[16]:



data = pd.read_csv(r"C:\Users\Aryaa Ishika\Downloads\plot summary.zip")


# In[17]:


data


# # Data Cleaning

# In[19]:


features=['Plot','Genre']
df=data[features]


# In[22]:


data.isnull().sum()


# In[23]:


data.describe(include='all')


# In[24]:


for i in data.columns:
    print(i,':',sum(data[i]=='?'))


# # Feature Extraction

# In[34]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# In[39]:


#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['Plot'], data['Genre'], test_size=0.2, random_state=42)


# In[40]:


# Initialize the TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

# Fit and transform the training data, and transform the testing data
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# # Model Training

# In[41]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Initialize the classifier
model = MultinomialNB()

# Train the model
model.fit(X_train_tfidf, y_train)


# # Evaluation

# In[42]:


# Predict the genres for the test set
y_pred = model.predict(X_test_tfidf)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Print the classification report
print(classification_report(y_test, y_pred))


# In[ ]:




