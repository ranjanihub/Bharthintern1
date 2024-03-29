#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np 
import pandas as pd


# In[10]:


import pandas as pd
encodings =['utf-8','latin1','ISO-88591','cp1252']
file_path =r"C:\Users\TRINITY ELE\Downloads\archive\spam.csv"
for encoding in encodings:
    try:
        df=pd.read_csv(file_path,encoding=encoding)
        print(f"file successfully read with encoding:{encoding}")
        break
    except UnicodeDecodeError:
        print(f"failed to read:{encoding}")
        continue
if 'df' in locals():
    print("csv file has been successfully loaded")
else:
    print("all encoding attempts failed")


# In[11]:


df.sample(5)


# In[12]:


df.shape


# In[13]:


#data cleaning
df.info()


# In[15]:


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[16]:


df.sample(5)


# In[17]:


df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)


# In[18]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[19]:


df['target'] = encoder.fit_transform(df['target'])


# In[20]:


df.head()


# In[21]:


df.isnull().sum()


# In[22]:


df.duplicated().sum()


# In[23]:


df=df.drop_duplicates(keep='first')


# In[24]:


df.duplicated().sum()


# In[25]:


df.shape


# In[26]:


#eda
df.head()


# In[28]:


df['target'].value_counts()


# In[30]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[31]:


import nltk


# In[32]:


get_ipython().system('pip install nltk')


# In[33]:


nltk.download('punkt')


# In[34]:


df['num_characters']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[35]:


df.head()


# In[36]:


df['num_sentences']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[37]:


df.head()


# In[39]:


df[['num_characters','num_sentences']].describe()


# In[41]:


df[df['target']==0][['num_characters','num_sentences']].describe()


# In[42]:


import seaborn as sns


# In[45]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_characters'])
sns.histplot(df[df['target']==1]['num_characters'], color='red')


# In[46]:


sns.pairplot(df,hue="target")


# In[ ]:




