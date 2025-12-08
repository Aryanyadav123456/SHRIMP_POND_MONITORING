#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas  as pd


# In[9]:


import json
import pandas as pd

with open(r"C:\Users\Admin\Downloads\shrimp_pond_monitoring_app\data\data_sample.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data["data"])   # <-- choose the correct key


# In[10]:


df


# In[27]:


df = df.dropna(subset=['spanningYear', 'cycle'])


# In[28]:


df.isnull().sum()


# In[11]:


import pandas as pd
import numpy as np
from datetime import datetime

def preprocess_pond_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess shrimp pond dataset for production.
    """

    # ---------------------------------------------------
    # 1. CLEAN COLUMN NAMES
    # ---------------------------------------------------
    df.columns = (
        df.columns.str.replace("\n", " ", regex=True)
                  .str.replace(r"\s+", " ", regex=True)
                  .str.strip()
    )

    # Standard column fixes
    rename_map = {
        "Survival Rate": "SurvivalRate",
        "Survival  Rate": "SurvivalRate",
        "Survival\n Rate": "SurvivalRate"
    }
    df = df.rename(columns=rename_map)

    # ---------------------------------------------------
    # 2. CLEAN DATE COLUMNS
    # ---------------------------------------------------
    date_cols = ["Stocking Date", "Sampling Date"]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format="%d-%m-%Y", errors="coerce")

    # ---------------------------------------------------
    # 3. CLEAN PERCENTAGE COLUMNS
    # ---------------------------------------------------
    if "SurvivalRate" in df.columns:
        df["SurvivalRate"] = (
            df["SurvivalRate"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df["SurvivalRate"] = pd.to_numeric(df["SurvivalRate"], errors="coerce") / 100

    # ---------------------------------------------------
    # 4. CONVERT NUMERIC COLUMNS
    # ---------------------------------------------------
    numeric_cols = [
        "Hectares", "Stocking", "SD(psm)", "DOC", "Week", "ABW",
        "Weekly Inc", "AWG(g)", "AWG 1", "AWG 2", "AWG 3",
        "FCR", "TCF", " Weekly TCF(kg)",
        "BM", "BM 1", "ABW 1", "BM 2", "ABW 2", "BM 3", "ABW 3",
        "Survival", "cycle"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(" ", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="ignore")

    # ---------------------------------------------------
    # 5. STANDARDIZE STATUS COLUMN
    # ---------------------------------------------------
    df["status"] = (
        df["status"]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({"ACTIVE": "ACTIVE", "HARVESTED": "HARVESTED"})
    )

    # ---------------------------------------------------
    # 6. FEATURE ENGINEERING
    # ---------------------------------------------------

    # Days between Stocking & Sampling
    df["DaysBetween"] = (df["Sampling Date"] - df["Stocking Date"]).dt.days

    # Month of sampling
    df["SamplingMonth"] = df["Sampling Date"].dt.month

    # Is harvested flag
    df["IsHarvested"] = np.where(df["status"] == "HARVESTED", 1, 0)

    # Active/havested pond label
    df["PondStatus"] = df["status"].map({
        "ACTIVE": "Active",
        "HARVESTED": "Harvested"
    })

    # Convert cycle to int
    if "cycle" in df.columns:
        df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce").astype("Int64")

    # ---------------------------------------------------
    # 7. REMOVE DUPLICATES
    # ---------------------------------------------------
    df = df.drop_duplicates()

    # ---------------------------------------------------
    # 8. SORT DATA FOR CONSISTENCY
    # ---------------------------------------------------
    df = df.sort_values(by=["Pond", "Sampling Date"])

    return df


# In[12]:


df


# In[13]:


df['Stocking Date'] = pd.to_datetime(df['Stocking Date'], dayfirst=True)
df['Sampling Date'] = pd.to_datetime(df['Sampling Date'], dayfirst=True)


# In[17]:


df['Survival\n Rate'] = df['Survival\n Rate'].str.replace('%', '').astype(float)


# In[30]:


df['cycle'] = df['cycle'].astype(int)


# In[31]:


df['cycle']


# In[32]:


df["BM_Gain"] = df["BM 3"] - df["BM 1"]


# In[33]:


df["BM_Gain"] 


# In[34]:


df["Pond_Age"] = (df["Sampling Date"] - df["Stocking Date"]).dt.days


# In[35]:


df["Pond_Age"]


# In[36]:


df.info()


# In[41]:


cat_cols = ["Pond", "Crop ID", "status", "spanningYear"]

for col in cat_cols:
    df[col] = df[col].astype(str).str.strip().str.upper()


# In[44]:


df.describe



# In[45]:


df.info()


# In[49]:


# Strip spaces and convert to snake_case
df.columns = (
    df.columns
    .str.strip()               # remove leading/trailing spaces
    .str.replace(" ", "_", regex=True)   # replace spaces with underscore
    .str.replace("\n", "_", regex=True) # replace newline chars
    .str.replace(r"[()]", "", regex=True) # remove parentheses
    .str.lower()               # convert to lowercase
)

# Check updated column names
print(df.columns.tolist())


# In[55]:


df.groupby("pond")["survival__rate"].mean().sort_values(ascending=False)


# In[59]:


df['pond'] = df['pond'].astype(str)
df['crop_id'] = df['crop_id'].astype(str)
df['hectares'] = df['hectares'].astype(float)
df['stocking_date'] = pd.to_datetime(df['stocking_date'])
df['sampling_date'] = pd.to_datetime(df['sampling_date'])
df['stocking'] = df['stocking'].astype(int)
df['sdpsm'] = df['sdpsm'].astype(int)
df['doc'] = df['doc'].astype(int)
df['week'] = df['week'].astype(str)
numeric_cols = ['abw','weekly_inc','awgg','awg_1','awg_2','awg_3','fcr','tcf','weekly_tcfkg','bm','bm_1','bm_2','bm_3','abw_1','abw_2','abw_3','survival','survival__rate','bm_gain','pond_age']
df[numeric_cols] = df[numeric_cols].astype(float)
df['status'] = df['status'].astype('category')
df['spanningyear'] = df['spanningyear'].astype(str)
df['cycle'] = df['cycle'].astype(int)


# In[60]:


# Growth rate (grams per day)
df['growth_rate'] = df['abw'] / df['doc']

# Survival efficiency
df['survival_efficiency'] = df['survival'] / 100

# Biomass density per hectare
df['biomass_density'] = df['bm'] / df['hectares']

# Pond age in months (if pond_age is in days)
df['pond_age_months'] = df['pond_age'] / 30

# FCR efficiency (lower is better)
df['fcr_efficiency'] = 1 / df['fcr']

# Time between stocking and sampling
df['days_stocked'] = (df['sampling_date'] - df['stocking_date']).dt.days


# In[62]:


import warnings
warnings.filterwarnings('ignore')


# In[63]:


# Encode status: ACTIVE=1, HARVESTED=0
df['status_encoded'] = df['status'].map({'ACTIVE': 1, 'HARVESTED': 0})

# One-hot encode spanning year if needed for ML
df = pd.get_dummies(df, columns=['spanningyear'], drop_first=True)


# In[64]:


# ABW should not be negative
df = df[df['abw'] >= 0]

# FCR should be reasonable (<5)
df = df[df['fcr'] < 5]

# Survival rate should be <=100%
df = df[df['survival__rate'] <= 100]


# In[65]:


# Quick statistics
df.describe().T

# Correlations
corr = df[['abw','weekly_inc','fcr','tcf','bm','survival','growth_rate','biomass_density']].corr()
print(corr)

# Weekly growth plot example
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.lineplot(data=df, x='doc', y='abw', hue='pond')
plt.title('ABW vs DOC per Pond')
plt.show()


# In[67]:


df.to_csv("shrimp_data_cleaned.csv", index=False)


# In[ ]:




