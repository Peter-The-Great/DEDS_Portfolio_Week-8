# %% [markdown]
# # Week 9-1: Unsupervised Machine Learning
# 
# Van Pjotr en Sennen.
# 
# Deze week hebben we het over ongesuperviseerde machine learning. Dit is een vorm van machine learning waarbij we geen gelabelde data hebben. Dit betekent dat we geen data hebben waarbij we weten wat de juiste output is. In plaats daarvan gaan we op zoek naar patronen in de data. Dit kan bijvoorbeeld zijn dat we clusters van data vinden, of dat we de data kunnen reduceren naar een kleinere dimensie.
# 
# Maar deze keer gaan we clustering gebruiken, een techniek waarbij we data in groepen verdelen. Dit kan bijvoorbeeld handig zijn als we een dataset hebben met verschillende soorten bloemen, en we willen weten welke bloemen bij elkaar horen. Of als we een dataset hebben met verschillende soorten klanten, en we willen weten welke klanten bij elkaar horen.
# 
# PR9-1: Great Outdoors wil graag weten in welke segmenten verkoopafdelingen (‘sales_branches’) opgedeeld kan worden. Er bestaan al retailersegmenten (table ‘retailer_segment’), Great Outdoors wil dus óók segmenten creëren voor verkoopafdelingen:
# - Pas waar nodig Dummy Encoding toe.
# - Train het initiële clustermodel.
# - Experimenteer met meerdere k’s door het berekenen van de inter- en intraclusterafstand.
# 
# Maar eerst een voorbeeld met de titanic.

# %% [markdown]
# ## Bibliotheken importeren

# %%
import pandas as pd
import sqlite3
from sklearn.cluster import KMeans
import math
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.simplefilter('ignore')

# %% [markdown]
# ## Data inlezen en kolommen selecteren

# %%
df = pd.read_csv("data/titanic2.csv", sep = ';')
df = df.filter(regex='^(?!Unnamed).*$')
df

# %%
df = df.loc[:, ['Pclass', 'Sex', 'Age', 'Survived']]
df

# %% [markdown]
# ## One-hot encoding van onafhankelijke niet-numerieke variabelen

# %%
df['Pclass'] = df['Pclass'].astype(str)
df.dtypes

# %%
dummies_dataframe = pd.get_dummies(df.loc[:, ['Sex', 'Pclass']])
dummies_dataframe

# %%
df = pd.concat([df, dummies_dataframe], axis = 1)
df = df.drop(['Sex'], axis = 1)
df = df.loc[:, ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'Age', 'Survived']]
df

# %% [markdown]
# ## Clusteringmodel bouwen met 2 dimensies

# %%
df_2d = df.loc[:, ['Age', 'Survived']]
df_2d

# %%
kmeans = KMeans(n_clusters = 3, random_state = 42)
kmeans.fit_predict(df_2d)

# %%
kmeans_centra = pd.DataFrame(kmeans.cluster_centers_)
kmeans_centra

# %%
for i in range(len(kmeans_centra.columns)):
    kmeans_centra = kmeans_centra.rename(columns = {i : f'{df_2d.columns[i]}'})
    
kmeans_centra

# %%
for src_index, _ in df_2d.iterrows():
    euclidian_distances = dict()
    print(f"Afstand van bronindex {src_index} tot...")

    for centrumindex, _ in kmeans_centra.iterrows():
        print(f"\tCentrumindex {centrumindex}:")
        euclidian_sum = 0

        for column_name in kmeans_centra.columns:
            current_difference = df_2d.at[src_index, column_name] - kmeans_centra.at[centrumindex, column_name]
            print(f'\t\t{df_2d.at[src_index, column_name]} - {kmeans_centra.at[centrumindex, column_name]} = {current_difference}')
            euclidian_sum += current_difference ** 2
        
        print(f'\tTotale euclidische som: {euclidian_sum}')
        euclidian_distance = math.sqrt(euclidian_sum)
        print(f'\tEuclidische afstand: {euclidian_distance}')
        euclidian_distances[centrumindex] = euclidian_distance
        print('------------------------------------------------')
    
    print(euclidian_distances)
    centrum_number = min(euclidian_distances, key = euclidian_distances.get)
    print(centrum_number)
    df_2d.at[src_index, 'Centrum'] = centrum_number
    print("================================================")

df_2d

# %%
plt.scatter(df_2d['Age'], df_2d['Centrum'], color = 'k')
plt.show()

# %%
df_2d.groupby('Centrum', as_index = False)['Centrum'].count()

# %% [markdown]
# ## Clusteringmodel bouwen met meer dan 2 dimensies (alle kolommen uit de dataset)

# %% [markdown]
# ### Centra berekenen

# %%
kmeans = KMeans(n_clusters = 6, random_state = 42)
prediction_results = kmeans.fit_predict(df)
prediction_results

# %%
df['Centrum'] = prediction_results
df

# %% [markdown]
# ### De juiste centra toewijzen aan rijen uit de dataset

# %%
df.groupby('Centrum', as_index = False)['Centrum'].count()

# %% [markdown]
# # De Sales Branches clusteren
# 

# %%
# Verbinding maken met de databases
conn_sales = sqlite3.connect('data/go_sales.sqlite')
conn_crm = sqlite3.connect('data/go_crm.sqlite')
conn_staff = sqlite3.connect('data/go_staff.sqlite')

# Gegevens ophalen uit de databases
sales_branch = pd.read_sql_query("SELECT * FROM sales_branch", conn_staff)
country1 = pd.read_sql_query("SELECT * FROM country", conn_sales)
country2 = pd.read_sql_query("SELECT * FROM country", conn_crm)
territory = pd.read_sql_query("SELECT * FROM sales_territory", conn_crm)

# %% [markdown]
# Hier gaan we de data van de great outdoors inlezen en bepaalde data dropen.

# %%
country = country1[['CURRENCY_NAME', 'LANGUAGE']]
country = pd.concat([country2, country], axis = 1)
sales_branch.drop('TRIAL633', axis=1, inplace=True)
country.drop('TRIAL219', axis=1, inplace=True)
territory.drop('TRIAL222', axis=1, inplace=True)

# %% [markdown]
# Nu gaan we mergen.

# %%
table = pd.merge(country, territory, left_on='SALES_TERRITORY_CODE', right_on='SALES_TERRITORY_CODE', how='left')
table = pd.merge(sales_branch, table, left_on='COUNTRY_CODE', right_on='COUNTRY_CODE', how='left')
table

# %%
table = table[['CURRENCY_NAME', 'COUNTRY_EN', 'TERRITORY_NAME_EN']]
df = pd.get_dummies(table)  # Dummy encoding
df.dropna(inplace=True)  # Optioneel: verwijder rijen met ontbrekende waarden

table = pd.concat([table, df], axis=1)
table

# %%
# Train het clustermodel
kmeans = KMeans(n_clusters=4, random_state=42)
prediction_results = kmeans.fit_predict(df)
prediction_results

kmeans_centra = pd.DataFrame(kmeans.cluster_centers_)
kmeans_centra

for i in range(len(kmeans_centra.columns)):
    kmeans_centra = kmeans_centra.rename(columns = {i : f'{df.columns[i]}'})
    
kmeans_centra

# %%
for src_index, _ in df.iterrows():
    euclidian_distances = dict()
    print(f"Afstand van bronindex {src_index} tot...")

    for centrumindex, _ in kmeans_centra.iterrows():
        print(f"\tCentrumindex {centrumindex}:")
        euclidian_sum = 0

        for column_name in kmeans_centra.columns:
            current_difference = df.at[src_index, column_name] - kmeans_centra.at[centrumindex, column_name]
            print(f'\t\t{df.at[src_index, column_name]} - {kmeans_centra.at[centrumindex, column_name]} = {current_difference}')
            euclidian_sum += current_difference ** 2
        
        print(f'\tTotale euclidische som: {euclidian_sum}')
        euclidian_distance = math.sqrt(euclidian_sum)
        print(f'\tEuclidische afstand: {euclidian_distance}')
        euclidian_distances[centrumindex] = euclidian_distance
        print('------------------------------------------------')
    
    print(euclidian_distances)
    centrum_number = min(euclidian_distances, key = euclidian_distances.get)
    print(centrum_number)
    df.at[src_index, 'Centrum'] = centrum_number
    print("================================================")

df

# %%
plt.scatter(table['TERRITORY_NAME_EN'], table['CURRENCY_NAME'], c = prediction_results, cmap = 'rainbow')
plt.show()


