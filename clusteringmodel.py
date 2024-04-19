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
# Om de segmenten van de verkoopafdelingen te bepalen, gaan we de volgende stappen doorlopen:
# - We kunnen de segmenten op basis van producten categorieën en het verkoop van die producten, bepalen welke segmenten we nodig zullen hebben voor onze verkoopafdeling.
# - We pakken daarbij ook de branches van de retailers erbij, zodat we de segmenten van de verkoopafdelingen kunnen bepalen.
# - Daarna zullen we gebaseerd op de omzet en kosten van de verkoopafdelingen, de segmenten bepalen.
# - Waar we dan de gemixte data in een clustering model zullen stoppen en de segmenten bepalen.
# 
# Eerst gaan we de data inladen.

# %%
# Verbinding maken met de databases
conn_sales = sqlite3.connect('data/go_sales.sqlite')
conn_crm = sqlite3.connect('data/go_crm.sqlite')
conn_staff = sqlite3.connect('data/go_staff.sqlite')

# Gegevens ophalen uit de databases (Helaas verouderd)
sales_branch = pd.read_sql_query("SELECT * FROM sales_branch", conn_staff)
# country1 = pd.read_sql_query("SELECT * FROM country", conn_sales)
# country2 = pd.read_sql_query("SELECT * FROM country", conn_crm)
# territory = pd.read_sql_query("SELECT * FROM sales_territory", conn_crm)

# orders_header = pd.read_sql_query("SELECT * FROM order_header;", conn_sales)
# order_details = pd.read_sql_query("SELECT * FROM order_details;", conn_sales)

# %% [markdown]
# Hier gaan we de data van de great outdoors inlezen en bepaalde data dropen.

# %%
# (Helaas Verouderd)
# country = country1[['CURRENCY_NAME', 'LANGUAGE']]
# country = pd.concat([country2, country], axis = 1)
# sales_branch.drop('TRIAL633', axis=1, inplace=True)
# country.drop('TRIAL219', axis=1, inplace=True)
# territory.drop('TRIAL222', axis=1, inplace=True)
# orders_header.drop('TRIAL885', axis=1, inplace=True)
# order_details.drop('TRIAL879', axis=1, inplace=True)
sales_branch.drop('TRIAL633', axis=1, inplace=True)
sales_branch = sales_branch.drop(["ADDRESS1", "ADDRESS2", "POSTAL_ZONE"], axis=1)
sales_branch

# %% [markdown]
# We zorgen er nu voor dat we de sales_branches city en region gaan omzetten in dummies.

# %%
table = sales_branch[['CITY', 'REGION']]
df = pd.get_dummies(table)  # Dummy encoding

# %% [markdown]
# ## Clusteringmodel bouwen met 2 dimensies
# 
# Eerst wordt een 2D-dataset gemaakt met de locaties van de verkoopfilialen.
# 
# Waar we ook dummy encoding toepassen.

# %%
table = pd.concat([sales_branch, df], axis=1)
table

# %% [markdown]
# Een k-means clusteringmodel wordt geïnstantieerd met het gewenste aantal clusters (3 of 4 in het eerste geval, met een willenkeurigheid van 42) en toegepast op de 2D-dataset.

# %%
# Train het clustermodel
kmeans = KMeans(n_clusters=5, random_state=42)
prediction_results = kmeans.fit_predict(df)
prediction_results

kmeans_centra = pd.DataFrame(kmeans.cluster_centers_)
kmeans_centra

for i in range(len(kmeans_centra.columns)):
    kmeans_centra = kmeans_centra.rename(columns = {i : f'{df.columns[i]}'})
    
kmeans_centra

# %% [markdown]
# De centra van de clusters worden geëxtraheerd en weergegeven.
# Voor elk punt in de dataset wordt de afstand tot elk clustercentrum berekend, en het punt wordt toegewezen aan het dichtstbijzijnde cluster.
# 

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

# %% [markdown]
# De resultaten worden gevisualiseerd door een scatterplot te maken van de kwantiteit versus de eenheidsprijs, waarbij de punten zijn gekleurd op basis van hun toegewezen cluster.

# %%
# (Helaas verouderd)
# plt.scatter(table['TERRITORY_NAME_EN'], table['CURRENCY_NAME'], c = prediction_results, cmap = 'rainbow')
# plt.show()
# Nu gaan we een bar weergeven, waar alle quantiteiten worden weergegeven per sales_branch
# Create a DataFrame with the branch names mapping
# branch_names = sales_branch[['SALES_BRANCH_CODE', 'CITY', 'REGION']]
# branch_names = branch_names.drop_duplicates()
# branch_names['CITY']

plt.scatter(table["CITY"].astype(str), table["REGION"].astype(str), color="blue")
plt.xticks(rotation=90)
plt.show()

# %% [markdown]
# De centra van de clusters worden geëxtraheerd en weergegeven.

# %%
df.groupby('Centrum', as_index = False)['Centrum'].count()

# %% [markdown]
# ## Clusteringmodel bouwen met meer dan 2 dimensies (alle kolommen uit de dataset)
# 
# Nu gaan we hetzelfde doen als hierboven, maar dan met alle kolommen uit de dataset. En misschien ook verder testen met verschillende k's.

# %%
kmeans = KMeans(n_clusters = 5, random_state = 42)
prediction_results = kmeans.fit_predict(df)
prediction_results

# %%
df['Centrum'] = prediction_results
df

# %%
df.groupby('Centrum', as_index = False)['Centrum'].count()

# %% [markdown]
# ## Evaluatie van de clustering

# %% [markdown]
# Hier gaan we de inter- en intraclusterafstand berekenen voor verschillende k's.

# %%
from typing import Literal, Any
intercluster_distance: Literal[0] = 0
intracluster_distance: Literal[0] = 0

common_columns: Any = df.columns.intersection(kmeans_centra.columns)

for centrumindex, _ in kmeans_centra[common_columns].iterrows():
    for src_index, _ in df[common_columns].iterrows():
        if df.at[src_index, "Centrum"] == centrumindex:
            diff = df.loc[src_index, common_columns] - kmeans_centra.loc[centrumindex, common_columns]
            distance = np.linalg.norm(diff[pd.to_numeric(diff, errors="coerce").notnull()])
            intracluster_distance += distance
        else:
            diff = df.loc[src_index, common_columns] - kmeans_centra.loc[centrumindex, common_columns]
            distance = np.linalg.norm(diff[pd.to_numeric(diff, errors="coerce").notnull()])
            intercluster_distance += distance

print(f"Intercluster distance (more is better): {intercluster_distance}")
print(f"Intracluster distance (less is better): {intracluster_distance}")

# %% [markdown]
# Nu voeren we deze analyse uit om het optimale aantal clusters (k) voor K-Means-clustering te bepalen met behulp van de elleboogmethode. Hierdoor kunnen we de inter- en intraclusterafstand optimaal berekenen voor verschillende k's.

# %%
intercluster_distances: list = []
intracluster_distances: list = []
k_options = range(1, 11)

for k in k_options:
    kmeans: KMeans = KMeans(n_clusters=k, random_state=42)
    df["Centrum"] = kmeans.fit_predict(df)
    kmeans_center = pd.DataFrame(kmeans.cluster_centers_, columns=df.columns)
    
    intercluster_distance: Literal[0] = 0
    intracluster_distance: Literal[0] = 0

    common_columns: Any = df.columns.intersection(kmeans_center.columns)

    for centrumindex, _ in kmeans_center[common_columns].iterrows():
        for src_index, _ in df[common_columns].iterrows():
            if df.at[src_index, "Centrum"] == centrumindex:
                diff = df.loc[src_index, common_columns] - kmeans_center.loc[centrumindex, common_columns]
                distance = np.linalg.norm(diff[pd.to_numeric(diff, errors="coerce").notnull()])
                intracluster_distance += distance
            else:
                diff = df.loc[src_index, common_columns] - kmeans_center.loc[centrumindex, common_columns]
                distance = np.linalg.norm(diff[pd.to_numeric(diff, errors="coerce").notnull()])
                intercluster_distance += distance

    intercluster_distances.append(intercluster_distance)
    intracluster_distances.append(intracluster_distance)

plt.plot(k_options, intercluster_distances, marker="o", label="Intercluster afstand", color="g")
plt.plot(k_options, intracluster_distances, marker="x", label="Intracluster afstand", color="b")
plt.xlabel("Hoeveelheid van k-clusters")
plt.ylabel("Cluster afstand")
plt.title("Inter- en Intracluster afstand")
plt.legend()
plt.show()


