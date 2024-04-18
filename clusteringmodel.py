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

orders_header = pd.read_sql_query("SELECT * FROM order_header;", conn_sales)
order_details = pd.read_sql_query("SELECT * FROM order_details;", conn_sales)

# %% [markdown]
# Hier gaan we de data van de great outdoors inlezen en bepaalde data dropen.

# %%
# (Helaas Verouderd)
# country = country1[['CURRENCY_NAME', 'LANGUAGE']]
# country = pd.concat([country2, country], axis = 1)
# sales_branch.drop('TRIAL633', axis=1, inplace=True)
# country.drop('TRIAL219', axis=1, inplace=True)
# territory.drop('TRIAL222', axis=1, inplace=True)
orders_header.drop('TRIAL885', axis=1, inplace=True)
order_details.drop('TRIAL879', axis=1, inplace=True)
sales_branch.drop('TRIAL633', axis=1, inplace=True)
sales_branch.drop(["ADDRESS1", "ADDRESS2", "POSTAL_ZONE"], axis=1)

# %% [markdown]
# Nu gaan we data mergen. Orders worden samengevoegd met details om een compleet beeld van de bestellingen te krijgen.
# 
# Productinformatie wordt geladen en samengevoegd om een lookup-tabel te maken voor productnummers naar productcategorieën.
# 
# Verkoopgegevens worden voorbereid, zoals de omzet- en afzetgegevens per verkoopfiliaal en de verhouding van de verkoop per productlijn.
# 
# Bij de productlijn wordt ook dummy encoding toegepast.
# 
# We zullen later de branches dummies geven.

# %%
order_full = pd.merge(order_details, orders_header, on='ORDER_NUMBER')
order_full['UNIT_SALE_PRICE'] = order_full['UNIT_SALE_PRICE'].astype(float)
order_full['UNIT_PRICE'] = order_full['UNIT_PRICE'].astype(float)
order_full['UNIT_COST'] = order_full['UNIT_COST'].astype(float)

# Lees productstabellen
product = pd.read_sql_query("SELECT * FROM product;", conn_sales)
product_type = pd.read_sql_query("SELECT * FROM product_type;", conn_sales)
product_line = pd.read_sql_query("SELECT * FROM product_line;", conn_sales)

# Maak lookuptabel voor productnumber -> naam productscategorie
product_line_lookup = pd.merge(product, product_type, on='PRODUCT_TYPE_CODE')
product_line_lookup = pd.merge(product_line_lookup, product_line, on='PRODUCT_LINE_CODE')
product_line_lookup = product_line_lookup.loc[:,['PRODUCT_NUMBER', 'PRODUCT_LINE_EN']]

# Maak dummies van product_line
product_line_dummies = pd.get_dummies(product_line_lookup['PRODUCT_LINE_EN'])

product_line_lookup = product_line_lookup.drop(['PRODUCT_LINE_EN'], axis=1)
product_line_lookup = pd.concat([product_line_dummies, product_line_lookup], axis = 1)

# Omzet/Afzetsdata
sales_profit_data = order_full.groupby('SALES_BRANCH_CODE').aggregate('sum') #.reset_index()
sales_profit_data = sales_profit_data[['QUANTITY', 'UNIT_COST', 'UNIT_PRICE', 'UNIT_SALE_PRICE']]

# Productsverkoopdata (Verhouding van verkoop per productlijn)
sales_product_data = pd.merge(order_full, product_line_lookup, on='PRODUCT_NUMBER')
sales_product_data = sales_product_data[['SALES_BRANCH_CODE', 'Camping Equipment', 'Golf Equipment', 'Mountaineering Equipment', 'Outdoor Protection' , 'Personal Accessories']]
sales_product_data = sales_product_data.groupby('SALES_BRANCH_CODE').aggregate(np.mean)

# %% [markdown]
# ## Clusteringmodel bouwen met 2 dimensies
# 
# Eerst wordt een 2D-dataset gemaakt met hoeveelheid en eenheidsprijs van de verkochte producten.
# 
# Waar we ook dummy encoding toepassen.

# %%
table = sales_profit_data[['QUANTITY', 'UNIT_SALE_PRICE']]
df = pd.get_dummies(table)  # Dummy encoding

table = pd.concat([table, df], axis=1)
table

# %% [markdown]
# Een k-means clusteringmodel wordt geïnstantieerd met het gewenste aantal clusters (3 of 4 in het eerste geval, met een willenkeurigheid van 42) en toegepast op de 2D-dataset.

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
branch_names = sales_branch[['SALES_BRANCH_CODE', 'CITY', 'REGION']]
branch_names = branch_names.drop_duplicates()
branch_names['CITY']

plt.scatter(branch_names['CITY'].astype(str), branch_names['REGION'].astype(str))
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
prediction_results = kmeans.fit_predict(sales_product_data)
prediction_results

# %%
sales_product_data['Centrum'] = prediction_results
sales_product_data

# %%
sales_product_data.groupby('Centrum', as_index = False)['Centrum'].count()

# %% [markdown]
# Hier visualiseren we de gemiddelde verkoop van verschillende productcategorieën binnen elk cluster.

# %%
cluster_sales_mean = sales_product_data.groupby('Centrum').mean()

# Visualisatie van gemiddelde verkoop per cluster
plt.figure(figsize=(10, 6))
plt.bar(cluster_sales_mean.index, cluster_sales_mean['Camping Equipment'], label='Camping Equipment')
plt.bar(cluster_sales_mean.index, cluster_sales_mean['Golf Equipment'], bottom=cluster_sales_mean['Camping Equipment'], label='Golf Equipment')
plt.bar(cluster_sales_mean.index, cluster_sales_mean['Mountaineering Equipment'], bottom=cluster_sales_mean['Camping Equipment']+cluster_sales_mean['Golf Equipment'], label='Mountaineering Equipment')
plt.bar(cluster_sales_mean.index, cluster_sales_mean['Outdoor Protection'], bottom=cluster_sales_mean['Camping Equipment']+cluster_sales_mean['Golf Equipment']+cluster_sales_mean['Mountaineering Equipment'], label='Outdoor Protection')
plt.bar(cluster_sales_mean.index, cluster_sales_mean['Personal Accessories'], bottom=cluster_sales_mean['Camping Equipment']+cluster_sales_mean['Golf Equipment']+cluster_sales_mean['Mountaineering Equipment']+cluster_sales_mean['Outdoor Protection'], label='Personal Accessories')
plt.xlabel('Cluster')
plt.ylabel('Gemiddelde verkoop')
plt.title('Gemiddelde verkoop per cluster per productcategorie')
plt.legend()
plt.show()

# %% [markdown]
# ## Evaluatie van de clustering

# %% [markdown]
# Hier gaan we de inter- en intraclusterafstand berekenen voor verschillende k's.

# %%
# Lijst om interclusterafstanden op te slaan
intercluster_distances = []

# Lijst om intraclusterafstanden op te slaan
intracluster_distances = []

# Lijst van verschillende k's om te evalueren
k_values = [2, 3, 4, 5, 6]

for k in k_values:
    # K-means clusteringmodel toepassen
    kmeans = KMeans(n_clusters=k, random_state=42)
    prediction_results = kmeans.fit_predict(sales_product_data)
    
    # Interclusterafstand
    intercluster_distance = np.sum(np.min(kmeans.transform(sales_product_data), axis=1)) / sales_product_data.shape[0]
    intercluster_distances.append(intercluster_distance)
    
    # Intraclusterafstand
    intracluster_distance = kmeans.inertia_ / sales_product_data.shape[0]
    intracluster_distances.append(intracluster_distance)

# Visualisatie van inter- en intraclusterafstanden voor verschillende k's
plt.plot(k_values, intercluster_distances, marker='o', label='Intercluster distance')
plt.plot(k_values, intracluster_distances, marker='x', label='Intracluster distance')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Distance')
plt.title('Inter- and Intracluster distances for different k values')
plt.legend()
plt.show()


