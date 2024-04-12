# %% [markdown]
# # PR 8.1 - Regressiemodellen in Machine Learning
# 
# Van Pjotr en Sennen
# 
# De opdracht van deze week is:
# 
# Great Outdoors wil graag weten hoeveel zij gaat verkopen op basis van een set onafhankelijke variabelen. Daarom wil zij een model trainen op basis van reeds bekende data, zodat deze volgend jaar in gebruik kan worden genomen. Je doet dus het volgende met de reeds bekende data:
# - Bedenk met welke onafhankelijke variabelen, die ook uit meerdere databasetabellen kunnen komen, dit naar verwachting het beste voorspeld kan worden en zet deze samen met de afhankelijke variabele in één DataFrame.
# - Pas waar nodig Dummy Encoding toe.
# - Snijd dit DataFrame horizontaal en verticaal op de juiste manier.
# - Train het regressiemodel.
# - Evalueer de performance van je getrainde regressiemodel.
# 
# #### Wat doen we ermee?
# We kunnen met dit model voorspellen hoeveel we gaan verkopen op basis van de onafhankelijke variabelen. Dit kan ons helpen om in te schatten hoeveel we moeten inkopen en hoeveel we moeten produceren. Dit kunnen we bijvoorbeeld gebruiken voor:
# - Globaal productverkoop
# - Verkoop per productlinie
# - Verkoop per individueel product

# %% [markdown]
# Hieronder zullen we de libaries importeren die we nodig hebben voor de opdrachten.

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import sqlite3

# %% [markdown]
# Om te beginnen maken we een connectie naar alle belangrijke data die wij nodig hebben. Zoals inventory, forecast, details en header.

# %%
# Deze had ik niet meer nodig.
forecast = pd.read_csv('data/GO_SALES_PRODUCT_FORECASTData.csv', sep=',')
inventory = pd.read_csv('data/GO_SALES_INVENTORY_LEVELSData.csv', sep=',')

# Verbind met de database en haal de tabellen op.
conn = sqlite3.connect('data/go_sales.sqlite')
details = pd.read_sql(con=conn, sql='SELECT * FROM order_details')
header = pd.read_sql(con=conn, sql='SELECT * FROM order_header')
product = pd.read_sql_query("SELECT * FROM product;", conn)

# Drop de kolommen die niet nodig zijn en irrelevant.
details = details[details.columns.drop(list(details.filter(regex="TRIAL")))]
header = header[header.columns.drop(list(header.filter(regex="TRIAL")))]

# %% [markdown]
# Hierna mergen we de order details en header met elkaar. Dit doen we op basis van de order id. We mergen de inventory met de order details en header op basis van de product id. We mergen de forecast met de inventory op basis van de product id.

# %%
order = pd.merge(header, details, on='ORDER_NUMBER')
order

# %% [markdown]
# Hier pakken we de kolommen moeten order date converteren.

# %%
# Selecteren van onafhankelijke variabelen en afhankelijke variabele
order['YEAR'] = order['ORDER_DATE'].astype(str).str.split("-").str.get(0)
order['MONTH'] = order['ORDER_DATE'].astype(str).str.split("-").str.get(1)
order['MONTH'] = order['MONTH'].str.replace("0","")

# Laat onnodige kolommen vallen
order = order[['PRODUCT_NUMBER', 'QUANTITY', 'YEAR', 'MONTH']]

# %% [markdown]
# In dit stukje code wordt er gecontroleerd op ongeldige waarden in onze order dataset. Als er een ongeldige waarde is, dam krijgen we een error.

# %%
# Bevestig dat er geen null of NaN waardes in de data zitten
bad_data = order[order.isna().any(axis=1) | order.isnull().any(axis=1)]
if len(bad_data) > 0:
    raise ValueError("Ongeldige waardes in brondata van orders")

order = order.groupby(['PRODUCT_NUMBER', 'YEAR', 'MONTH'])
order = order.aggregate('sum').reset_index()

# %% [markdown]
# Hier voeren we een aantal data-verwerkingsstappen uit met behulp van de pandas-bibliotheek.

# %%
# Samenvoegen van forecast en inventory data
inventory = inventory.rename(columns={'INVENTORY_MONTH':'MONTH','INVENTORY_YEAR':'YEAR'})
df = pd.merge(forecast, inventory, on=['MONTH', 'YEAR', 'PRODUCT_NUMBER'])

# Bevestig dat er geen null of NaN waardes in de data zitten
bad_data = df[df.isna().any(axis=1) | df.isnull().any(axis=1)]
if len(bad_data) > 0:
    raise ValueError("Ongeldige waardes in brondata van orders")

# Selecteren van onafhankelijke variabelen en afhankelijke variabele
# Product number, quantity, year, month, forecast, inventory
df['PRODUCT_NUMBER'] = df['PRODUCT_NUMBER'].astype(int)
df['MONTH'] = df['MONTH'].astype(int)
df['YEAR'] = df['YEAR'].astype(int)
order['PRODUCT_NUMBER'] = order['PRODUCT_NUMBER'].astype(int)
order['MONTH'] = order['MONTH'].astype(int)
order['YEAR'] = order['YEAR'].astype(int)

df = pd.merge(order, df, on=['MONTH', 'YEAR', 'PRODUCT_NUMBER'])

# Bevestig dat er geen null of NaN waardes in de data zitten
bad_data = df[df.isna().any(axis=1) | df.isnull().any(axis=1)]
if len(bad_data) > 0:
    raise ValueError("Ongeldige waardes in brondata van orders")

# %% [markdown]
# Hier pakken we de product uit en zorgen we ervoor dat de product id en de product name in een dataframe komen te staan in de juiste type.

# %%
product['PRODUCT_NUMBER'] = product['PRODUCT_NUMBER'].astype(int)
product['PRODUCTION_COST'] = product['PRODUCTION_COST'].astype(float)
product['MARGIN'] = product['MARGIN'].astype(float)
product = product[['PRODUCT_NUMBER', 'PRODUCTION_COST', 'PRODUCT_TYPE_CODE', 'MARGIN']]

df = pd.merge(product, df, on='PRODUCT_NUMBER')

# Bevestig dat er geen null of NaN waardes in de data zitten
bad_data = df[df.isna().any(axis=1) | df.isnull().any(axis=1)]
if len(bad_data) > 0:
    raise ValueError("Ongeldige waardes in brondata van orders")

df

# %% [markdown]
# Hier wordt een dataset voorbereid voor verdere analyse, waar we de order details en header samenvoegen. Daarna maken we ook nog dummie data aan via maand en product type.

# %%
df['MONTH'] = df['MONTH'].astype(str)
month_dummies = pd.get_dummies(df.loc[:,['MONTH']])

df = pd.concat([df, month_dummies], axis=1)
df.drop(['MONTH'], axis=1)


df['PRODUCT_TYPE_CODE'] = df['PRODUCT_TYPE_CODE'].astype(str)
product_line_dummies = pd.get_dummies(df.loc[:,['PRODUCT_TYPE_CODE']])

df = pd.concat([df, product_line_dummies], axis=1)
df.drop(['PRODUCT_TYPE_CODE'], axis=1)

df.drop(['PRODUCT_NUMBER'], axis=1)

df = df.rename(columns=str.lower)

# %% [markdown]
# Nu gaan we de tabellen vertical en horizontaal snijden. We maken een tabel met de onafhankelijke variabelen en een tabel met de afhankelijke variabelen. de afhankelijke variabele is de order quantity en de onafhankelijke variabelen zijn de product type en de maand.

# %%
x = df.drop('quantity', axis=1)
y = df.loc[:,['quantity']]
x

# %% [markdown]
# Nu gaan we de data trainenen en testen. We maken een train en test set aan. We trainen de data en voorspellen de data. Voor beide x en y.

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

# %% [markdown]
# Met de training data en de test data gesplitst en getrained, gaan we nu beginnnen aan het maken van een model.

# %%
reg_model = linear_model.LinearRegression()
reg_model = LinearRegression().fit(x_train,y_train)
reg_model

# %% [markdown]
# Hier gaan we met het gegeven regressiemodel de data voorspellen en kijken hoe goed het model is. Dat zullen we later laten zien.

# %%
y_pred = reg_model.predict(x_test)
y_pred

# %% [markdown]
# De voorspellingen zijn gedaan en worden nu in een dataframe omgezet, we veranderen de naam van een van de kolomen naar 'Predicted_Quantity'.

# %%
prediction_df = pd.DataFrame(y_pred)
prediction_df = prediction_df.rename(columns={0: 'predicted_quantity'})
prediction_df

# %% [markdown]
# Hierin worden de voorspellingen met de test data toegevoegd en wordt daarna gecheckt of de predicted_quantity niet null of NaN is.

# %%
y_test_prediction_merge = pd.concat([y_test.reset_index()['quantity'], prediction_df], axis=1)
y_test_prediction_merge.loc[y_test_prediction_merge['predicted_quantity'].notna(),:]

# %% [markdown]
# Nu wordt het regressie model in beeld gebracht. Nu zal via de grafiek te zien zijn hoe goed het model is. Dit doen we door gebruik te maken van pyplot en seaborn.

# %%
colors = np.interp(y_test_prediction_merge["quantity"], (y_test_prediction_merge["quantity"].min(), y_test_prediction_merge["quantity"].max()), (0, 66))
plt.scatter(y_test_prediction_merge['quantity'], y_test_prediction_merge['predicted_quantity'], s=5, c=colors, cmap="turbo", alpha=0.8)
plt.xlabel('Hoeveelheid')
plt.ylabel('Verwachte hoeveelheid')
plt.title('Hoeveelheid vs Verwachte hoeveelheid')
plt.colorbar(orientation="vertical", label="Percentage Sale", extend="both")
line_size = np.linspace(y_test_prediction_merge["quantity"].min(), y_test_prediction_merge["quantity"].max(), 100)
plt.arrow(line_size.min(), line_size.min(), line_size.max(), line_size.max(), color='red', linestyle='-')
plt.show()

# %% [markdown]
# Met deze functie gaan we de gemiddelde kwadratische fout berekenen.

# %%
mean_squared_error(y_test_prediction_merge['quantity'], y_test_prediction_merge['predicted_quantity'])

# %% [markdown]
# Hieronder doen we ook nog het absolute gemiddelde fout berekenen.

# %%
mean_absolute_error(y_test_prediction_merge['quantity'], y_test_prediction_merge['predicted_quantity'])

# %% [markdown]
# Hier kunnen we het model scoren op hoe goed het model is. Dit doen we door de reg_model te berekenen. Dus daarna de accuracy score te berekenen.

# %%
reg_model.score(x_test, y_test)


