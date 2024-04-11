# %% [markdown]
# # Week 9-2: Unsupervised Machine Learning met Apriori
# Van Pjotr en Sennen.
# 
# Deze week hebben we het over ongesuperviseerde machine learning. Dit is een vorm van machine learning waarbij we geen gelabelde data hebben. Dit betekent dat we geen data hebben waarbij we weten wat de juiste output is. In plaats daarvan gaan we op zoek naar patronen in de data. Dit kan bijvoorbeeld zijn dat we clusters van data vinden, of dat we de data kunnen reduceren naar een kleinere dimensie.
# 
# Deze keer hebben we het over Frequent Itemsets met het A-Priori-algoritme. Dit is een algoritme dat gebruikt wordt om patronen te vinden in data. Het wordt vaak gebruikt in de retailsector om te kijken welke producten vaak samen worden gekocht. Dit kan bijvoorbeeld gebruikt worden om producten in een winkel anders te plaatsen, zodat producten die vaak samen worden gekocht ook dicht bij elkaar staan.
# 
# 
# ## De Opdracht
# PR9-2: Great Outdoors wil graag weten welke producten vaak samen gekocht worden door klanten, door het bouwen van Frequent Itemsets met A-Priori-algoritme. Tip: merge eerst de tabellen 'product' en 'order_details' om een juiste tabel met brongegevens te krijgen waarop je het algoritme kan toepassen. 
# - Pas waar nodig Dummy Encoding toe.
# - Train het initiële algoritme.
# - Experimenteer met meerdere support & confidence thresholds.
# - De volgende webpagina's kun je als inspiratie gebruiken. Zij bevatten "codekapstokken" die uitleggen hoe je Frequent Itemsets moet maken en daarmee antwoord kunt geven op de vraag welke producten vaak samen gekocht worden.
#     - [geeksforgeeks](https://www.geeksforgeeks.org/implementing-apriori-algorithm-in-python/)
#     - [towardsdatascience](https://towardsdatascience.com/apriori-association-rule-mining-explanation-and-python-implementation-290b42afdfc6)

# %% [markdown]
# Eerst zorgen we ervoor dat we alle libraries hebben geimporteerd die we nodig hebben.

# %%
import numpy as np 
import pandas as pd 
from mlxtend.frequent_patterns import apriori, association_rules
import sqlite3
import apriori_python as ap
import matplotlib.pyplot as plt

# %% [markdown]
# Hierna gaan we onze data pakken die we nodig hebben. En de tabelen samenvoegen die we nodig hebben.

# %%
sales_conn = sqlite3.connect('data/go_sales.sqlite')

# Pak het Order Nummer en Product Nummer uit de order_details tabel.
order_details = pd.read_sql_query('SELECT * FROM order_details', sales_conn)
order_details = order_details[['ORDER_NUMBER', 'PRODUCT_NUMBER']]

#
products = pd.read_sql_query('SELECT * FROM product', sales_conn)
products = products[['PRODUCT_NUMBER', 'PRODUCT_NAME']]
dummy_products = pd.get_dummies(products['PRODUCT_NAME'])
products = products.drop('PRODUCT_NAME', axis=1)
products = pd.concat([products, dummy_products], axis=1)
products

# %% [markdown]
# Hier gaan we de data mergen en de data met product id's en product namen samenvoegen. Die product namen zijn allemaal one hot encoded.

# %%
# Merge de order_details en products tabellen op PRODUCT_NUMBER
merged_data = pd.merge(order_details, products, on='PRODUCT_NUMBER', how='inner')

# Verwijder de kolommen die niet nodig zijn voor het algoritme
merged_data = merged_data.drop(['PRODUCT_NUMBER'], axis=1)
merged_data = merged_data.groupby('ORDER_NUMBER').aggregate('max')
merged_data
# Train het a-priori algoritme op de samengevoegde gegevens
frequent_itemsets = apriori(merged_data, min_support=0.01, verbose=True, use_colnames=True)
frequent_itemsets

# # Sorteer de frequent_itemsets op basis van de support-waarden
sorted_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

# # Plot de frequentie van de itemsets
plt.bar(x = range(0, 30), height = sorted_itemsets['support'][0:30], tick_label = sorted_itemsets['itemsets'][0:30])
plt.xticks(rotation=90)
plt.show()


# %%
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])
rules


