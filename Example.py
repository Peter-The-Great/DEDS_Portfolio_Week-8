import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import sqlite3

forecast = pd.read_csv('data/GO_SALES_PRODUCT_FORECASTData.csv', sep=',')
inventory = pd.read_csv('data/GO_SALES_INVENTORY_LEVELSData.csv', sep=',')

conn = sqlite3.connect('data/go_sales.sqlite')
details = pd.read_sql(con=conn, sql='SELECT * FROM order_details')
header = pd.read_sql(con=conn, sql='SELECT * FROM order_header')
product = pd.read_sql_query("SELECT * FROM product;", conn)
details = details.drop(columns=['TRIAL879'])
header = header.drop(columns=['TRIAL885'])

order = pd.merge(header, details, on='ORDER_NUMBER')
order

order['YEAR'] = order['ORDER_DATE'].astype(str).str.split("-").str.get(0)
order['MONTH'] = order['ORDER_DATE'].astype(str).str.split("-").str.get(1)
order['MONTH'] = order['MONTH'].str.replace("0","")

order = order[['PRODUCT_NUMBER', 'QUANTITY', 'YEAR', 'MONTH']]

bad_data = order[order.isna().any(axis=1) | order.isnull().any(axis=1)]
if len(bad_data) > 0:
    raise ValueError("Ongeldige waardes in brondata van orders")

order = order.groupby(['PRODUCT_NUMBER', 'YEAR', 'MONTH'])
order = order.aggregate('sum').reset_index()

inventory = inventory.rename(columns={'INVENTORY_MONTH':'MONTH','INVENTORY_YEAR':'YEAR'})
df = pd.merge(forecast, inventory, on=['MONTH', 'YEAR', 'PRODUCT_NUMBER'])


bad_data = df[df.isna().any(axis=1) | df.isnull().any(axis=1)]
if len(bad_data) > 0:
    raise ValueError("Ongeldige waardes in brondata van orders")

df['PRODUCT_NUMBER'] = df['PRODUCT_NUMBER'].astype(int)
df['MONTH'] = df['MONTH'].astype(int)
df['YEAR'] = df['YEAR'].astype(int)
order['PRODUCT_NUMBER'] = order['PRODUCT_NUMBER'].astype(int)
order['MONTH'] = order['MONTH'].astype(int)
order['YEAR'] = order['YEAR'].astype(int)

df = pd.merge(order, df, on=['MONTH', 'YEAR', 'PRODUCT_NUMBER'])

bad_data = df[df.isna().any(axis=1) | df.isnull().any(axis=1)]
if len(bad_data) > 0:
    raise ValueError("Ongeldige waardes in brondata van orders")

product['PRODUCT_NUMBER'] = product['PRODUCT_NUMBER'].astype(int)
product['PRODUCTION_COST'] = product['PRODUCTION_COST'].astype(float)
product['MARGIN'] = product['MARGIN'].astype(float)
product = product[['PRODUCT_NUMBER', 'PRODUCTION_COST', 'PRODUCT_TYPE_CODE', 'MARGIN']]

df = pd.merge(product, df, on='PRODUCT_NUMBER')

bad_data = df[df.isna().any(axis=1) | df.isnull().any(axis=1)]
if len(bad_data) > 0:
    raise ValueError("Ongeldige waardes in brondata van orders")

df

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

x = df.drop('quantity', axis=1)
y = df.loc[:,['quantity']]
x

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

reg_model = linear_model.LinearRegression()
reg_model = LinearRegression().fit(x_train,y_train)
reg_model

y_pred = reg_model.predict(x_test)
y_pred

prediction_df = pd.DataFrame(y_pred)
prediction_df = prediction_df.rename(columns={0: 'predicted_quantity'})
prediction_df

y_test_prediction_merge = pd.concat([y_test.reset_index()['quantity'], prediction_df], axis=1)
y_test_prediction_merge.loc[y_test_prediction_merge['predicted_quantity'].notna(),:]

plt.scatter(y_test_prediction_merge['quantity'], y_test_prediction_merge['predicted_quantity'])
plt.xlabel('quantity')
plt.ylabel('predicted_quantity')
plt.show()

mean_squared_error(y_test_prediction_merge['quantity'], y_test_prediction_merge['predicted_quantity'])

mean_absolute_error(y_test_prediction_merge['quantity'], y_test_prediction_merge['predicted_quantity'])