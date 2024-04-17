# %% [markdown]
# # PR8-2: Classificatiemodellen in Machine Learning.
# Van Pjotr en Sennen
# 
# Hieronder is een voorbeeld uit het hoorcollege hoe we de classificatiemodellen in Machine Learning kunnen toepassen. We gaan een classificatiemodel maken met de Titanic dataset. Waar de volgende stappen in voorkomen:

# %% [markdown]
# Hier importeren we alle benodigde libraries en lezen we de data in.

# %%
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree 
from sklearn import metrics
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Laad de connnectie met de database voor later.
conn = sqlite3.connect('data/go_sales.sqlite')

# %% [markdown]
# Eerst gaan we alle data inlezen die nodig zijn voor deze opdracht.

# %%
df = pd.read_csv('data/titanic.csv')
df

# %% [markdown]
# Hierna pakken we de data die we echt nodig hebben voor dit experiment.

# %%
df = df.loc[:, ['pclass', 'sex', 'age', 'survived']]
df

# %% [markdown]
# Met de data die we nodig hebben in hand gaan we nu beginnen aan het one-hot encoden van de data.

# %%
df['pclass'] = df['pclass'].astype(str)
df.dtypes

# %% [markdown]
# Hierna pakken we de dummie data die we dan gaan gebruiken voor het trainen van het model.

# %%
dummies_dataframe = pd.get_dummies(df.loc[:, ['sex','pclass']])
dummies_dataframe

# %% [markdown]
# Nu moeten de de dummie data nog toevoegen aan de orginele dataframe.
# Dan droppen wij de sex kolom en dan pakken we de rest van de tabel mee.

# %%
df = pd.concat([df, dummies_dataframe], axis=1)
df = df.drop(['sex'], axis=1)
df = df.loc[:, ['pclass_1','pclass_2','pclass_3','sex_female','sex_male','age', 'survived']]
df

# %% [markdown]
# Hier gaan we de data opsplitsen in een train en test set. Dus verticaal opsplitsen in x en y.

# %%
x = df.drop('survived', axis=1)
y = df['survived']
x

# %% [markdown]
# Nu gaan we de data splitsen in een train en test set. Dus horizontaal opsplitsen in x en y.

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

# %% [markdown]
# ### Met max_depth=2
# Nu gaan we de decision tree classifier en daarna opbouwen en uiteindelijk evalueren.

# %%
dtree = DecisionTreeClassifier(max_depth=2)
dtree = dtree.fit(x_train, y_train)
tree.plot_tree(dtree, feature_names=x.columns)
plt.show()

# %% [markdown]
# Daarna gaan we de data voorspellen en de confusion matrix maken.

# %%
predicted_df = pd.DataFrame(dtree.predict(x_test))
predicted_df = predicted_df.rename(columns={0: 'Predicted Survived'})
model_results_frame = pd.concat([y_test.reset_index()['survived'], predicted_df], axis=1)
model_results_frame

# %%
confusion_matrix = metrics.confusion_matrix(model_results_frame['survived'], model_results_frame['Predicted Survived'])
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels=['Dood', 'Overleeft'])

cm_display.plot()
plt.show()

# %% [markdown]
# Nu moeten we nog de score berekenen van de classifier.

# %%
metrics.accuracy_score(model_results_frame['survived'], model_results_frame['Predicted Survived'])

# %% [markdown]
# ### Geen max depth
# Maar omdat we zojuist een decision tree classifier van een max depth van 2 hebben gemaakt, gaan we nu een classifier maken met geen max depth.

# %%
dtree = DecisionTreeClassifier()
dtree = dtree.fit(x_train, y_train)
tree.plot_tree(dtree, feature_names=x.columns)
plt.show()

# %%
predicted_df = pd.DataFrame(dtree.predict(x_test))
predicted_df = predicted_df.rename(columns={0: 'Predicted Survived'})
model_results_frame = pd.concat([y_test.reset_index()['survived'], predicted_df], axis=1)
model_results_frame

# %%
confusion_matrix = metrics.confusion_matrix(model_results_frame['survived'], model_results_frame['Predicted Survived'])
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels=['Dood', 'Overleeft'])

cm_display.plot()
plt.show()

# %%
metrics.accuracy_score(model_results_frame['survived'], model_results_frame['Predicted Survived'])

# %% [markdown]
# # Nu de echte opdracht
# 
# Great Outdoors wil graag weten wat de retourredenen gaan zijn op basis van een set onafhankelijke variabelen. Daarom wil zij een model trainen op basis van reeds bekende data, zodat deze volgend jaar in gebruik kan worden genomen. Let op: de retourreden kan ook "n.v.t." zijn, niet elke order wordt namelijk geretourneerd; je zult dit moeten aanpakken door een join tussen "returned_item" en "order_details". Je doet dus het volgende met de reeds bekende data:
# - Bedenk met welke onafhankelijke variabelen dit naar verwachting het beste voorspeld kan worden en zet deze samen met de afhankelijke variabele in één DataFrame.
# - Pas waar nodig Dummy Encoding toe.
# - Snijd dit DataFrame horizontaal en verticaal op de juiste manier.
# - Train het classificatiemodel.
# - Evalueer de performance van je getrainde classificatiemodel a.d.h.v. een confusion matrix.
# 
# Dit is de opdracht die we gaan uitvoeren. In dit geval gaan we de retourreden voorspellen door gebruik te maken van producten type en de prijs van het product.

# %%
order_details = pd.read_sql_query('SELECT * FROM order_details', conn)
returned_item = pd.read_sql_query('SELECT * FROM returned_item', conn)
returned_reason = pd.read_sql_query('SELECT * FROM return_reason', conn)
product = pd.read_sql_query("SELECT * FROM product", conn)

# %% [markdown]
# Tabellen gaan we nu mergen, hiervoor heb ik laatst een functie voor gemaakt om alle tabellen te mergen en kijken we alleen of de data wel van toepassing is in de tabel:

# %%
merged_table = pd.merge(order_details, returned_item, on='ORDER_DETAIL_CODE', how='outer')
merged_table = pd.merge(merged_table, returned_reason, on='RETURN_REASON_CODE', how='outer')
merged_table
filtered_table = merged_table.dropna()
filtered_table

# %% [markdown]
# Hiernaa gaan we wat data eruit halen naast de Not A Number values. Met daarbij droppen we ook de date want die is niet van toepassing.
# 
# Daarnaast moeten we ook de TRAIL kolomen verwijderen, omdat deze niet van toepassing zijn.

# %%
filtered_table = filtered_table.drop(columns=["RETURN_DATE"])
print(list(filtered_table.filter(regex='TRIAL')))
filtered_table = filtered_table[filtered_table.columns.drop(list(filtered_table.filter(regex='TRIAL')))]

# %%
selected_columns = filtered_table
selected_columns

# %% [markdown]
# Hier kijken we naar welke kolommen we nodig hebben voor de voorspelling. De return_description is waarschijnlijk de kolom die we nodig hebben. Aangezien die maar een waarde heeft van 5.s

# %%
# We moeten controleren of er kolommen zijn die categorische variabelen bevatten die moeten worden omgezet naar dummyvariabelen.
# Laten we eerst controleren welke kolommen categorisch zijn en hoeveel unieke waarden ze bevatten.
for column in selected_columns.columns:
    if selected_columns[column].dtype == 'object':
        print(f"{column}: {selected_columns[column].nunique()} unieke waarden")

# In dit geval is de kolom RETURN_DESCRIPTION_EN categorisch.
# We passen Dummy Encoding toe op deze kolom.
dummies_dataframe = pd.get_dummies(selected_columns.loc[:, ["RETURN_DESCRIPTION_EN"]])

df = pd.concat([selected_columns, dummies_dataframe], axis=1)
df

# %%
x = df.drop(columns=["RETURN_REASON_CODE", "RETURN_DESCRIPTION_EN", "RETURN_DESCRIPTION_EN_Defective product", "RETURN_DESCRIPTION_EN_Incomplete product", "RETURN_DESCRIPTION_EN_Unsatisfactory product", "RETURN_DESCRIPTION_EN_Wrong product ordered", "RETURN_DESCRIPTION_EN_Wrong product shipped"]) # Onafhankelijke variabelen
y = df[["RETURN_DESCRIPTION_EN"]] # Afhankelijke variabele
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
y

# %% [markdown]
# Nu zetten we alles in een Decision Tree Classifier en gaan we de data trainen.

# %%
dtree = DecisionTreeClassifier(max_depth=4, random_state=1)
dtree.fit(x_train, y_train)
tree.plot_tree(dtree, feature_names=x.columns, filled=True)
plt.show()
scores = cross_val_score(dtree, x_train, y_train, cv=5)

print("Gemiddelde kruisvalidatiescore:", scores.mean())
print("Standaarddeviatie van kruisvalidatiescores:", scores.std())

# %% [markdown]
# Nu gaan we voorspellingen maken en dan kijken of de data een goede score heeft. Score is 0.96, dus dat is een goede score.

# %%
predicted_df = pd.DataFrame(dtree.predict(x_test))
predicted_df = predicted_df.rename(columns={0: 'Predicted Return Reason'})
predicted_df

model_results_frame = pd.concat([y_test.reset_index()['RETURN_DESCRIPTION_EN'], predicted_df], axis=1)

print("Dit is de nauwkeurigheid van het model:", metrics.accuracy_score(model_results_frame['RETURN_DESCRIPTION_EN'], model_results_frame['Predicted Return Reason']))
model_results_frame

# %% [markdown]
# Nu gaan we de data in een confusion matrix zetten, waar we de prediction en de actual values in gaan zetten.

# %%
confusion_matrix = metrics.confusion_matrix(model_results_frame['RETURN_DESCRIPTION_EN'], model_results_frame['Predicted Return Reason'])
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels=selected_columns['RETURN_DESCRIPTION_EN'].unique())

cm_display.plot()
plt.ylabel('Werkelijke Waardes Label')
plt.xticks(rotation=45)
plt.xlabel('Verwachte Label')
plt.show()

# %% [markdown]
# Nog een keer gaan we de data scoren, die zojuist in de confusion matrix is gezet. Dan gaan we kijken naar accuracy, sensitivity en specificity.

# %%
from sklearn.metrics import classification_report

print("Dit is de accuracy van het model:", metrics.accuracy_score(model_results_frame['RETURN_DESCRIPTION_EN'], model_results_frame['Predicted Return Reason']))
print("Dit is de sensitivity van het model:", metrics.recall_score(model_results_frame['RETURN_DESCRIPTION_EN'], model_results_frame['Predicted Return Reason'], average='weighted'))
print("Dit is de specificity van het model:", metrics.precision_score(model_results_frame['RETURN_DESCRIPTION_EN'], model_results_frame['Predicted Return Reason'], average='weighted'))


