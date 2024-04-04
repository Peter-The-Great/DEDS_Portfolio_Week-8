# %% [markdown]
# # PR8-2: Classificatiemodellen in Machine Learning.
# 
# Great Outdoors wil graag weten wat de retourredenen gaan zijn op basis van een set onafhankelijke variabelen. Daarom wil zij een model trainen op basis van reeds bekende data, zodat deze volgend jaar in gebruik kan worden genomen. Let op: de retourreden kan ook "n.v.t." zijn, niet elke order wordt namelijk geretourneerd; je zult dit moeten aanpakken door een join tussen "returned_item" en "order_details". Je doet dus het volgende met de reeds bekende data:
# - Bedenk met welke onafhankelijke variabelen dit naar verwachting het beste voorspeld kan worden en zet deze samen met de afhankelijke variabele in één DataFrame.
# - Pas waar nodig Dummy Encoding toe.
# - Snijd dit DataFrame horizontaal en verticaal op de juiste manier.
# - Train het classificatiemodel.
# - Evalueer de performance van je getrainde classificatiemodel a.d.h.v. een confusion matrix.

# %% [markdown]
# Hier importeren we alle benodigde libraries en lezen we de data in.

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt

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


