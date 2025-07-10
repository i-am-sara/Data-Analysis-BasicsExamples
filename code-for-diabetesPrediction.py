# imports
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt


# upload the csv
df = pd.read_csv("diabetes.csv")
df

#general stats
df.describe()

df.isnull().sum()

# in these case it is not necessary (we just prove that the database used is 'clean') but is better if we add a verification
df.isnull().sum()

cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] # It cannot be 0 
df[cols] = df[cols].replace(0, np.nan)
df[cols] = df[cols].fillna(df[cols].median())

# at first i want to know how many people have the diferents outcomes
sns.countplot(x='Outcome', data=df)
plt.title('Distribución de personas con/ sin diabetes')
plt.xticks([0, 1], ['No Diabetes', 'Diabetes'])
plt.show()

# with this im able to know the correlation betwean all the charactericts with the other ones
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de correlación')
plt.show()

# 
features = ['Glucose', 'BloodPressure', 'BMI', 'Age']

for feature in features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Outcome', y=feature, data=df)
    plt.title(f'{feature} según diagnóstico')
    plt.xticks([0, 1], ['No Diabetes', 'Diabetes'])
    plt.show()

# histograms
# outliers
for col in features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, y=col)
    plt.title(f'Outliers en {col}')
    plt.show()

# some relations between diferents features
sns.scatterplot(data=df, x='Glucose', y='BMI', hue='Outcome')
plt.title('Glucosa vs IMC')
plt.show()
sns.scatterplot(data=df, x='Insulin', y='Pregnancies', hue='Outcome')
plt.title('Insulin vs pregnacies')
plt.show()

# train for the machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))


