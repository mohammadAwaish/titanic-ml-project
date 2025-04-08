# titanic-ml-project
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
df.head()

df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df = df.fillna(df.median())

sns.countplot(x='Survived', data=df)
sns.boxplot(x='Survived', y='Age', data=df)
sns.heatmap(df.corr(), annot=True)
plt.show()

X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

new_passenger = np.array([[1000, 3, 0, 22.0, 1, 0, 7.25, 0]])

prediction = model.predict(new_passenger)
print("Survived" if prediction[0] == 1 else "Did not survive")
