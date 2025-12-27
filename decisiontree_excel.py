import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel("C:/Users/ADMIN/OneDrive/Desktop/bank-full.xlsx")
print(df)

le = LabelEncoder()

for column in df.columns:
    df[column] = le.fit_transform(df[column])

print(df)

X = df.iloc[:, :-1]   
y = df.iloc[:, -1]

model = DecisionTreeClassifier(max_depth=5)
model.fit(X, y)

plt.figure(figsize=(12, 6), dpi=150)

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    fontsize=5
)

plt.show()

