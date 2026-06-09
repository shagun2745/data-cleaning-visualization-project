import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("titanic.csv")

print("Original Shape:", df.shape)

print(df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df.drop('Cabin', axis=1, inplace=True)

df.drop_duplicates(inplace=True)

df.to_csv("cleaned_titanic.csv", index=False)

plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title("Survival Distribution")
plt.savefig("survival_distribution.png")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=20)
plt.title("Age Distribution")
plt.savefig("age_distribution.png")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='Sex', data=df)
plt.title("Gender Distribution")
plt.savefig("gender_distribution.png")
plt.show()

print("Project Completed!")