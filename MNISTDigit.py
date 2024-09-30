import pandas as pd
import numpy as np

np.random.seed(1212)

df_train = pd.read_csv('train.csv/train.csv')
df_test = pd.read_csv('test.csv/test.csv')

df_train.head() # 784 features, 1 label




df_label = df_train.iloc[:, 0]
df_features = df_train.iloc[:, 1:785]


X_test = df_test.iloc[:, 0:784]

# print(X_test.shape)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_cv, y_train, y_cv = train_test_split(df_features, df_label, test_size = 0.2, random_state = 1212)

X_train = X_train.values.reshape(33600, 784) #(33600, 784)
X_cv = X_cv.values.reshape(8400, 784) #(8400, 784)

X_test = X_test.values.reshape(28000, 784)

model = RandomForestClassifier()

# Train the model on the training set
model.fit(X_train, y_train)

# Perform cross-validation on the training set
cv_scores = cross_val_score(model, X_train, y_train, cv=5) 



print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())

# Evaluate on the test set
predictions = model.predict(X_test)

df=pd.DataFrame(predictions)
print(type(df))
df.to_csv(predictions.csv,index=True)