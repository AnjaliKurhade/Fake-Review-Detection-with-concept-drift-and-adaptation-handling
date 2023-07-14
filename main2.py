import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\Akshay\Desktop\flipkart reviews\Training\reviews.csv")
df1 = df.iloc[:, [0, 1, 3, 2]].copy() 
df1['label'].replace(['CG', 'OR'], [0, 1], inplace=True)

xfeatures = df['text_']
yfeatures = df['label']
x_train, x_test, y_train, y_test = train_test_split(xfeatures, yfeatures, test_size=0.2)
 
pipe = Pipeline([('tfidf', TfidfVectorizer()), ('lr', LogisticRegression(max_iter=5000))])

parameters = {'tfidf__max_df': [0.5, 1.0], 'tfidf__ngram_range': [(1, 1), (1, 2)],
              'lr__C': [0.1, 1, 10], 'lr__penalty': ['l1', 'l2']}

grid_search = GridSearchCV(pipe, parameters, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)

print("Best parameters: ", grid_search.best_params_)


y_predict = grid_search.predict(x_test)

cr = classification_report(y_test,y_predict)
print(cr)
acc = accuracy_score(y_test,y_predict)
print(acc)
