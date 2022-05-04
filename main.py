# make necessary imports

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('fake_or_real_news.csv')
"""
df.head()##此方法用于返回数据帧或序列的前n行(默认值为5)。
print(df.head())
def create_distribution(dataFile):
    return sns.countplot(x='label', data=dataFile, palette='hls')

create_distribution(df)
plt.show( )#sns.countplot() 用于类别特征的频数条形图
"""
"""
def data_qualityCheck():
    print("Checking data qualitites...")
    df.isnull().sum()
    df.info()
    print("check finished.")
data_qualityCheck()
"""
# Separate the labels and set up training and test datasets

# Get the labels
x=df['text']
y = df['label']
"""
y.head()
print(y.head())
"""
#Split the dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.23, random_state=53)
from sklearn.feature_extraction.text import CountVectorizer #CountVectorizer会将文本中的词语转换为词频矩阵
from sklearn.feature_extraction.text import TfidfVectorizer #TfidfVectorizer在构建词汇表（特征词表）时考虑了词语文档频次
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
#print(count_vectorizer)
#vocab size
"""
print(count_train.shape)
print(count_vectorizer.vocabulary_)
"""
count_test = count_vectorizer.transform(X_test)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
#print(tfidf_train.shape)
#get train data feature names
#print(tfidf_train)
#print(tfidf_train.A[0:10])
tfidf_test = tfidf_vectorizer.transform(X_test)


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import KFold
import itertools
## Pipeline is all about very first I have to apply tfidf_vectorizer, then I have to apply MultinomialNB on data

nb_pipeline = Pipeline([
        ('NBTV',tfidf_vectorizer),
        ('nb_clf',MultinomialNB())])
nb_pipeline.fit(X_train,y_train)
predicted_nbt = nb_pipeline.predict(X_test)
print(predicted_nbt)
score = metrics.accuracy_score(y_test, predicted_nbt)
print('Accuracy:{}'.format(round(score*100,2)))
confusion_matrix(y_test,predicted_nbt)
nbc_pipeline = Pipeline([
        ('NBCV',count_vectorizer),
        ('nb_clf',MultinomialNB())])
nbc_pipeline.fit(X_train,y_train)
predicted_nbc = nbc_pipeline.predict(X_test)
score = metrics.accuracy_score(y_test, predicted_nbc)
print('Accuracy:{}'.format(round(score*100,2)))
confusion_matrix(y_test,predicted_nbc)
linear_clf = Pipeline([
        ('linear',tfidf_vectorizer),
        ('pa_clf',PassiveAggressiveClassifier(max_iter=50))])
linear_clf.fit(X_train,y_train)
pred = linear_clf.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print('Accuracy:{}'.format(round(score*100,2)))
confusion_matrix(y_test,pred)
print(metrics.classification_report(y_test, pred))
print((y_test, pred))