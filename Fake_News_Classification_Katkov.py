import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#utilizing panda read
df = pd.read_csv("TestFakeReal (1).csv")
print(df.head())

y = df.label
#train and test samples
X_train, X_test, y_train, y_test = train_test_split(df["text"], y, test_size=0.6, random_state=22)

#tfidf and count vectorizers
count_vectorizer = CountVectorizer(stop_words="english")
tfidf_vectorizer = TfidfVectorizer(stop_words="english")

count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)

tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

#using naive bayes to find accuracy and matrix
nb_count = MultinomialNB()
nb_count.fit(count_train, y_train)
pred_count = nb_count.predict(count_test)

nb_tfidf = MultinomialNB()
nb_tfidf.fit(tfidf_train, y_train)
pred_tfidf = nb_tfidf.predict(tfidf_test)

#accuracy score
count_score = accuracy_score(y_test, pred_count)
print('CountVectorizer accuracy:', count_score)

score = accuracy_score(y_test, pred_tfidf)
print('TFIDF accuracy score:', score)

#confusion matrix for identification
cm = confusion_matrix(y_test, pred_tfidf, labels=['FAKE', 'REAL'])
print(cm)
