from nltk.stem import PorterStemmer, WordNetLemmatizer
import matplotlib.pyplot as plt
import numpy as pd
import pandas as pd
from nltk.corpus import stopwords
from string import punctuation
import re

df_train = pd.read_csv('Training_Dataset/train.csv')
df_test = pd.read_csv('Training_Dataset/test.csv')
df_train.info()
# drop the column ID from df_train
df_train.drop('ID',axis=1,inplace=True)

# drop the column highlight from df_train
df_train.drop('highlight',axis=1,inplace=True)
# remove any rows with NAN
df_train = df_train.dropna()
df_train.info()

# drop the column ID from df_test
df_test.drop('ID',axis=1,inplace=True)
df_test.info()

#combine both the dataframes into df
df = pd.concat([df_train, df_test], ignore_index=True)
df.info()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['sentiment'],test_size=0.2,random_state=42)

lemmatizer = WordNetLemmatizer() #changes word into its original meaning
stemmer = PorterStemmer()        #removes suffix
def smooth(text):
    
    #removing urls
    text = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0–9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)', '', text, flags=re.MULTILINE)  # to remove links that start with HTTP/HTTPS in the tweet
    text = re.sub(r'[-a-zA-Z0–9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)', '', text, flags=re.MULTILINE) # to remove other url links
    
    #removing apostrophes so that contracted words don't lose meaning and saving into remove_apos
    remove_apos = [word for word in text if word not in "'`"]
    remove_apos = ''.join(remove_apos)
    
    #removing punctuatin from remove_apos and saving into text_without_punc
    text_without_punc = [word if word not in punctuation else ' ' for word in remove_apos ]
    text_without_punc = ''.join(text_without_punc)
    
    #removing stop words from text_without_punc and saving into text_without_sw
    text_without_sw = [word.lower() for word in text_without_punc.split() if word.lower() not in stopwords.words('english')]
    text_without_sw = ' '.join(text_without_sw)
    
    #removing numbers
    text_without_num = [word for word in text_without_sw.split() if not word.isdigit()]
    text_without_num = ' '.join(text_without_num)
    
    #lemmatizing text_without_sw and saving into lemmatized
    lemmatized = [lemmatizer.lemmatize(word) for word in text_without_num.split()]
    lemmatized = ' '.join(lemmatized)
    
    #stemming lemmatized and saving into clean_text
    clean_text = [stemmer.stem(word) for word in lemmatized.split()]
    stemmed = ' '.join(clean_text)

    return clean_text

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
bag_of_words = CountVectorizer(analyzer=smooth).fit(X_train,y_train)
bow_transformed = bag_of_words.transform(X_train)

tfidf = TfidfTransformer().fit(bow_transformed)
tfidf_transformed = tfidf.transform(bow_transformed)

#transforming testing set
bag_of_words_test = bag_of_words.transform(X_test)
bow_test_transformed = tfidf.transform(bag_of_words_test)  


# evaluating accuracy using RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
rf = RandomForestClassifier(n_estimators=100).fit(tfidf_transformed, y_train)
rf_accuracy = rf.score(bow_test_transformed,y_test)
print("\n Accuracy of RandomForest Classifier" ,rf_accuracy)
predict_rf = rf.predict(bow_test_transformed)
print("\n \n ",classification_report(y_test, predict_rf))

# evaluating accuracy using Support Vector Classifier
from sklearn.svm import SVC
svc = SVC(C = 1000, gamma = 0.001).fit(tfidf_transformed, y_train)
svc_accuracy = svc.score(bow_test_transformed, y_test)
print("\n Accuracy of Support Vector Classifier" ,svc_accuracy)
predict_svc = svc.predict(bow_test_transformed)
print("\n \n ",classification_report(y_test, predict_svc))


# evaluating accuracy using LogisticRegression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs',multi_class='auto',n_jobs=10).fit(tfidf_transformed, y_train)
lr_accuracy = lr.score(bow_test_transformed, y_test)
print("\n Accuracy of LogisticRegression" ,lr_accuracy)
predict_lr = lr.predict(bow_test_transformed)
print("\n \n ", classification_report(y_test, predict_lr))

# evaluating accuracy using NaiveBayes
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(tfidf_transformed, y_train)
nb_accuracy = nb.score(bow_test_transformed, y_test)
print("\n Accuracy of Naive Bayes:", nb_accuracy)
predict_nb = nb.predict(bow_test_transformed)
print("\n\n", classification_report(y_test, predict_nb))


# prediction using Random Forest Classifier as it has highest accuracy.
from sklearn.pipeline import Pipeline
pl = Pipeline([
    ('vectorizer',CountVectorizer(analyzer=smooth)),
    ('tfidf',TfidfTransformer()),
    ('rfc',RandomForestClassifier(n_estimators=200))
])
pl.fit(X_train,y_train)
flag =0
while(flag == 0) :
    print("Choose an option 1. Enter a Tweet to check it's sentiment 2. Exit")
    value = int(input())
    if value == 2:
        break
    to_predict = input()
    prediction = pl.predict([to_predict])
    print("\n \n " ,to_predict ,' \n The tweet result is  : ',prediction[0])
    print("******************************************")
    print()
print("********* Thank You!! ***********")

# Bar chat representation of the accuracies.
models = ['Random Forest', 'Support Vector Classifier', 'Logistic Regression', 'Naive Bayes']
accuracies = [rf_accuracy, svc_accuracy, lr_accuracy, nb_accuracy]

plt.bar(models, accuracies, color=['green', 'orange', 'blue', 'red'])
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies')
plt.ylim(0, 1)  # Set the y-axis limit between 0 and 1 for accuracy
plt.show()

    
