import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
import re

path = "D:\\Coding\\Machine Learning\\Spam detection\\dataset\\spam.csv"
data = pd.read_csv( path , encoding="latin1")
data=data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
data.rename(columns={"v1":"label", "v2":"msg"},inplace=True)
le = LabelEncoder()
data['label_num'] = le.fit_transform(data["label"]) #spam = 1
#print(data)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

x = data['msg']
y = data['label_num']
x = data['msg'].apply(clean_text)
msgs = x.to_numpy()
tokenized_msg = [word_tokenize(d) for d in msgs]

stop_words = set(stopwords.words("english"))
filtered_msg = [[ns for ns in d if ns not in stop_words] for d in tokenized_msg]
lemmentizer = WordNetLemmatizer()
lemmed_msg = [[lemmentizer.lemmatize(word) for word in d] for d in filtered_msg]

prevector_msg = [' '.join(d) for d in lemmed_msg]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(prevector_msg)
features = vectorizer.get_feature_names_out()

# sns.countplot(x=data['label'])
# plt.title("Count of 0s and 1s")
# plt.show()

xtrain, xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2,random_state=42)

cls = MultinomialNB()
cls.fit(xtrain,ytrain)
print("Model Created")

ypred = cls.predict(xtest)

total = ytest.size
count  = sum(ytest != ypred)

print("total",total,count )
#total 1115 35

true_positive = sum((ypred==1)&(ytest==1))
false_positive = sum((ypred==1)&(ytest==0))
false_negative = sum((ypred==0)&(ytest==1))
print("Total ",total)
print("True Positives ",true_positive)
print("False positives ",false_positive)
print("Flase negatives ",false_negative)

precision = true_positive/(true_positive+false_positive)
recall = true_positive/(true_positive+false_negative)
f1 = 2*(precision*recall)/(precision+recall)

print("Precision ",precision)
print("Recall ",recall)
print("F1 ",f1)


#----------Before Custom Preprocessing(Tf-idf)----------

# True Positives  108
# False positives  0
# Flase negatives  42
# Precision  1.0
# Recall  0.72
# F1  0.8372093023255813

#ngram_range(1,2)
# True Positives  90
# False positives  0
# Flase negatives  60
# Precision  1.0
# Recall  0.6
# F1  0.7499999999999999

#ngram_range(2,2)
# True Positives  85
# False positives  0
# Flase negatives  65
# Precision  1.0
# Recall  0.5666666666666667
# F1  0.7234042553191489

#----------After Custom Preprocessing----------

# True Positives  106
# False positives  0
# Flase negatives  44
# Precision  1.0
# Recall  0.7066666666666667
# F1  0.828125

#ngram_range(1,2)
# True Positives  95
# False positives  0
# Flase negatives  55
# Precision  1.0
# Recall  0.6333333333333333
# F1  0.7755102040816326

#ngram_range(2,2)
# True Positives  78
# False positives  0
# Flase negatives  72
# Precision  1.0
# Recall  0.52
# F1  0.6842105263157895