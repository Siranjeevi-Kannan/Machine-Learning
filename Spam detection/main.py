import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
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
vectorizer = CountVectorizer(ngram_range=(2,2))
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

# print("total",total,count )
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

#----------Before Custom Preprocessing(Bow)----------

#ngram_range(1,1)
# True Positives  136
# False positives  20
# Flase negatives  14
# Precision  0.8717948717948718
# Recall  0.9066666666666666
# F1  0.8888888888888887

#ngram_range(1,2)

# True Positives  138
# False positives  25
# Flase negatives  12
# Precision  0.8466257668711656
# Recall  0.92
# F1  0.8817891373801918


#ngram_range(2,2)
# True Positives  143
# False positives  176
# Flase negatives  7
# Precision  0.4482758620689655
# Recall  0.9533333333333334
# F1  0.6098081023454157

#----------After Custom Preprocessing----------

#ngram_range(1,1)
# True Positives  137
# False positives  22
# Flase negatives  13
# Precision  0.8616352201257862
# Recall  0.9133333333333333
# F1  0.8867313915857605

#ngram_range(1,2)
# True Positives  141
# False positives  29
# Flase negatives  9
# Precision  0.8294117647058824
# Recall  0.94
# F1  0.88125


#ngram_range(2,2)
# True Positives  145
# False positives  225
# Flase negatives  5
# Precision  0.3918918918918919
# Recall  0.9666666666666667
# F1  0.5576923076923077

'''
False Positive
tell reached
k call ah
okay name ur price long legal wen pick u ave x am xx
dont worry guess busy
new car house parent new job hand
evo download flash jealous
need coffee run tomo believe time week already
lol yes friendship hanging thread cause u buy stuff
coffee cake guess
ì_ come
know anthony bringing money school fee pay rent stuff like thats need help friend need
get ur st ringtone free reply msg tone gr top tone phone every week å per wk opt send stop
sorry call later
quite late lar ard anyway wun b drivin
hmmm thought said hour slave late punish
time fix spelling sometimes get completely diff word go figure
meanwhile shit suite xavier decided give u lt gt second warning samantha coming playing jay guitar impress shit also think doug realizes live anymore
love come took long leave zaher got word ym happy see sad left miss
shall fine avalarr hollalater
aight ill get fb couple minute
'''

'''
False Negative
k tell anything
would really appreciate call need someone talk
always putting business put picture as facebook one open people ever met would think picture room would hurt make feel violated
gon na go get taco
wait know wesley town bet hella drug
lol know dramatic school already closed tomorrow apparently drive inch snow supposed get
opinion jada kusruthi lovable silent spl character matured stylish simple pls reply
urgent call landline complimentary ibiza holiday cash await collection sae c po box sk wp ppm
nutter cutter ctter cttergg cttargg ctargg ctagg ie
hey gave photo registered driving ah tmr wan na meet yck
ur cash balance currently pound maximize ur cash send go p msg cc po box tcr w
yup bathe liao
free call sir waiting
'''
