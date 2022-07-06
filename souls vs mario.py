# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 21:08:58 2022

@author: Nate
"""
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
import spacy
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from spacy import displacy
import requests
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
from nltk.corpus import stopwords
    


URL="https://darksouls.wiki.fextralife.com/Lore"

html_content = requests.get(URL).text
soup = BeautifulSoup(html_content, "lxml")
stop_words = set(stopwords.words('english'))

#get dark souls paragraphs
dark_page = requests.get("https://darksouls.wiki.fextralife.com/Lore")
dark_soup = BeautifulSoup(dark_page.content, 'html.parser')
#print(dark_soup)
dark_body=dark_soup.body.text
#dark_paras = dark_soup.find_all('p')

#soup.find_all('document')[-1].extract()

dark_p = dark_soup.find_all('p')
dark_par = []
for x in dark_p:
    x = x.get_text()
    dark_par.append(str(x))
    
   
#get mario paragraphs
mario_page = requests.get("https://www.mariowiki.com/Super_Mario_Galaxy")
mario_soup = BeautifulSoup(mario_page.content, 'html.parser')
#print(mario_soup)
mario_body=mario_soup.body.text
#dark_paras = dark_soup.find_all('p')


mario_p = mario_soup.find_all('p')
mario_par = []
for x in mario_p:
    x = x.get_text()
    mario_par.append(str(x))
    

#entity recognition
NER = spacy.load("en_core_web_sm")

#get dark souls entities
dark_entities= NER(dark_body)

#print ds gpe entities and labels
# def unique(lst):
#     x = np.array(lst)
#     print(np.unique(x))

# dark_ent_people = []

# for word in dark_entities.ents:
#     if word.label_ == "PERSON":
#         if word.text not in dark_ent_people:
#             dark_ent_people.append([word.text,word.label_])
#         else:
#             continue
# print(unique(dark_ent_people))

#get mario entities
# mario_entities = NER(mario_body)
# mario_ent_people = []

# for word in mario_entities.ents:
#     if word.label_ == "PERSON":
#         if word.text not in mario_ent_people:
#             mario_ent_people.append([word.text,word.label_])
#         else:
#             continue
# print(unique(mario_ent_people))



#word clouds
# dark_nostop = ' '.join([word for word in dark_body.split() if word not in (stopwords.words('english'))])
# mario_nostop = ' '.join([word for word in mario_body.split() if word not in (stopwords.words('english'))])


# dark_wordcloud = WordCloud().generate(dark_nostop)
# plt.imshow(dark_wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.savefig('wordcloud11.png')
# plt.show()


# mario_wordcloud = WordCloud().generate(mario_nostop)
# plt.imshow(mario_wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.savefig('wordcloud11.png')
# plt.show()
    


#text classification, bag of words

#ds data frame
df_dark = pd.DataFrame(dark_par, columns =['Text'])
df_dark['Source'] = "Dark Souls"
df_dark = df_dark.drop(df_dark.index[141:])

#mario data frame
df_mario = pd.DataFrame(mario_par, columns =['Text'])
df_mario['Source'] = "Mario"

#combine
df_combined = pd.concat([df_dark,df_mario],ignore_index=True)


#modeling
df_dummy = pd.get_dummies(df_combined, columns = ['Source'])

labels = df_dummy[['Source_Dark Souls']]
data = df_dummy[['Text']]

cv = CountVectorizer()
tfidf = TfidfTransformer()

train_data, test_data, train_labels, test_labels = train_test_split(data,labels,test_size=0.2,random_state = 10)


train_data = cv.fit_transform(train_data['Text'])
test_data = cv.transform(test_data['Text'])
train_data_tfidf = tfidf.fit_transform(train_data)
test_data_tfidf = tfidf.fit_transform(test_data)

#decision tree
scores = []

for i in range(1,21):
  tree = DecisionTreeClassifier(random_state = 1,max_depth=i)
  tree.fit(train_data,train_labels)
  scores.append(tree.score(test_data,test_labels))

# plt.plot(range(1,21),scores)
# plt.show()

best_score = max(scores)
print("Tree: Depth with best score is " + str(scores.index(best_score) + 1) + " with score of " + str(best_score))
y_pred_tree = tree.predict(test_data)
print(confusion_matrix(test_labels, y_pred_tree))

#K Nearest Neighbors
scores_knn = []

for i in range(1,51):
    KNN = KNeighborsClassifier(n_neighbors = i)
    KNN.fit(train_data,np.ravel(train_labels))
    scores_knn.append(KNN.score(test_data,test_labels))
   
best_score_knn = max(scores_knn)
print("KNN: Best score uses k of " + str(scores_knn.index(best_score_knn) + 1) + " with score of " + str(best_score_knn))
y_pred_knn = KNN.predict(test_data)
print(confusion_matrix(test_labels, y_pred_knn))


#svm
svm = SVC()
svm.fit(train_data,np.ravel(train_labels))
print("\nSVM: " + str(svm.score(test_data, test_labels)))
y_pred_svm = svm.predict(test_data)
print(confusion_matrix(test_labels, y_pred_svm))


#pass agg
passagg = PassiveAggressiveClassifier(C = 0.5)
passagg.fit(train_data,np.ravel(train_labels))
print("\nPass-agg: " + str(passagg.score(test_data, test_labels)))
y_pred_passagg = passagg.predict(test_data)
print(confusion_matrix(test_labels, y_pred_passagg))


#tree tfidf

# scores_tfidf = []

# for i in range(1,21):
#   tree = DecisionTreeClassifier(random_state = 1,max_depth=i)
#   tree.fit(train_data_tfidf,train_labels)
#   scores_tfidf.append(tree.score(test_data_tfidf,test_labels))

# plt.plot(range(1,21),scores)
# plt.show()

# best_score = max(scores_tfidf)
# print("Tree: Depth with best score is " + str(scores_tfidf.index(best_score) + 1) + " with score of " + str(best_score))
# y_pred_tree = tree.predict(test_data_tfidf)
# print(confusion_matrix(test_labels, y_pred_tree))


# new_points = ["The opening cutscene establishes the premise of the game. Dragons once ruled the world during the Age of Ancients. A primordial fire known as the First Flame manifests in the world, establishing a distinction between life and death, and light and dark. Four beings find Lord Souls near the First Flame, granting immense power: Gwyn: the Lord of Sunlight, Nito: the First of the Dead, the Witch of Izalith, and the Furtive Pygmy. Gwyn, Nito, and the Witch use their new power to destroy the dragons and take control over the world, while the Furtive Pygmy is said to be forgotten, and thus begins the Age of Fire. Over time, as the First Flame begins to fade while humans rise in power, Gwyn sacrifices himself to prolong the Age of Fire. The main story takes place towards the end of this second Age of Fire, at which point humanity is said to be afflicted with an undead curse related to a circular, flaming symbol on their bodies known as the Darksign. Those humans afflicted with the undead curse perpetually resurrect after death until they eventually lose their minds, a process referred to as hollowing."]
# new_points = cv.transform(new_points)
# print(tree.predict(new_points))


