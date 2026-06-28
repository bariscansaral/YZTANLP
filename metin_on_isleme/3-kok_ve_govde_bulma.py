"""
Kök (stem) ve gövde (lemma) bulma
    -stemming: porter stemmer
    -lemmatization: word net lemmatizer

nltk kütüphanesi kullanacağız.
"""

import nltk
nltk.download('wordnet') #lemma bulmak için gerekli wordnet veri tabanı
nltk.download("omw-1.4") #wordnet için ek dil desteği

#Stemming (Kök bulma)

from nltk.stem import PorterStemmer #İngilizce için popüler stemmer algoritması
stemmer=PorterStemmer() #Porter stemmer nesnesini oluşturmak için
words_stem=["playing","played","plays","happier","happily","studying","studies"]
stems=[stemmer.stem(w) for w in words_stem] #words_stem listesi içinde dolaşan w'nun olduğu kelimenin kökünü bulup stems listesine ataması işlemi
print("Orijinal:",words_stem)
print("Stems:",stems) #playi güzel buluyor ancak happy ve study fiilerinde sorun var.


#Lemmatization (Gövde bulma)

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
words_lemma=["running","ran","gone","better","children"]
lemma=[lemmatizer.lemmatize(w) for w in words_lemma]
print("Orijinal:",words_lemma)
print("Lemmas:",lemma)