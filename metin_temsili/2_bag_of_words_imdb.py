"""

IMDB film yorumları içeren veri seti ile Bag of words ile metin temsili gerçekleştiricez
    -csv dosyasından veri okunacak(IMDB Dataset.csv)

    -text cleaning yapılacak (küçük büyük harf, rakam özel harflerden kurtulma ve stop wordslerden kurtulma)
    -bag of words yöntemiyle metinleri sayısal vektörlere dönüştürücez yani metin temsili gerçekleştirilecek
    - kelime frekansları hesaplanacak ve en sık geçen 5 kelime listelenecek

"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words=set(stopwords.words('english'))

df=pd.read_csv("IMDB Dataset.csv")
reviews=df["review"] #Yorumları değişkene atadık
sentiment=df["sentiment"]  #Etiketleri değişkene atadık

#Metin temizleme fonksiyonu
def clean_text(text):
    #Tüm harfleri küçük harfe çevirmek
    text=text.lower()
    #Rakamları kaldırma
    text=re.sub(r"\d+","",text)
    #Özel karakterleri kaldırma
    text=re.sub(r"[^\w\s]","",text)
    #Çok kısa kelimeleri (2 harften kısa) silme
    text=" ".join([word for word in text.split() if len(word)>2])
    #stopwordleri çıkarmak

    text = " ".join([word for word in text.split() if word not in stop_words])
    #Temizlenmiş veriyi return etmek
    return text

cleaned_text=[clean_text(y) for y in reviews]

#Bag of Words modeli
bow_model=CountVectorizer()
review_vectors=bow_model.fit_transform(cleaned_text[:75])
print("Yorum vektörleri",review_vectors)

#vocabulary (kelime kümesi)
vocab=bow_model.get_feature_names_out()
#arraye çevirelim
vector=review_vectors.toarray()
print("Vektör temsili: ",vector) #Düzgün bir çıktı vermedi o yüzden bu temsilleri df'e dönüştüreceğiz

#Vektörleri df'e çevirme
df_bow=pd.DataFrame(vector,columns=vocab)
print("df_bow çıktısı: ",df_bow.head())

#Kelime frekanslarını hesaplayacağız. Her kelimenin toplam kaç adet geçtiğini bulacağız.

word_counts=review_vectors.sum(axis=0).A1

#Kelimeler ile frekansları bir sözlükte eşleştirelim
word_freq=dict(zip(vocab,word_counts))
#En çok geçen 5 kelime

most_5=Counter(word_freq).most_common(5)
print("\nEn çok geçen 5 kelime",most_5)