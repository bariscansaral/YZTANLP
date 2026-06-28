"""
SMS Spam veri seti üzerinden TF-IDF analizi gerçekleştireceğiz.
Adımlar:
    -csv dosyasından sms verisi yüklenecek
    -TF-IDF vectorizer kullanarak tüm sms metinlerini sayısal vektörlere dönüştürücez
    -Her kelimenin ortalama TF-IDF skoru hesaplanacak
    -Sonuçları df'e aktarıp en yüksek skora sahip 10 kelimeyi görüntüleyeceğiz

"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
stop_words=set(stopwords.words('english'))

df=pd.read_csv("sms_spam.csv")

#text sütununda yer alan sms mesajlarını alma
text=df["text"]
print(df.head())

#Önce 2. py dosyasındaki fonksiyonu kopyala yapıştır yapıyoruz düzenleme işiyle uğraşmamak için
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
cleaned_text=[clean_text(y) for y in text]


#tf-idf işlemleri
tf_idf_model=TfidfVectorizer()
#mesajları vektöre dönüştürme
text_vector=tf_idf_model.fit_transform(cleaned_text)
#kelime kümesi
vocab=tf_idf_model.get_feature_names_out()
#her kelimenin tf_idf skorları
tf_idf_scores=text_vector.mean(axis=0).A1
df_tf_idf=pd.DataFrame({"Kelime":vocab,"Ortalama TF-IDF":tf_idf_scores})
df_tf_idf_sorted=df_tf_idf.sort_values(by="Ortalama TF-IDF",ascending=False)
print(df_tf_idf_sorted[:10])