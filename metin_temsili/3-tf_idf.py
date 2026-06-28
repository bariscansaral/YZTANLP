"""

Örnek birkaç cümle üzerinden TF-IDF uygulayarak cümleleri vektörleştirmeyi istiyoruz.
Adımlar:
    -Küçük bir belge oluştur.
    -TF-IDF vectorizer ile belgeleri sayısal vektörlere dönüştür.
    -Kelime kümesi çıkar (vocab)
    -Belgelerin TF-IDF vektör temsillerini elde et
    -Tüm belgeler için kelimelerin ortalama TF-IDF değerlerini hesaplayacağız.

"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

#Örnek belgeleri tanımlama
docs=["köpek çok tatlı bir hayvandır.", #1. belge
      "köpek ve kuşlar çok tatlı hayvanlardır.",
      "inekler süt üretirler.",
      "köpek, köpek, köpek, köpek"
      ]

#TF-IDF işlemleri
tfidfmodel=TfidfVectorizer()
#Belgeleri sayısal vektörlere dönüştürme
docs_vector=tfidfmodel.fit_transform(docs)
#Kelime kümesini çıkarma
vocab=tfidfmodel.get_feature_names_out()
#Belgelerin TF-IDF değerlerini numpy formatına çevirelim
vector=docs_vector.toarray()
print(f"Kelime Kümesi:\n {vocab}\n\nTF-IDF matrisi:\n {vector}")
#vektör matrisini daha okunabilir hale getirmek için pandas dataframesine çevirelim
df_tfidf=pd.DataFrame(vector, columns=vocab)
print("\nVektör Dataframe Hali:\n",df_tfidf)

#Her kelimenin belgeler arasındaki ortalama tf idf değerini hesaplama
ortalama_tf_idf=df_tfidf.mean(axis=0)
print("Kelimelerin ortalama TF-IDF değerleri:\n",ortalama_tf_idf)