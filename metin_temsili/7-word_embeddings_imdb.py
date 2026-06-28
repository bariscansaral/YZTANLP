"""

IMDB film yorumları üzerinden word2vec tabanlı kelime vektörleri üreteceğiz ve KMeans algoritması kullanarak kümelere ayıracağız.
Adımlar:
    -Veriseti yüklenecek
    -Metinler temizlenecek (küçük harf dönüşümü özel karakterlerden kurtulma kısa kelimelerin kaldırılması...)
    -Tokenizasyon yapılacak
    -word2vec modeli tanımlanacak ve embedding gerçekleştirilecek
    -ilk 500 kelime KMeans ile ikili kümeleme yapılacak
    -PCA ile 50 boyuttan 2 boyuta indigreme gerçekleştirilecek
    -Sonuçlar iki boyutta görselleştirilecek

"""

import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

#verisetini yükleme
df=pd.read_csv("IMDB Dataset.csv")
text=df["review"]
print(df.head())

#metin temizleme
def text_cleaning(text):
    text=text.lower()
    text=re.sub(r"\d+","",text) #sayıları kaldırır.
    text=re.sub(r"[^\w\s]","",text) #özel karakterleri kaldırır.
    text=" ".join([word for word in text.split() if len(word)>2]) #çok kısa kelimeleri siler.
    return text

#yorumların temizlemiş hali
cleaned_text=[text_cleaning(y) for y in text]
print(cleaned_text[:5])

#tokenizasyon
tokenize_text=[simple_preprocess(y) for y in cleaned_text]

#word2vec modeli tanımlama
word2vec_model=Word2Vec(
    sentences=tokenize_text, #eğitim verisi
    vector_size=50, # vektör boyutu
    window=5,
    min_count=1,
    sg=0
)

#eğitimden sonra kelime vektörlerini al
word_vect=word2vec_model.wv

#ilk 500 kelime
words=list(word_vect.index_to_key)[:500]
vectors=[word_vect[w] for w in words]

#KMeans algoritması
kmeans=KMeans(n_clusters=2,random_state=10) #2 küme belirleme
kmeans.fit(vectors)
labels=kmeans.labels_ #her küme için kelime etiketi

#PCA ile boyut indirgeme
pca=PCA(n_components=2)
pca_vectors=pca.fit_transform(vectors)

#görselleştirme
plt.figure()
plt.scatter(pca_vectors[:,0],pca_vectors[:,1], c=labels, cmap="viridis")

#küme merkezi işaretleme
centers=pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0],centers[:,1], c="red", marker="x", s=150, label="Merkez")

#kelimeleri noktaların üzerine yazalım
for i, word in enumerate(words):
    plt.text(pca_vectors[i,0],pca_vectors[i,1],word, fontsize=8)
plt.title("Word2Vec + KMeans kümeleme")
plt.legend()
plt.show()