"""

Küçük bir veriseti üzerinden Word Embedding yani kelime gömme yapılacak. Daha sonra PCA ile görselleştirme yapılacak.
    -word2vec: google tarafından geliştirilen embedding yöntemi
    -fasttext: facebook (meta) tarafından geliştirilen embedding yöntemi bu ikisini deneyeceğiz.
Adımlar:
    -Örnek cümle veriseti oluşturulacak
    -Cümleler tokenlara çevrilecek.
    -word2vec ve fasttext kullanarak modeller eğitilecek.
    -Her iki modelden elde edilen vektörleri pca analizi ile üç boyuta indirgenecek
    -Kelime vektörleri üç boyutlu olarak görselleştirilecek.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import Word2Vec,FastText
from gensim.utils import simple_preprocess

#Örnek veriseti oluşturma
sentences=[
    "Köpek çok tatlı bir hayvandır.",
    "Köpekler evcil hayvanlardır.",
    "Kediler genellikle bağımsız hareket etmeyi severler.",
    "Köpekler sadık ve dost canlısı hayvanlardır.",
    "Hayvanlar insanlar için iyi arkadaşlardır.",
    "Türkiye'nin başkenti Ankara'dır.",
    "Ankara ve Gaziantep yemekleri çok güzel."
]

#Cümleleri tokenize etmek (Küçük harf ve noktalama temizliği otomatik olarak yapılır tokenize ederken!!!!!!!!!!!!)
tokenize_sentences=[simple_preprocess(c) for c in sentences]
print("Tokenize edilmiş cümleler:\n",tokenize_sentences)

#word2vec
word2vec_model=Word2Vec(sentences=tokenize_sentences, #eğitim verisi
                        vector_size=50, #vektör boyutu
                        window=5, #pencere boyutu
                        min_count=1, #en az kaç defa geçen kelimeler alınsın
                        sg=0 #skip gram = 1
                        )

#fasttext
fasttext_model=FastText(sentences=tokenize_sentences,
                        vector_size=50,
                        window=5,
                        min_count=1,
                        sg=0)


def plot_word_embeddings(model,baslik):
    #modelin kelime vektörlerini alalım
    word_vector=model.wv
    #ilk 1000 kelimeyi almak (zaten bu verisetinde 1000 kelime yok)
    words=list(word_vector.index_to_key)[:1000]
    vectors=[word_vector[w] for w in words]

    #PCA ile boyut indirgemesi
    pca=PCA(n_components=3) #3 boyuta indirgeme
    pca_vectors=pca.fit_transform(vectors)

    #3 boyutlu görselleştirme
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(111, projection='3d')

    #noktaları çizdirme
    ax.scatter(pca_vectors[:,0],pca_vectors[:,1],pca_vectors[:,2])
    #kelimeleri noktaların yanına yazmak
    for i, word in enumerate(words):
        ax.text(pca_vectors[i,0],pca_vectors[i,1],pca_vectors[i,2], word, fontsize=13)
    ax.set_title(baslik)
    ax.set_xlabel("Bileşen 1")
    ax.set_ylabel("Bileşen 2")
    ax.set_zlabel("Bileşen 3")
    plt.show()

#plot_word_embeddings(word2vec_model,"Word2Vec Gösterimi")
plot_word_embeddings(fasttext_model,"FastText Gösterimi")