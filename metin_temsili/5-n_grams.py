"""

Verilen örnek cümleler üzerinden N-gram (unigram, bigram, trigram) analizi gerçekleştireceğiz.
Adımlar:
    -Örnek belgeler tanımlanacak
    -CountVectorizer ile unigram bigram ve trigram örnekleri oluşturulacak
    -Her bir model için özellik (kelime, kelime grubu) listesi çıkarılacak
    -Sonuçlar ekrana yazdırılacak

"""

from sklearn.feature_extraction.text import CountVectorizer

#Örnek belgeler:

docs=[
    "Bu çalışma bir N-Gram çalışmasıdır.",
    "Bu çalışma doğal dil işleme çalışmasıdır."
]

#unigram objesi
unigram=CountVectorizer(ngram_range=(1,1))

#bigram objesi
bigram=CountVectorizer(ngram_range=(2,2))

#trigram objesi
trigram=CountVectorizer(ngram_range=(3,3))

#unigram analizi
X_unigram=unigram.fit_transform(docs) #sayısal vektöre dönüştürme işi
unigram_ozellikler=unigram.get_feature_names_out() #Kelime listesi, vocab

#bigram analizi
X_bigram=bigram.fit_transform(docs)
bigram_ozellikler=bigram.get_feature_names_out()

#trigram analizi
X_trigram=trigram.fit_transform(docs)
trigram_ozellikler=trigram.get_feature_names_out()

print("Unigram:\n",unigram_ozellikler)
print("\n\nBigram:\n",bigram_ozellikler)
print("\n\nTrigram:\n",trigram_ozellikler)