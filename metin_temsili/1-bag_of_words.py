"""

Basit bir metin listesini alarak skcikit-learn'ın count vectorized sınıfıyla sayısal vektörlere dönüştüreceğiz yani metin temsili yapacağız ve bunu yaparken bag of words kullanacağız.

"count vectorized" kelimelerin kaç defa geçtiğini sayar ve vektör temsiline dönüştürür.

    -kelime kümesi (vocabulary)
    -her metin listesi sayısal vektörler ile temsil edilecek

"""

from sklearn.feature_extraction.text import CountVectorizer

#Örnek metinlerden oluşan küçük bir veri seti
dokumanlar=[
    "kedi bahçede", #1. örnek cümle
    "kedi evde"     #2. örnek cümle
]

kelime_sayac=CountVectorizer()

#dokumanları sayısal vektörleri çevirme yani bag of words uygulama
dokuman_vektorleri=kelime_sayac.fit_transform(dokumanlar)
kelime_kumesi=kelime_sayac.get_feature_names_out()
print(kelime_kumesi)

vektor_temsili=dokuman_vektorleri.toarray()
print(vektor_temsili)