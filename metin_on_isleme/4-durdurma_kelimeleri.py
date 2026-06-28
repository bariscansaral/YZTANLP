"""
Durdurma kelimelerini çıkarma yöntemleri.
    -İngilizce stop words çıkarma (nltk kütüphanesi kullanılacak)
    -Türkçe stop words çıkarma (nltk kütüphanesi kullanılacak)
    -Kütüphanesiz manuel olarak stop words çıkarma
"""

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords') #Stopwords veri setini indirir



#İngilizce Stop words analizi
stop_words_eng=set(stopwords.words('english')) #İngilizce stop words listesi
eng_text="This is just a simple example to show how to stop words can be removed from sentences."
eng_text_list=eng_text.split()
print(eng_text_list)
filtered_words_eng=[word for word in eng_text_list if word.lower() not in stop_words_eng]
print("Orijinal:",eng_text)
print("Filtered:",filtered_words_eng)



#Türkçe stop words analizi
stop_words_tur=set(stopwords.words('turkish')) #Türkçe stop words listesi
tur_text="Merhaba, bugün sizler ile birlikte Google NLP eğitimi gerçekleştiriyoruz. Bu eğitim sizler için çok faydalı olacaktır."
tur_text_list=tur_text.split()
filtered_words_tur=[word for word in tur_text_list if word.lower() not in stop_words_tur]
print("Orijinal:",tur_text)
print("Filtered:",filtered_words_tur)



#Manuel stop words çıkarma işlemi
custom_tr_stopwords=["bu","ile","de","da","mi"]
custom_text="Bu bir denemedir, bunun için amacımız metinlerdeki bazı kelimeleri çıkartmaktır."
custom_text_list=custom_text.split()
filtered_custom_words_tr=[word for word in custom_text_list if word.lower() not in custom_tr_stopwords]
print("Orijinal:",custom_text)
print("Filtered:",filtered_custom_words_tr)