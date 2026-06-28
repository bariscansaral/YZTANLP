"""
Amaç
    Temel veri temizleme adımları:
    1-Fazla boşlukların kaldırılması
    2-Büyük harflerin küçük harflere dönüştürülmesi
    3-Noktalama işaretlerinin kaldırılması (anlamsal olarak önemli olsa da çoğu zaman gürültü olarak kabul edilir)
    4-Özel karakterlerin kaldırılması
    5-Yazım hatalarının düzeltilmesi
    6-HTML etiketlerinin kaldırılması

    Bu işlemler için  "textblob" ve "beautifulsoup4" kütüphanelerinden yararlanacağız
"""


#Fazla boşlukların temizlenmesi
raw_text="Python,     Google    NLP    dersi."
print(raw_text.split())#Elimizde bulunan texti boşluklara göre ayırma işlemini gerçekleştirecek fonksiyon.
normalized_text1=" ".join(raw_text.split())
print("Fazla boşlukları temizlenmiş veri: ",normalized_text1)



#Büyük ve küçük harf dönüşümü
raw_text="PYTHON, GooGle NLP"
normalized_text2=raw_text.lower() # Tüm harfleri küçük yapan fonksiyon
print(f"Büyük küçük harfi düzeltilmiş veri: {normalized_text2}")



#Noktalama işaretlerinden kurtulma
import string
raw_text="AI Natural-Language-Processing!"
normalized_text3=raw_text.translate(str.maketrans('', '', string.punctuation)) #Noktalama işaretlerini boş string ile değiştiren fonksiyon
print("Noktalama işaretleri silinmiş veri:", normalized_text3)



#Özel karakterlerden kurtulma
import re #Regular expression kütüphanesi yani düzenli ifadeler.
raw_text="Natural @ Language % Processing &"
normalized_text4=re.sub(r"[^A-Za-z0-9\s]","",raw_text)
print("Özel karakterleri silinmiş veri:", normalized_text4)



#Yazım hatalarından kurtulma
from textblob import TextBlob #İngilizce odaklı kelimeler için çalışan bir kütüphane
raw_text="İt is ammaizing in 2045"
normalized_text5=TextBlob(raw_text).correct() #Yazım hatalarını düzelt
print("Yazım hatası düzeltilmiş veri:", normalized_text5)



#HTML etiketlerinden düz metin elde etmek
from bs4 import BeautifulSoup
raw_html="<div> 2045 Google <div>"
normalized_text6=BeautifulSoup(raw_html,"html.parser").get_text()
print("HTML etiketlerinden temizlenmiş veri:", normalized_text6)