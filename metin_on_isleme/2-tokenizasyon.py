"""

Doğal dil işleme için temel ön adımlarımızdan olan 'tokenizasyon' gerçekleştirilecek
    1-Kelime tokenizasyonu
    2-Cümle tokenizasyonu
    3-Karakter tokenizasyonu (ilerde anlatılacak)

Tokenizasyon işlemi için 'nltk' kütüphanesi kullanılacak

"""

import nltk #natural language tool kit bu kütüphanenin açılımı
nltk.download("punkt")      # Kelime ve cümle tokenizasyonu için gerekli veriyi indirir.
nltk.download("punkt_tab")  # word_tokenize() fonksiyonunun ihtiyaç duyduğu punkt_tab kaynak dosyalarını indirir.

#Örnek text tanımla

raw_text="Merhaba Google! Bu bir NLP eğitimidir. NLP eğitiminin ilerleyen aşamalarında LLM konusunu öğrenelim."

#Kelime Tokenizasyonu
word_tokens=nltk.word_tokenize(raw_text)
print(f"Kelime tokenleri: {word_tokens}")


#Cümle tokenizasyonu

sentence_tokens=nltk.sent_tokenize(raw_text)
print(f"Cümle tokenleri: {sentence_tokens}")