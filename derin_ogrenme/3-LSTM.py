"""

LSTM Tabanlı bir dil modeli ile metin üretimi gerçekleştirilecek (text generation).
Eğitim verisi olarak gemini ile oluşturulmuş türkçe günlük cümleler kullanacağız.
Model verilen bir başlangıç kelimesinden yeni kelimeler ya da cümleler üretir.

Adımlar:
    -Eğitim verisinin hazırlanması (gemini ile üretilmiş günlük türkçe cümleler)
    -Tokenizasyon işlemi gerçekleştirilecek (kelimeler sayısal vektörlere çevrilecek)
    -N-Gram dizileri oluşturulacak (dil modeli için girdi ve çıktı çiftleri hazırlanacak)
    -Padding yapılarak tüm diziler aynı uzunluğa getirilecek
    -LSTM tabanlı modelin kurulması gerçekleştirilecek
    -Modelin eğitimi gerçekleştirilecek
    -Yeni metin üretimi için fonksiyon yazılacak

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

#Eğitim veri setinin oluşturulması
texts = [
    "Bugün hava çok güzel, dışarıda yürüyüş yapmayı düşünüyorum.",
    "Kitap okumak beni gerçekten mutlu ediyor.",
    "Sabah erken kalkıp kahve içmeyi seviyorum.",
    "Yeni şeyler öğrenmek hayatımı daha anlamlı hale getiriyor.",
    "Bilgisayar programlama konusunda kendimi geliştirmek istiyorum.",
    "Doğada vakit geçirmek insanı sakinleştiriyor.",
    "Müzik dinlemek stresimi azaltıyor.",
    "Arkadaşlarımla güzel sohbetler yapmayı seviyorum.",
    "Yapay zeka teknolojileri her geçen gün gelişiyor.",
    "Python dili ile birçok farklı proje yapılabilir.",
    "Bugün ders çalışmak için iyi bir gün.",
    "Başarılı olmak için düzenli çalışmak gerekiyor.",
    "Sabahları spor yapmak enerjimi artırıyor.",
    "Yeni bir kitap okumaya başladım.",
    "Film izlemek boş zamanlarımda yaptığım aktivitelerden biridir.",
    "Kediler çok sevimli ve eğlenceli hayvanlardır.",
    "Kahve kokusu bana huzur veriyor.",
    "Teknoloji hayatımızın birçok alanını değiştirdi.",
    "Daha sağlıklı yaşamak için dengeli beslenmeliyiz.",
    "Öğrenmek insanın hayatı boyunca devam eden bir süreçtir.",
    "Bugün bilgisayarımda yeni bir proje geliştirdim.",
    "Yazılım geliştirme sabır ve dikkat gerektirir.",
    "Doğru planlama başarıya ulaşmayı kolaylaştırır.",
    "İnsanlar iletişim kurarak birbirlerini daha iyi anlayabilir.",
    "Güzel bir manzara izlemek beni rahatlatıyor.",
    "Yağmurlu havalarda evde kitap okumayı seviyorum.",
    "Yeni yerler keşfetmek farklı deneyimler kazandırır.",
    "Spor yapmak fiziksel ve zihinsel sağlığa faydalıdır.",
    "Gelecek için hedefler belirlemek önemlidir.",
    "Kendime yeni beceriler kazandırmaya çalışıyorum.",
    "Bir problemi çözmek bazen yaratıcı düşünmeyi gerektirir.",
    "Sabırlı insanlar hedeflerine daha kolay ulaşabilir.",
    "Bilgi paylaşmak öğrenme sürecini hızlandırır.",
    "İyi bir arkadaş zor zamanlarda destek olur.",
    "Güneşli günler insanın ruh halini olumlu etkiler.",
    "Tek başına zaman geçirmek bazen iyi hissettirir.",
    "Yeni tarifler denemek yemek yapmayı eğlenceli hale getirir.",
    "Hayallerimizi gerçekleştirmek için çalışmalıyız.",
    "Düzenli uyku sağlıklı bir yaşam için önemlidir.",
    "Kütüphanede sessiz bir ortamda çalışmayı seviyorum.",
    "İnternet sayesinde bilgiye ulaşmak çok kolaylaştı.",
    "Gelişen teknoloji yeni fırsatlar oluşturuyor.",
    "Bir işi severek yapmak başarıyı artırır.",
    "Hatalardan ders çıkarmak kişisel gelişim sağlar.",
    "Yolculuk yapmak farklı kültürleri tanımamıza yardımcı olur.",
    "Küçük adımlar büyük başarıların başlangıcı olabilir.",
    "Arkadaşlarla oyun oynamak eğlenceli vakit geçirmenin yoludur.",
    "Sabah kahvaltısı günün en önemli öğünlerinden biridir.",
    "Çalışma ortamının düzenli olması verimliliği artırır.",
    "Yeni fikirler üretmek yaratıcılığı geliştirir.",
    "Bilim insanları dünyayı anlamaya çalışıyor.",
    "Doğayı korumak gelecek nesiller için önemlidir.",
    "Güzel bir hikaye okumak hayal gücünü geliştirir.",
    "Teknoloji sayesinde insanlar daha hızlı iletişim kuruyor.",
    "Bir hedefe ulaşmak için kararlı olmak gerekir.",
    "Günlük tutmak düşüncelerimizi düzenlememize yardımcı olur.",
    "Resim yapmak duyguları ifade etmenin güzel bir yoludur.",
    "Dil öğrenmek farklı kültürleri anlamayı sağlar.",
    "Yeni bir dil öğrenmek sabır gerektirir.",
    "Mühendislik alanında yeni gelişmeler yaşanıyor.",
    "Yazılım projelerinde takım çalışması önemlidir.",
    "Veri bilimi günümüzde önemli bir alan haline geldi.",
    "Makine öğrenmesi birçok sektörde kullanılmaktadır.",
    "Derin öğrenme teknolojileri yapay zekanın gelişmesini sağlıyor.",
    "Doğal dil işleme bilgisayarların insan dilini anlamasına yardımcı olur.",
    "Programlama öğrenmek problem çözme yeteneğini geliştirir.",
    "Kod yazarken dikkatli olmak hataları azaltır.",
    "Bir projeyi tamamlamak büyük bir motivasyon sağlar.",
    "Yeni bir uygulama geliştirmek heyecan verici olabilir.",
    "Bilgisayar oyunları eğlenceli zaman geçirmeyi sağlar.",
    "İnsan beyninin çalışma şekli oldukça karmaşıktır.",
    "Araştırma yapmak yeni bilgiler edinmemizi sağlar.",
    "Merak etmek öğrenmenin başlangıcıdır.",
    "Güzel bir gün geçirmek için küçük şeylerden mutlu olabiliriz.",
    "Başarı için disiplinli olmak gerekir.",
    "Kendimizi geliştirmek için sürekli çalışmalıyız.",
    "Zamanı doğru kullanmak hayatımızı kolaylaştırır.",
    "Teknoloji ile eğitim imkanları artmaktadır.",
    "Online eğitim sayesinde yeni bilgiler öğrenilebilir.",
    "Bir kitap bazen insanın düşüncelerini değiştirebilir.",
    "Seyahat etmek farklı bakış açıları kazandırır.",
    "Dostluklar hayatımızı daha güzel hale getirir.",
    "Empati yapmak insan ilişkilerini güçlendirir.",
    "Pozitif düşünmek motivasyonu artırabilir.",
    "Sabırlı olmak zor durumlarda yardımcı olur.",
    "Yeni deneyimler insanı geliştirir.",
    "Kendine inanmak başarı yolunda önemlidir.",
    "Planlı çalışmak hedeflere ulaşmayı kolaylaştırır.",
    "Bilgisayar bilimleri sürekli gelişen bir alandır.",
    "Yapay zeka gelecekte birçok alanda kullanılacak.",
    "Veriler doğru analiz edildiğinde önemli bilgiler ortaya çıkar.",
    "Bir algoritma problemi çözmek için tasarlanır.",
    "Kodlama mantığı geliştikçe yeni projeler yapmak kolaylaşır.",
    "Öğrenme süreci zaman ve emek ister.",
    "Her gün yeni bir bilgi öğrenmek faydalıdır.",
    "Küçük başarılar büyük hedeflere götürebilir.",
    "İyi alışkanlıklar hayat kalitesini artırır.",
    "Teknoloji insan yaşamını kolaylaştırmaya devam ediyor.",
    "Gelecekte yeni keşifler yapılmaya devam edecek."
]

#Metin ön işleme aşaması (preprocessing)
#tokenizer -> her kelimeyi benzersiz bir indexe atar
tokenizer=Tokenizer()
tokenizer.fit_on_texts(texts)
total_words=len(tokenizer.word_index)+1 #toplam kelime sayımız

#n-gram dizileri oluşturma
#örn: "bugün hava çok güzel" -> input: "bugün hava" -> hedef:"çok"
input_sequences=[]
for text in texts:
    token_list=tokenizer.texts_to_sequences(text)[0] #cümleyi sayısal indexlere çevirir
    for i in range(1,len(token_list)):
        n_gram_sequences=token_list[:i+1]
        input_sequences.append(n_gram_sequences)

#en uzun dizinin uzunluğunu çekmek
max_sequence_length=max(len(x) for x in input_sequences)

#padding -> tüm dizileri aynı uzunluğa getirmek
input_sequences=pad.sequences(input_sequences,maxlen=max_sequence_length, padding="pre")

#girdi (x) ve hedef değişken (y) ayırma işlemi
X=input_sequences[:,: -1] #son kelime hariç tüm kelimeler girdi
y=input_sequences[:,-1]  # son kelime

#one-hot-encoding
y=tf.keras.utils.to_categorical(y,num_classes=total_words)