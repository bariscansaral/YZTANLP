"""

IMDB film yorumları veriseti üzerinden GRU tabanlı bir duygu analizi (sentiment classification) modeli geliştirilecek.

Adımlar:
    -Gerekli kütüphaneler import edilecek ve gerekli veriseti yüklenecek
    -Padding ile sabit uzunluklu dizilere dönüştürme işlemi gerçekleştirilecek
    -GRU tabanlı model kurulacak
    -Modelin derlenmesi ve eğitimlesi yapılacak
    -Modelin test seti üzerinden değerlendirilmesi, test edilmesi (evaluation)
    -Yeni yorum tahmini için bir fonksiyon yazılacak

"""

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense

# ==========================================
# 1. PARAMETRELER VE VERİSETİ YÜKLEME
# ==========================================
num_words = 10000  # Sözlükte tutulacak en sık geçen kelime sayısı
max_sequence_length = 200  # Her yorumun sabitleneceği kelime (token) sayısı

# X: Sayısal matris (yorumlar), y: Etiketler (0 = Negatif, 1 = Pozitif)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)
print(f"Train Boyutu: {len(X_train)} | Test Boyutu: {len(X_test)}")

# ==========================================
# 2. PADDING (BOYUT SABİTLEME) İŞLEMİ
# ==========================================
X_train_padded = pad_sequences(X_train, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test, maxlen=max_sequence_length)
print(f"X_train_padded Şekli: {X_train_padded.shape}")
print(f"X_test_padded Şekli:  {X_test_padded.shape}")

# ==========================================
# 3. GRU MODEL MİMARİSİNİN KURULMASI
# ==========================================
embedding_dim = 100  # Her kelimenin temsil edileceği vektör boyutu

# Boş ardışık model nesnesi oluşturuluyor
model = Sequential()

# Katmanlar sırayla .add() ile ekleniyor
model.add(Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(GRU(units=64, return_sequences=False))
model.add(Dense(units=1, activation='sigmoid'))
"""
model = Sequential([
    # Kelimeleri yoğun vektörlere dönüştüren katman
    Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=max_sequence_length),

    # Zaman serisi/metin verisini işleyen GRU katmanı
    GRU(units=64, return_sequences=False),

    # İkili sınıflandırma (0 veya 1) için çıkış katmanı
    Dense(units=1, activation='sigmoid')
])

İstersen model nesesini böyle de oluşturabilirsin bunun avantajı okunabilirliktir daha kompakt ve düzgün bir kod sunar

Üstte yaptığımız model.add kod dizisinin avantajı da esnekliktir Kod çalışırken (dinamik olarak) modele müdahale etmeyi kolaylaştırır. 
Örneğin bir for döngüsü veya if-else koşulu yazarak katman sayılarını ya da nöron sayılarını otomatik olarak değiştirebilirsin (Hyperparameter tuning yaparken çok işe yarar).
"""


# Modelin derlenmesi
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ==========================================
# 4. MODELİN EĞİTİLMESİ VE DEĞERLENDİRİLMESİ
# ==========================================
history = model.fit(
    X_train_padded, y_train,
    batch_size=128,
    epochs=3,
    validation_split=0.2,
    verbose=1
)

# Test seti performans ölçümü
loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=1)
print(f"\n[SONUÇ] Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f}")

# ==========================================
# 5. KELİME DICTIONARY MAPPING VE TAHMİN FONKSİYONLARI
# ==========================================
# Keras'ın orijinal sözlüğünü indiriyoruz: {"the": 1, "a": 2, ...}
word_index = imdb.get_word_index()

# İndeksleri kaydırarak tersine çeviriyoruz (Index -> Word)
index_to_word = {v + 3: k for k, v in word_index.items()}
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<UNKNOWN>"


def decode_review(encoded_review):
    """ Sayı dizisi halindeki yorumu tekrar okunabilir metne dönüştürür. """
    return " ".join([index_to_word.get(i, "?") for i in encoded_review])


def classify_review(review_sequence):
    """ Sayısal yorum dizisini modele sokup tahmin üretir. """
    padded = pad_sequences([review_sequence], maxlen=max_sequence_length)
    prob = model.predict(padded)[0][0]
    label = "Positive" if prob > 0.5 else "Negative"
    return label, prob


# ==========================================
# 6. TEST VE GÖSTERİM
# ==========================================
print("\n--- Orijinal Metne Dönüştürülen Yorum ---")
decoded = decode_review(X_test[0])
print(decoded)

print("\n--- Model Tahmini ---")
pred_label, prob = classify_review(X_test[0])
print(f"Tahmin: {pred_label} (Güven Skoru: {prob:.4f})")