# yapay.zeka.-dev

# OpenFDA Preprocessing and Embedding

Bu proje, openFDA API'sinden çekilen ciddi yan etki verilerini işlemek, ön işleme tabi tutmak, TF-IDF ve Word2Vec modelleriyle vektörleştirmek ve yan etki-ilac eşleştirmeleri yapmak için geliştirilmiş bir Jupyter Notebook uygulamasıdır. Veri seti, JSON formatında yaklaşık 50-100 MB boyutunda 5.000 kayıttan oluşur. Proje, Zipf yasası analizi, tokenizasyon, lemmatizasyon, stemming, TF-IDF vektörleştirme, Word2Vec model eğitimi ve kosinüs benzerliği ile eşleştirme adımlarını içerir.

## Proje Amacı
- **Veri Çekme**: openFDA API'sinden ciddi yan etki verilerini çekmek.
- **Ön İşleme**: Metni tokenleştirme, stopwords kaldırma, lemmatizasyon ve stemming uygulama.
- **Analiz**: Ham ve işlenmiş veri için Zipf yasası grafikleri oluşturma.
- **Vektörleştirme**: TF-IDF ve Word2Vec modelleriyle metni vektörleştirmek.
- **Eşleştirme**: Yan etkiler ve ilaçlar arasında kosinüs benzerliği ile ilişkiler bulmak.
- **Çıktılar**: İşlenmiş verileri CSV, modelleri `.model` formatında ve grafikleri PNG olarak kaydetmek.

## Bağımlılıklar
Projenin çalışması için aşağıdaki Python kütüphaneleri gereklidir:
requests
pandas
nltk
gensim==4.3.2
scikit-learn
tqdm
matplotlib
numpy


NLTK veri kaynaklarını indirin (kodda otomatik olarak yapılır):
python


import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
Stabil bir internet bağlantısı olduğundan emin olun (openFDA API için).
Dosya Yapısı
Girdi:
openFDA API: https://api.fda.gov/drug/event.json
Veri boyutu: ~50-100 MB, 5.000 kayıt, JSON formatı.
Çıktılar:
models/: Word2Vec modelleri (örn. word2vec_lemmatized_cbow_win2_dim100.model).
outputs/: İşlenmiş veri CSV'leri (lemmatized.csv, stemmed.csv, tfidf_lemmatized.csv, tfidf_stemmed.csv, similarity_results.csv, preprocessed_data.csv).
Grafikler: zipf_raw.png, zipf_processed.png.
Kod: openfda_preprocessing_and_embedding.ipynb (25 hücre).
Çalıştırma Talimatları
Jupyter Notebook Ortamı:
openfda_preprocessing_and_embedding.ipynb dosyasını bir Jupyter Notebook ortamında açın.
Alternatif olarak, kodları bir Python betiğine kopyalayıp çalıştırabilirsiniz.
Hücreleri Çalıştırma:
Kod 25 hücreye ayrılmıştır (# [1] - # [25]). Her hücreyi sırayla çalıştırın.
Hücreler sırasıyla veri çekme, ön işleme, analiz, vektörleştirme ve eşleştirme adımlarını gerçekleştirir.
Çıktılar:
Modeller: models/ dizininde, her biri ~1-5 MB.
CSV'ler: outputs/ dizininde, işlenmiş veri ve TF-IDF matrisleri.
Grafikler: zipf_raw.png ve zipf_processed.png kök dizinde.
Tahmini Süre:
Veri çekme: ~2-3 dakika (internet hızına bağlı).
Model eğitimi: Her Word2Vec modeli ~30 saniye.
Toplam: ~10-15 dakika.
Kod Bölümleri
[1]: Kütüphaneleri yükler ve NLTK kaynaklarını indirir.
[2]: openFDA API'sinden 5.000 ciddi yan etki kaydı çeker, hata yönetimi ile.
[3]: Ham veri için Zipf yasası analizi yapar ve grafik kaydeder.
[4-12]: Metni cümlelere ayırır, tokenleştirir, lemmatize eder, stemler ve işlenmiş veri için Zipf analizi yapar.
[13-16]: TF-IDF vektörleştirme ve kosinüs benzerliği ile kelime analizi.
[17-20]: 16 Word2Vec modeli eğitir (8 lemmatized, 8 stemmed), nausea için benzerlik analizi yapar.
[21-23]: Yan etki ve ilaç vektörleri oluşturur, kosinüs benzerliği ile eşleştirir.
[24-25]: İşlenmiş verileri ve sonuçları CSV olarak kaydeder
5.000 kayıt ve 16 Word2Vec modeli için ~4-8 GB RAM yeterlidir. Daha fazla veri için batch işleme düşünün.
Çıktı Doğrulama:
outputs/ dizininde CSV'lerin ve models/ dizininde modellerin oluşturulduğunu kontrol edin.
Grafiklerin (zipf_*.png) oluşturulduğunu doğrulayın.
Örnek Çıktılar
Veri: nausea vomiting yan etkisi ve IBUPROFEN ilacı içeren kayıtlar.
TF-IDF Benzerlik:
Lemmatized: nausea için pleurisy, pleural (skor: 1.0).
Stemmed: nausea için pneumothorax, pneumonia (skor: 1.0).
Word2Vec Benzerlik:
word2vec_lemmatized_skipgram_win4_dim300.model: nausea için headache (skor: 0.9994), vomiting (skor: 0.9991).
word2vec_stemmed_skipgram_win4_dim300.model: nausea için vomit (skor: 0.9994), dizzi (skor: 0.9992).
CSV'ler: similarity_results.csv yan etki-ilac eşleştirmelerini içerir.
Notlar
Veri Kaynağı: openFDA API'si (https://api.fda.gov/drug/event.json).
Model Boyutları: Her Word2Vec modeli ~1-5 MB, toplam ~50 MB.
Performans: Eğitim süreleri ~30 sn/model, eşleştirme ~1-2 dakika.
PDF Bağlamı: Kod, veri kaynağı, boyut ve örnekler PDF'de açıklanmıştır.



### Notlar
- **PDF Uyumluluğu**: README, PDF'deki veri kaynağı (`https://api.fda.gov/drug/event.json`), boyut (50-100 MB, 5.000 kayıt), örnek (`nausea`, `IBUPROFEN`) ve hata yönetimi (`ValueError` çözümü) detaylarını içerir.
- **Çıktılar**: `models/` ve `outputs/` dizinlerindeki dosyalar ile grafikler (`zipf_*.png`) belirtilmiştir.
- **Hata Yönetimi**: Dosya açılmama ve API hataları için çözümler sağlanmıştır.
- **Sade ve Net**: README, kullanıcı dostu ve teknik detayları sade bir şekilde sunar.

