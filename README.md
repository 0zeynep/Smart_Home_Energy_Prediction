# Sertifikalarım

# Smart Home Energy Prediction

## Projenin Amacı

Bu proje, akıllı ev sistemlerinde enerji verimliliğini arttırmak için gelecekteki enerji tüketimini tahmin etmeyi amaçlamaktadır.

## Kullanılan Veri Seti
Projede Smart Home Dataset.csv kullanılmıştır.Veri seti şunları içerir:

Time(unix formatında),

House overall [kW] / use [kW]: Evin o saniyedeki toplam elektrik tüketimi.

gen [kW]: Evde elektrik üretimi varsa üretilen miktar.

Dishwasher [kW]: Bulaşık makinesi tüketimi.

Furnace 1 & 2 [kW]: Isıtma sistemi / fırın ünitelerinin tüketimi.

Home office [kW]: Ev ofisindeki cihazların tüketimi.

Fridge [kW]: Buzdolabı tüketimi.

Wine cellar [kW]: Şarap mahzeni soğutucusu tüketimi.

summary: Hava durumunun metin özeti (Örn: Clear - Açık, Cloudy - Bulutlu).

apparentTemperature: Hissedilen sıcaklık

cloudCover: Bulutluluk oranı (0-1 arası veya kategori).

## Veri Önişleme ve Temizlik

Eksik veriler temizlendi.

     df = df.dropna() 

Unix .okunabilir datetime formatına çevrildi.

    df['time'] = pd.to_numeric(df['time'], errors='coerce') # Time sütunundaki her şeyi sayıya çevirir
    df['datetime'] = pd.to_datetime(df['time'], unit='s')#unix time ı okunabilir zamana çevirir
    df['hour'] = df['datetime'].dt.hour # saat bilgisini alır

Makine öğrenmesi modellerinin anlayabilmesi için summary  ve cloudCover gibi metinsel veriler Label Encoding ile sayısal değerlere dönüştürüldü.

    le=LabelEncoder() #Kategorik verileri sayısal değere  çevirir  (0,1,2,3)
    df['summary']=le.fit_transform(df['summary'].astype(str)) 
    df['cloudCover']=le.fit_transform(df['cloudCover'].astype(str))

## Future Enginering (Özellik Mühendisliği)

  Modelin başarısını arttırmak için baseline oluşturulmuştur.
  
  Elektrik tüketimi büyük ölçüde günün saati ve hava durumu koşullarına bağlıdır. Modelin bu deseni daha kolay öğrenmesi için şu işlem uygulanmıştır:
  
  1) Veri seti Saat (hour) ve Hava Durumu Özeti (summary) bazında gruplandırılmıştır.
  2)  Her grup için cihazların (Ev geneli, Bulaşık makinesi, Buzdolabı) ortalama tüketim değerleri hesaplanmıştır.
  3)  Bu ortalama değerler, transform('mean') fonksiyonu ile veri setinin boyutunu değiştirmeden, ilgili satırlara referans (baseline) sütunu olarak eklenmiştir.
     

    cihazlar = ['House overall [kW]', 'Dishwasher [kW]', 'Fridge [kW]']
    for cihaz in cihazlar:
    column = f'{cihaz}_baseline'
    # 'hour' ve 'summary' (Hava durumu) bazında grupla, cihazın ortalamasını al
    df[column] = df.groupby(['hour', 'summary'])[cihaz].transform('mean')

    print(df[['hour', 'summary', 'House overall [kW]', 'House overall [kW]_baseline']].head())

Bu işlem sayesinde modele şu ipucu verilmiştir: "Şu anki saatte ve bu hava koşulunda, bu ev geçmişte ortalama X kW elektrik tüketmiştir." Bu referans noktası, modelin varyasyonları daha iyi öğrenmesini sağlamış ve tahmin hatasını (MAE) düşürmüştür.

## Hedefi ve Özellikleri Belirleme 

Özellikler (X): Saat, Sıcaklık, Nem, Rüzgar Hızı, Basınç, Görüş Mesafesi ve Baseline (Geçmiş Ortalama) verileri

Hedef (y): Evin Toplam Enerji Tüketimi (House overall [kW])

## Veri Setinin Ayrılması

Veri seti %80 eğitim , %20 test şeklinde ikiye ayrılmıştır.

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Ölçeklendirme

Veri setindeki özelliklerin  farklı birim ve büyüklüklere sahip olması, makine öğrenmesi modellerinin yanlış ağırlıklandırma yapmasına neden olabilir.Özellikle knn için scaling önemlidir.Sayısal değeri büyük olan özellikleri daha önemli saymasını engeller.

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


## KNN Modeli

  Projede ilk olarak ,veri noktaları arasındaki benzerlik ilişkisini temel alan KNN (K-En Yakın Komşu) algoritması kullanılmıştır.

  fit ile eğitim aşaması gerçekleştirilir.Model, eğitim verisindeki tüm noktaların koordinatlarını ve sonuçlarını hafızasına kaydeder.

  Test setindeki veriler alınarak tahmin yapılır.

  Bulunan en yakın 5 komşunun elektrik tüketim değerlerinin ortalaması alınarak tahmin üretilmiştir.

  
    knn_model = KNeighborsRegressor(n_neighbors=5) # 5 en yakın komşuya bakılır
    knn_model.fit(X_train_scaled, y_train)
    knn_preds = knn_model.predict(X_test_scaled) 

## Random Forest 

  Projede ikinci model olarak,topluluk öğrenmesi yöntemi olan Random Forest kullanılmıştır.

  Random Forest birden fazla karar ağacı oluşturur ve her ağaç ayrı tahmin yapar. Bu tahminlerin ortalaması alınır ve sonuç üretilir.

  Model için 25 adet karar ağacı oluşturulmuştur.

  Random Forest ,Knn gib mesafeye bakmaz,kurallara bakar bu yüzden ölçeklendirmeye gerek duyulmuaz.


    rf_model = RandomForestRegressor(n_estimators=25, random_state=42)#100 karar ağacı  olunca çok yavaş çalıştı
    rf_model.fit(X_train, y_train) 
    rf_preds = rf_model.predict(X_test)

## Korelasyon Matrisi

Özellikler arasındaki ilişkiyi gösteren ısı haritası aşağıdaki gibidir:

## Model Performansı

Ortalama Hata(MAE) :Modelin tahminlerinin gerçek değerden ortalama ne kadar saptığını gösterir.

Bu değerin 0'a  yakın olması istenir.


R2 Score:Modelin, verideki değişimi ne kadar iyi açıkladığını gösteren orandır.

 Bu değerin %100'e (1)  yakın olması istenir:

Aşağıda KNN ve Random Forest  modellerinin MAE ve R2 değerleri gösterilmektedir:

     --- KNN PERFORMANSI ---
    KNN MAE (Hata): 0.3361
    KNN R2 (Doğruluk): %44.58

    --- RANDOM FOREST PERFORMANSI ---
    RF MAE (Hata): 0.3153
    RF R2 (Doğruluk): %52.46


BU değerlere göre Random  Forest daha başarılı olmuştur.Birden fazla karar ağacı oluşturarak çalıştığı için verideki dalgalanmaları ve ani değişimleri KNN'e göre  daha iyi öğrenmiştir.









    






