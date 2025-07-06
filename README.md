# Araç Fiyat Tahmin Projesi

Bu proje, araç verilerini kullanarak fiyat tahmini yapan bir makine öğrenmesi uygulamasıdır. Proje, özellik seçimi için Genetik Algoritma (GA), Parçacık Sürüsü Optimizasyonu (PSO) ve Karınca Kolonisi Optimizasyonu (ACO) gibi meta-sezgisel algoritmaları kullanmaktadır.

## Özellikler

- Veri ön işleme ve temizleme
- Gelişmiş özellik mühendisliği
- Meta-sezgisel algoritmalarla özellik seçimi:
  - Genetik Algoritma (GA)
  - Parçacık Sürüsü Optimizasyonu (PSO)
  - Karınca Kolonisi Optimizasyonu (ACO)
- Ensemble özellik seçimi
- Çoklu model eğitimi ve değerlendirme
- Görselleştirme ve raporlama
- Model kaydetme ve yükleme

## Proje Yapısı

```
├── src/
│   ├── Main.py             
│   ├── DataLoader.py        
│   ├── Preprocessing.py    
│   ├── FeatureSelection.py 
│   ├── Train.py           
│   ├── Test.py             
│   └── CreatingScores.py   
├── models/                 
├── outputs/               
├── requirements.txt       
└── README.md               
```

## Kullanım

### Veri Ön İşleme ve Model Eğitimi

```python
python src/Main.py
```

Bu komut:
1. Veriyi yükler
2. Ön işleme adımlarını uygular
3. Özellik seçimini gerçekleştirir
4. Modelleri eğitir ve değerlendirir
5. Sonuçları görselleştirir

### Çıktılar

Proje çalıştırıldığında `outputs/` dizininde şu dosyalar oluşturulur:

- `feature_selection_visualization.png`: Özellik seçimi görselleştirmesi
- `feature_selection_results.csv`: Detaylı özellik seçimi sonuçları
- `model_performance_metrics.txt`: Model performans metrikleri
- `SelectedFeatures.txt`: Seçilen özelliklerin listesi
- `price_correlations.png`: Fiyat korelasyonları görselleştirmesi
- `price_distribution_log_transformed.png`: Fiyat dağılımı görselleştirmesi

## Özellik Seçimi

Proje, üç farklı meta-sezgisel algoritma kullanarak özellik seçimi yapar:

1. **Genetik Algoritma (GA)**
   - Popülasyon tabanlı optimizasyon
   - Çaprazlama ve mutasyon operatörleri
   - Uygunluk fonksiyonu olarak R2 skoru

2. **Parçacık Sürüsü Optimizasyonu (PSO)**
   - Parçacıkların pozisyon ve hız güncellemesi
   - Yerel ve global en iyi pozisyonların takibi
   - Atalet, bilişsel ve sosyal bileşenler

3. **Karınca Kolonisi Optimizasyonu (ACO)**
   - Feromon tabanlı optimizasyon
   - Karıncaların özellik seçim olasılıkları
   - Feromon buharlaşması ve güncelleme

### Ensemble Yaklaşımı

- Her algoritmanın seçtiği özellikler birleştirilir
- En az 2 algoritma tarafından seçilen özellikler final setine dahil edilir
- Hiçbir özellik 2 veya daha fazla algoritma tarafından seçilmezse, GA'nın seçtiği özellikler kullanılır

## Modeller

Proje iki farklı regresyon modeli kullanır:

1. **Linear Regression**
   - Basit doğrusal regresyon
   - Tüm özelliklerin doğrusal kombinasyonu

2. **Ridge Regression**
   - L2 regularizasyonu ile doğrusal regresyon
   - Aşırı öğrenmeyi önlemek için ceza terimi

## Performans Metrikleri

Her model için şu metrikler hesaplanır:
- RMSE (Root Mean Square Error)
- R2 Score
- MAE (Mean Absolute Error)

## Görselleştirmeler

Proje, çeşitli görselleştirmeler sunar:
- Özellik seçimi ısı haritası
- Toplam oy sayısı çubuk grafiği
- Fiyat korelasyonları
- Fiyat dağılımı
- Model performans karşılaştırmaları



