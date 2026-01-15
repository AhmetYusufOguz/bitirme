# Disaster Relief Distribution Optimizer

**Bitirme Projesi - Integrated Location-Routing Problem with Post-Disaster Relief Distribution**

Bu proje, afet sonrası yardım dağıtımı için entegre konum-rotalama problemini çözen hibrit optimizasyon algoritmaları içerir.

## 📋 Proje Özeti

- **Problem**: Afet sonrası depolar nereye açılmalı ve araçlar hangi rotaları izlemeli?
- **Amaçlar**: 
  1. Zaman penceresi ihlal cezasını minimize et (zamanında teslimat)
  2. Operasyonel maliyeti minimize et (depo + araç + taşıma)
- **Çözüm**: Hibrit PA-LRP algoritması (PSO + ACO)

## 🗂️ Proje Yapısı

```
project/
├── core/
│   ├── problem.py          # Problem tanımları (Area, Depot, Problem)
│   ├── solution.py         # Çözüm gösterimi (Solution, Route, ParetoFront)
│   └── validator.py        # Kısıt kontrolleri
├── algorithms/
│   ├── aco.py              # Ant Colony Optimization (rota optimizasyonu)
│   ├── pso.py              # Particle Swarm Optimization (depo ataması)
│   ├── pa_lrp.py           # Hibrit PA-LRP algoritması (ANA ALGORITMA)
│   └── ap.py               # Alternatif hibrit (karşılaştırma için)
├── metrics/
│   └── metrics_all.py      # IGD, HV, QM, SM metrikleri
├── gui/
│   └── gui_complete.py     # Tkinter arayüz + matplotlib görselleştirme
├── main.py                 # Ana çalıştırma scripti
└── README.md               # Bu dosya
```

## 🚀 Kurulum

### Gereksinimler

```bash
pip install numpy matplotlib
```

Python 3.7+ gereklidir.

### Projeyi İndirme

```bash
git clone <your-repo-url>
cd project
```

## 💻 Kullanım

### 1. GUI Modu (Önerilen - Başlangıç için)

En basit kullanım:

```bash
python main.py
```

GUI açılır:
1. **Problem Parameters** bölümünde parametreleri ayarlayın
2. **Generate Problem** butonuna tıklayın
3. Algoritmalardan istediğinizi seçin (PA-LRP önerilir)
4. **Run Optimization** ile çalıştırın
5. **Show Results** ile Pareto frontu ve haritayı görün

### 2. CLI Modu (Testler için)

#### Sadece PA-LRP:
```bash
python main.py --cli --run-pa-lrp
```

#### Tüm algoritmaları karşılaştır:
```bash
python main.py --cli --run-pa-lrp --run-pso --run-aco --run-ap --show-plots
```

#### Özel problem boyutu:
```bash
python main.py --cli --run-pa-lrp \
    --num-areas 100 \
    --num-depots 10 \
    --vehicle-capacity 250 \
    --num-iterations 100
```

#### Grafikleri kaydet (gösterme):
```bash
python main.py --cli --run-pa-lrp --show-plots --no-display
```

### 3. Python Scripti İçinde Kullanım

```python
from core.problem import DisasterReliefProblem
from algorithms.pa_lrp import PALRP

# Problem oluştur
problem = DisasterReliefProblem.generate_random_instance(
    num_areas=50,
    num_depots=5,
    seed=42
)

# PA-LRP ile çöz
solver = PALRP(problem, num_particles=30, num_pso_iterations=50)
pareto_front = solver.solve()

# En iyi çözümü al
best_solution = solver.get_best_solution_by_preference(
    weight_f1=0.6,  # Zaman penceresi önceliği
    weight_f2=0.4   # Maliyet önceliği
)

print(f"Best solution: f1={best_solution.f1_penalty_cost:.2f}, "
      f"f2={best_solution.f2_operational_cost:.2f}")
```

## 📊 Çıktılar

### Konsol Çıktısı
```
==============================================================
PA-LRP Algorithm Started
Problem: 50 areas, 5 depots
==============================================================

--- PSO Iteration 1/50 ---
  Best f1 (penalty): 145.23
  Best f2 (cost): 892.45
  Pareto front size: 8

...

==============================================================
PA-LRP Algorithm Completed
Final Pareto Front Size: 23
==============================================================
```

### Metrik Karşılaştırması
```
Algorithm       IGD          HV           QM       SM         Size
--------------------------------------------------------------------------------
PA-LRP          0.0000       15234.56     1.0000   5.4321     23
PSO             15.2341      12456.78     0.3478   8.9012     12
ACO             22.4567      11234.56     0.2174   12.345     8
AP              18.9876      11987.65     0.2609   10.234     10
```

### Grafikler

1. **Pareto Front Comparison**: Algoritmaların Pareto frontlarını karşılaştırır
2. **Route Map**: En iyi çözümün rota haritası
3. **Convergence History**: Yakınsama grafiği (sadece PA-LRP)

## 🧪 Test Senaryoları

### Küçük Ölçek (Hızlı Test)
```bash
python main.py --cli --run-pa-lrp \
    --num-areas 15 \
    --num-depots 3 \
    --num-iterations 20
```

### Orta Ölçek (Ana Testler)
```bash
python main.py --cli --run-pa-lrp \
    --num-areas 50 \
    --num-depots 5 \
    --num-iterations 50
```

### Büyük Ölçek (Performans Testi)
```bash
python main.py --cli --run-pa-lrp \
    --num-areas 150 \
    --num-depots 20 \
    --num-iterations 100
```

### Tüm Algoritmalarla Karşılaştırma
```bash
python main.py --cli \
    --run-pa-lrp --run-pso --run-aco --run-ap \
    --num-areas 50 --num-depots 5 \
    --show-plots
```

## 📈 Performans Metrikleri

### IGD (Inverted Generational Distance)
- **Ne ölçer**: Elde edilen Pareto frontunun optimal fronta yakınlığı
- **İyi değer**: Düşük (0'a yakın)
- **Formül**: Referans noktalardan elde edilen noktalara ortalama mesafe

### HV (Hyper-Volume)
- **Ne ölçer**: Pareto frontunun kapladığı hacim
- **İyi değer**: Yüksek
- **Formül**: Referans nokta tarafından domine edilen alan

### QM (Quantity Metric)
- **Ne ölçer**: Algoritmanın bulduğu non-dominated çözüm oranı
- **İyi değer**: Yüksek (1'e yakın)
- **Formül**: Birleşik fronttaki çözüm sayısı / toplam

### SM (Spacing Metric)
- **Ne ölçer**: Çözümlerin dağılım düzgünlüğü
- **İyi değer**: Düşük (düzgün dağılım)
- **Formül**: Komşu noktalara mesafelerin standart sapması

## 🎯 Başarı Kriterleri

✅ **Araç Kullanımı**: Ortalama ≥ %85 (validator ile kontrol edin)
✅ **Pareto Kalitesi**: PA-LRP diğerlerinden üstün IGD ve HV değerleri
✅ **Hesaplama Süresi**: 3600 saniye limit
✅ **Çözüm Geçerliliği**: Tüm kısıtlar sağlanmalı

## 🔧 Parametre Ayarlama

### PA-LRP Parametreleri

```python
solver = PALRP(
    problem,
    # PSO parametreleri (depo ataması için)
    num_particles=30,          # Parçacık sayısı
    num_pso_iterations=50,     # PSO iterasyon sayısı
    pso_w=1.0,                 # Atalet katsayısı
    pso_c1=2.0,                # Bilişsel katsayı
    pso_c2=2.0,                # Sosyal katsayı
    
    # ACO parametreleri (rota optimizasyonu için)
    num_ants=30,               # Karınca sayısı
    num_aco_iterations=20,     # ACO iterasyon sayısı
    aco_alpha=1.0,             # Feromon önemi
    aco_beta=0.0,              # Mesafe önemi (0 = sadece feromon)
    aco_rho=0.3,               # Buharlaşma oranı
    aco_q=100.0                # Feromon miktarı
)
```

### Önerilen Ayarlar

**Hızlı Test** (5-10 dakika):
- num_particles=20, num_pso_iterations=30
- num_ants=20, num_aco_iterations=15

**Normal** (15-30 dakika):
- num_particles=30, num_pso_iterations=50
- num_ants=30, num_aco_iterations=20

**Detaylı** (30-60 dakika):
- num_particles=40, num_pso_iterations=100
- num_ants=40, num_aco_iterations=30

## 📝 Rapor İçin Öneriler

### 1. Problem Tanımı
- Matematiksel model (makaledeki Equations 1-16)
- Amaç fonksiyonları ve kısıtlar
- Zaman penceresi kavramı

### 2. Metodoloji
- PA-LRP hibrit yaklaşımı (Fig. 1)
- PSO'nun depo ataması için kullanımı
- ACO'nun rota optimizasyonu için kullanımı

### 3. Deneysel Sonuçlar
- Test senaryoları (küçük, orta, büyük)
- Metrik karşılaştırmaları (Table 4)
- Pareto front grafikleri

### 4. Tartışma
- PA-LRP'nin üstünlüğü
- Hesaplama süreleri
- Pratik uygulanabilirlik

## 🐛 Sorun Giderme

### "No module named 'core'"
```bash
# Proje ana dizininde olduğunuzdan emin olun
cd project
python main.py
```

### Çok yavaş çalışıyor
```bash
# Daha az iterasyon kullanın
python main.py --cli --run-pa-lrp --num-iterations 20
```

### GUI açılmıyor
```bash
# CLI modunu kullanın
python main.py --cli --run-pa-lrp
```

[![DOI](https://zenodo.org/badge/1107586583.svg)](https://doi.org/10.5281/zenodo.18249142)

## 📚 Referanslar

Wei, X., Qiu, H., Wang, D., Duan, J., Wang, Y., & Cheng, T. C. E. (2020). 
*An integrated location-routing problem with post-disaster relief distribution*. 
Computers & Industrial Engineering, 147, 106632.

## 👥 Katkıda Bulunanlar

- Hüseyin Emre Sekanlı
- Ahmet Yusuf Oğuz
- Danışman: Didem Gözüpek

## 📧 İletişim

Sorularınız için: [email protected]

---



**Not**: Bu proje Gebze Teknik Üniversitesi Bilgisayar Mühendisliği Bölümü bitirme projesidir.
