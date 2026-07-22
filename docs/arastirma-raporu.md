# Lokalde LLM Çalıştırma — Araştırma Raporu

## 1. Giriş

Bu çalışmanın amacı, internet bağlantısına ihtiyaç duymadan tamamen lokal
çalışan bir dil modeli altyapısının kurulabilirliğini test etmek, gereken
donanımı ve maliyeti ortaya koymak, ve elde edilen performansı ölçülebilir
sayılarla raporlamaktır.

Çalışma iki çıktı üretti:

1. **Ölçüm altyapısı** — iki güncel orta boy modeli aynı koşullarda kıyaslayan,
   tekrarlanabilir bir benchmark aracı.
2. **Çalışan prototip** — şirket içi İK dokümanlarını yanıtlayan, dışarıya veri
   göndermeyen bir soru-cevap (RAG) sistemi.

Rapordaki tüm performans sayıları, bu altyapıyla kendi test donanımımda
ölçülmüştür; hiçbiri üretici beyanı veya tahmin değildir.

### Test Donanımı

| Bileşen | Değer |
|---|---|
| İşlemci | Apple M4 Pro (14 çekirdek) |
| Bellek | 48 GB **unified memory** |
| Depolama | NVMe SSD |
| İşletim sistemi | macOS 26 (Darwin 25.3) |
| Çalıştırma aracı | Ollama 0.32.1 |

---

## 2. LLM ve Embedding Nedir?

### Büyük Dil Modeli (LLM)

Çok büyük miktarda metin üzerinde eğitilmiş, kendisine verilen metnin
devamında hangi kelimenin gelme olasılığının en yüksek olduğunu tahmin eden
yapay sinir ağıdır. Bu basit görev yeterince büyük ölçekte öğrenildiğinde
model; soru yanıtlama, özetleme, çeviri ve kod yazma gibi işleri yapabilir
hâle gelir.

*Örnek:* Qwen3.5, Gemma 4, Llama, Mistral.

### Embedding Modeli

Metni, anlamını sayısal olarak temsil eden bir vektöre (sayı dizisine)
dönüştüren modeldir. Anlamca yakın iki metin, vektör uzayında birbirine yakın
konumlanır. Bu sayede kelime birebir eşleşmese bile anlamsal arama yapılabilir.

*Örnek:* Bu projede kullanılan `qwen3-embedding:0.6b`, her metin parçasını
**1024 boyutlu** bir vektöre çevirir. "Yıllık izin hakkım kaç gün?" sorusu ile
dokümandaki "Hizmet süresi 1–5 yıl arası: 16 iş günü" tablosu hiçbir ortak
kelime içermemesine rağmen vektör uzayında yakın çıkar ve doğru parça bulunur.

### Neden ikisi birlikte?

RAG (Retrieval-Augmented Generation) mimarisinde embedding modeli **doğru
bilgiyi bulur**, LLM ise **bulunan bilgiyi anlaşılır bir cevaba dönüştürür**.
LLM'in kendi eğitim verisinde şirketinizin izin politikası yoktur; embedding
katmanı olmadan cevap uydurmak zorunda kalır.

---

## 3. Lokal Kurulumun Avantajları ve Dezavantajları

### Avantajlar

**Veri gizliliği.** Şirket içi doküman, müşteri verisi veya personel bilgisi
hiçbir noktada cihazdan çıkmaz. Bu projede geliştirilen sistemde tüm trafik
`localhost` üzerindedir; sistem internet bağlantısı kapalıyken de tam
işlevseldir. KVKK ve benzeri düzenlemeler açısından, veriyi hiç göndermemek
en güçlü uyum argümanıdır.

**Kesintisiz erişim.** Sağlayıcı kesintisi, kota limiti veya fiyat değişikliği
sistemin çalışmasını etkilemez.

**Maliyet öngörülebilirliği.** İstek başına ödeme yerine tek seferlik donanım
maliyeti vardır; gider, hacimden bağımsız sabit bir kaleme dönüşür. Dikkat:
öngörülebilir olması *ucuz* olması anlamına gelmez — bu çalışmanın fiyat
araştırması, bu ölçekte lokal kurulumun buluttan ucuz **olmadığını** gösterdi
(Bölüm 4.4).

**Düşük gecikme ve kolay entegrasyon.** Yerel ağdaki veri tabanları ve iç
sistemlerle ağ turu olmadan çalışır.

### Dezavantajlar — dürüst değerlendirme

Raporun tarafsız olması için bunların da belirtilmesi gerekir:

- **Kalite farkı.** 9–12 milyar parametreli lokal modeller, en büyük ticari
  modellerin karmaşık akıl yürütme performansını yakalayamaz. Basit soru-cevap
  ve doküman özetlemede fark küçüktür; çok adımlı analizde belirgindir.
- **Başlangıç maliyeti.** Donanım peşin ödenir ve bu ölçekte geri dönmez.
  Aynı sınıf bir modeli bulutta çalıştırmak yılda ~600 TL tutarken, lokal
  kurulumun yalnızca elektriği bunun birkaç katıdır (Bölüm 4.4). Lokalin
  gerekçesi tasarruf değil, verinin dışarı çıkmamasıdır.
- **Bakım yükü.** Model güncellemeleri, sürüm uyumluluğu ve donanım arızası
  kurum içi sorumluluktadır.
- **Eşzamanlılık sınırı.** Tek bir iş istasyonu aynı anda sınırlı sayıda
  kullanıcıya hizmet verebilir.

---

## 4. Donanım Gereksinimleri

### 4.1 Model Boyutu ve Bellek İlişkisi

Modelin ağırlıkları belleğe yüklenmek zorundadır. Kaba hesap:

```
Gereken bellek (GB) ≈ Parametre sayısı (milyar) × Byte/parametre
```

| Veri tipi | Parametre başına | 9B model için |
|---|---|---|
| FP32 | 4 byte | ~36 GB |
| FP16 / BF16 | 2 byte | ~18 GB |
| INT8 (q8_0) | ~1 byte | ~9 GB |
| 4-bit (q4_K_M) | ~0,5 byte | ~5 GB |

Buna ek olarak **KV cache** (bağlam belleği) gerekir; uzun bağlam
pencerelerinde bu miktar birkaç GB'a ulaşabilir. Pratikte model boyutunun
üzerine %20–40 pay bırakmak gerekir.

Ölçüm bu payı somutlaştırıyor. 32K bağlam penceresiyle çalışırken Ollama'nın
bildirdiği bellek:

| Model | Disk boyutu | Bellekte (32K bağlam) | Fark |
|---|---|---|---|
| `qwen3.5:9b` | 6,6 GB (6,14 GiB) | 6,29 GiB | +%2 |
| `gemma4:12b` | 7,6 GB (7,08 GiB) | 7,85 GiB | +%11 |

Yani 4-bit quantize edilmiş bu modellerde ek yük, tabloda gösterilen kaba
hesabın öngördüğünden düşük kaldı. Ancak bu sayı **bağlam uzunluğuna bağlıdır**:
KV cache, pencere büyüdükçe doğrusal büyür. Her iki model de 256K bağlamı
destekliyor; ölçüm Ollama'nın varsayılanı olan 32K ile yapıldı. Tam pencere
kullanılsaydı KV cache payı kat kat artardı. Donanım planlarken sorulması
gereken soru "model kaç GB?" değil, **"model + hedeflenen bağlam kaç GB?"**
olmalıdır.

> Bu bölümdeki bellek sayıları Ollama'nın kendi raporundan alınmıştır; süreç
> RSS'i neden bu iş için yanıltıcı olduğu Bölüm 9.4'te ayrıntılı anlatılmıştır.

### 4.2 Apple Silicon: Unified Memory

NVIDIA sistemlerinde VRAM (ekran kartı belleği) ve sistem RAM'i ayrıdır; model
VRAM'e sığmazsa performans ciddi düşer. Apple Silicon'da ise **tek bir bellek
havuzu** vardır ve GPU bu havuzun büyük bölümüne doğrudan erişir.

Pratik sonucu: 48 GB RAM'e sahip bir Mac, 24 GB VRAM'li bir RTX 4090'dan daha
büyük modelleri belleğe sığdırabilir. Buna karşılık ham hesaplama gücü
(FLOPS) ve bellek bant genişliği üst seviye NVIDIA kartlarının gerisindedir.

> **Kural:** Apple Silicon'da "kaç GB VRAM" diye sorulmaz; toplam RAM'in ne
> kadarını modele ayırabileceğiniz sorulur. macOS varsayılan olarak toplam
> belleğin yaklaşık %75'ini GPU'ya tahsis edebilir.

### 4.3 Sadece CPU ile çalıştırma

Mümkündür ancak token üretim hızı GPU'ya kıyasla tipik olarak 5–10 kat düşer.
Etkileşimli bir asistan için kullanışsız, gece toplu iş (batch) çalıştırmak
için kabul edilebilir.

### 4.4 Tedarik ve maliyet analizi (Türkiye pazarı)

Staj dokümanının 4.3 maddesi, donanım fiyatlarının oynak olması nedeniyle
maliyet araştırmasının **çalışma sırasında** yapılmasını istiyor. Aşağıdaki
fiyatlar **22 Temmuz 2026** tarihinde toplanmıştır ve gün içinde değişebilir.
Kur: TCMB 21.07.2026 bülteni, 1 USD = 47,20 TL.

> **2026'nın özel koşulu.** Küresel bir DRAM/GDDR7 kıtlığı sürüyor ve bu
> bölümdeki her sayıyı etkiliyor: 32 GB'lık bir DDR5 kiti 2024–25'te
> 4.500–5.500 TL bandındayken bugün 22.000–27.500 TL; Apple, Mac mini'nin
> 32 GB ve 64 GB seçeneklerini Türkiye'de satıştan kaldırdı; NVIDIA'nın
> 24 GB'lık RTX 50 Super serisi süresiz ertelendi. Yani "biraz daha bellek al"
> tavsiyesi bu yıl alışılmadık ölçüde pahalı.

#### Apple Silicon — Apple Türkiye mağazası

| Ürün | Bellek | Fiyat |
|---|---|---|
| Mac mini M4 | 16 GB | 64.499 TL |
| Mac mini M4 | 24 GB | 76.999 TL |
| Mac mini M4 Pro | 24 GB | 104.999 TL |
| Mac mini M4 Pro | 48 GB | 161.249 TL |
| Mac Studio M4 Max | 36 GB | 159.999 TL |

Yetkili perakendecilerde (epey/Akakçe taraması) aynı ürünler belirgin biçimde
daha ucuz görünüyor — örneğin Mac mini M4 24 GB için 50.500–64.600 TL. Bu fark
Türkiye'de olağandır ancak **garanti ve fatura koşulları farklı olduğu için iki
fiyat aynı satırda gösterilmemelidir.** Perakendede hâlâ M4 nesli MacBook stoğu
bulunuyor ve fiyat/performans açısından ciddi bir seçenek.

#### NVIDIA

| Kart | VRAM | Fiyat |
|---|---|---|
| RTX 5060 Ti 16GB | 16 GB | 36.000 – 40.000 TL |
| RTX 5090 | 32 GB | 172.800 – 200.000 TL |
| RTX 5070 Ti Super / 5080 Super | 24 GB | **piyasada yok** — süresiz ertelendi |

Buradaki en önemli bulgu bir fiyat değil, bir **boşluk**: Temmuz 2026 itibarıyla
tüketici tarafında 16 GB ile 32 GB arasında ara segment yoktur. 16 GB yetmeyen
bir iş yükü, doğrudan 172.000 TL'lik karta sıçramak zorunda kalır. Apple
tarafında ise 24, 36 ve 48 GB ara basamakları mevcuttur — bu, unified memory
mimarisinin bu ölçekteki en somut pratik avantajıdır.

#### Ekran kartı tek başına bir sistem değildir

Maliyet hesabının en sık yapılan hatası budur. 38.000 TL'lik bir RTX 5060 Ti,
çalışır bir iş istasyonu anlamına gelmez; üzerine işlemci, 32 GB sistem RAM'i
(bu yıl tek başına 22.000 TL+), anakart, güç kaynağı, kasa ve NVMe SSD gerekir.
Türkiye'de hazır satılan sistemler üzerinden bakıldığında:

| Konfigürasyon | Fiyat |
|---|---|
| Ryzen 5 7500F + RTX 5060 Ti 16GB | 48.329 TL |
| Ryzen 7 5700X + 32 GB RAM + RTX 5060 Ti 16GB | 61.499 TL |

Yani ekran kartı, toplam sistemin ancak %60–75'idir. Aynı bellek sınıfındaki
Mac mini M4 24 GB ile karşılaştırıldığında ikisi benzer bantta kalıyor; fark
kurulum emeği ve güç tüketiminde ortaya çıkıyor.

#### Elektrik

Mac mini M4 Pro için Apple'ın resmî ölçümü: boşta 5 W, azami 140 W. NVIDIA
sistemleri için tüketim **ölçülmemiştir**, kart incelemelerinden tahmin
edilmiştir (5060 Ti sistemi ~90 W boşta / ~300 W yük). Günün %30'unda üretim
yapıldığı varsayımıyla, 7/24 açık bırakılan bir kurulum için yıllık gider:

| Sistem | Yıllık tüketim | Yıllık maliyet |
|---|---|---|
| Mac mini M4 Pro | ~399 kWh | ~1.700 TL |
| RTX 5060 Ti sistemi | ~1.340 kWh | ~5.800 TL |
| RTX 5090 sistemi | ~2.444 kWh | ~10.600 TL |

kWh birim fiyatı 4,32 TL alınmıştır (mesken, ikinci kademe). **Uyarı:** EPDK'nın
birincil tarife tablosuna doğrudan erişilemedi; ikincil kaynaklar 2,59–4,32
TL/kWh arasında çelişiyor. Tarife çeyreklik güncellendiği için teslim öncesi
`epdk.gov.tr` üzerinden teyit edilmelidir.

#### Bulut API ile karşılaştırma — ve rahatsız edici sonuç

Amortisman hesabı ancak **doğru karşılaştırma** yapılırsa anlamlıdır. Lokalde
çalıştırılan model 9–12B sınıfındadır; onu en pahalı ticari modellerle
kıyaslamak lokal kurulumu haksız yere kârlı gösterir. Adil kıyas, *aynı sınıf
açık modelin bulutta ne tuttuğudur.*

50 kişilik bir şirket, günde 200 soru (aylık 4.400 sorgu), sorgu başına ~4.000
girdi + ~500 çıktı token varsayımıyla:

| Seçenek | Aylık | Yıllık |
|---|---|---|
| **Gemma 3 12B — bulut (DeepInfra)** | ~47 TL | ~564 TL |
| Ucuz sınıf ticari model (Flash/Lite sınıfı) | ~360–730 TL | ~4.400–8.700 TL |
| Orta sınıf ticari model (Haiku/Luna sınıfı) | ~1.350–1.450 TL | ~16.000–17.500 TL |
| Üst sınıf ticari model (Sonnet/Opus sınıfı) | ~4.000–6.800 TL | ~48.000–81.000 TL |

Sonuç açıktır ve raporun başındaki beklentiyle çelişir: **aynı sınıf bir model
bulutta yılda ~600 TL tutarken, lokal kurulumun yalnızca elektriği yılda
1.700–5.800 TL'dir.** Donanımın 48.000–160.000 TL'lik peşin maliyeti hiç hesaba
katılmasa bile lokal kurulum bu kıyasta **hiçbir kullanım yoğunluğunda amorti
etmez** — hacim arttıkça fark kapanmaz, açılır.

Lokal kurulum ancak *üst sınıf* bir ticari modelin yerine geçtiği varsayılırsa
maliyet olarak anlamlı hale gelir (bu senaryoda RTX 5060 Ti sistemi ~1,5 yılda,
Mac mini M4 Pro 48 GB ~3,5 yılda başa baş gelir). Ancak 12B'lik bir modelin
en büyük ticari modelleri ikame ettiğini varsaymak, Bölüm 3'te belirtilen kalite
farkı nedeniyle teknik olarak savunulamaz.

#### Karar: gerekçe maliyet değil, veri egemenliğidir

Bu çalışmanın maliyet tarafındaki en dürüst çıktısı şudur: **lokal LLM kurulumu
bu ölçekte bir tasarruf kalemi değil, ölçülebilir bir gizlilik primidir.**
Kurumsal karar şöyle kurulmalıdır: "lokal daha ucuz" değil, "İK dokümanlarının,
özlük dosyalarının ve maaş verilerinin kurum dışına hiç çıkmaması karşılığında
yılda birkaç bin TL fazladan ödeniyor." KVKK yükümlülüğü olan bir veri
kümesinde bu prim kolaylıkla haklı çıkarılabilir; genel amaçlı bir sohbet
asistanında çıkarılamaz.

Maliyet gerekçesi yalnızca üç koşul birlikte sağlanırsa kurulabilir:
(a) kıyaslanan bulut modeli üst sınıfsa, (b) donanım en az üç yıl
kullanılacaksa, (c) donanım yalnız bu iş için değil başka yüklerle de
paylaşılacaksa.

> **Kaynak ve güven notu.** Apple fiyatları `apple.com/tr` mağazasından,
> NVIDIA fiyatları Akakçe/Cimri listelerinden, hazır sistem fiyatları
> Trendyol/Technopat ilanlarından alınmıştır (hepsi 22.07.2026). Şu kalemler
> **doğrulanamamıştır ve tahmin edilmemiştir:** RTX 5090'lı tam sistem
> maliyeti, tekil CPU/anakart/PSU fiyatları, NVIDIA sistemlerinin gerçek güç
> tüketimi, EPDK'nın birincil tarife tablosu. Bu kalemler rapora tahmin olarak
> değil, "doğrulanamadı" olarak girmiştir.

---

## 5. Quantization (Kuantizasyon)

Model ağırlıkları normalde 16 veya 32 bit ondalıklı sayılarla saklanır.
Quantization, bu sayıları daha az bit ile temsil ederek bellek ihtiyacını ve
işlem yükünü düşürme işlemidir.

### GGUF format önekleri

| Format | Anlamı |
|---|---|
| `q8_0` | 8-bit, kaliteye en yakın, bellek kazancı sınırlı |
| `q5_K_M` | 5-bit, karma hassasiyet |
| `q4_K_M` | **4-bit karma** — kritik katmanlar daha yüksek bitte tutulur; kalite/boyut dengesinin standardı |
| `q4_0` | Düz 4-bit, daha eski ve daha kayıplı |

### Kazanç ve kayıp

- **Kazanç:** Bellek tüketimi FP16'ya göre yaklaşık 4'te 1'e iner, model
  hızlanır ve daha mütevazı donanımda çalışır.
- **Kayıp:** Dil ve mantık yeteneğinde küçük düşüş olur. `q4_K_M` seviyesinde
  bu kayıp günlük kullanımda çoğunlukla fark edilmez.

**Bu projedeki uygulama:** Her iki model de `q4_K_M` seviyesinde kullanıldı.
Bu bir zorunluluk değil, **ölçüm geçerliliği şartıdır** — farklı quantization
seviyelerindeki iki modeli kıyaslamak, farklı ayarlardaki iki motoru
kıyaslamak gibidir; sonuç anlamsız olur.

### Dequantization

Quantize edilmiş ağırlıkların işlem sırasında geçici olarak daha yüksek
hassasiyete geri açılmasıdır. Bellekte küçük kalır, hesaplama anında açılır.

---

## 6. Açık Kaynak Ekosistemi ve Araçlar

### 6.1 Çalıştırma araçları

| Araç | Tür | Değerlendirme |
|---|---|---|
| **Ollama** | CLI + HTTP API | Tek komutla model indirme, otomatik GPU/CPU dağılımı, OpenAI uyumlu API. **Bu projede kullanıldı.** |
| **LM Studio** | GUI | Grafik arayüz, model keşfi ve deneme için elverişli; yerel API sunucusu açabilir. Teknik olmayan kullanıcılar için en erişilebilir seçenek. |
| **llama.cpp** | C/C++ motor | Ollama ve LM Studio'nun altında yatan motor. Doğrudan kullanımı en fazla kontrolü verir, en fazla emek ister. |
| **vLLM** | Sunum motoru | Yüksek eşzamanlılık ve toplu çıkarım için; tek kullanıcılı senaryoda gereksiz karmaşıklık. |

**Ollama'nın seçilme gerekçesi:** HTTP API'si sürecin ölçülmesine izin veriyor —
her yanıtta `eval_count` ve `eval_duration` sayaçlarını döndürüyor. Benchmark'ın
token/s değerleri bu sayaçlardan hesaplandı; duvar saati ölçümü kullanılsaydı
ağ ve akış (streaming) gecikmesi sonuca karışırdı.

> **Not:** Ollama 0.19 sürümünden itibaren Apple Silicon'da MLX motorunu
> kullanmaktadır. Bazı modellerin `-mlx` etiketli varyantları vardır ve bunlar
> yalnızca Apple donanımında çalışır. Bu çalışmada, Windows/CUDA ortamına
> taşınabilirliği korumak için **GGUF** varyantları tercih edilmiştir.

### 6.2 Uygulama geliştirme çerçeveleri

**LangChain / LlamaIndex / Haystack** — RAG hattı kurmayı kolaylaştıran
kütüphanelerdir.

Bu projede **bilinçli olarak kullanılmadılar.** Gerekçe: parçalama, embedding,
vektör arama ve prompt kurulumu toplamda ~400 satır kod tutuyor. Bir çerçeve
kullanmak bu satırları gizler ama ölçüm noktalarına erişimi de zorlaştırır ve
öğrenme amacını zayıflatır. Üretim ölçeğinde (çoklu veri kaynağı, ajan
davranışı) tercih farklı olurdu.

### 6.3 Model kaynakları

Orijinal modeller **Hugging Face** üzerinde yayınlanır. Quantize edilmiş GGUF
sürümler için `bartowski`, `unsloth` ve `lmstudio-community` hesapları takip
edilir. Ollama kütüphanesi (`ollama.com/library`) bu süreci soyutlar; model
etiketleri doğrudan `ollama pull` ile çekilebilir.

---

## 7. Model Seçimi

Staj dokümanında örnek olarak **Llama 3 8B** ve **Gemma 2 9B** verilmişti.
Bu modeller 2024 tarihlidir ve Temmuz 2026 itibarıyla eskimiştir; Ollama
kütüphanesinde daha güçlü ve aynı boyut sınıfında halefleri bulunmaktadır.
Rapora güncel olmayan bir kıyas koymamak için model listesi yeniden
araştırılmış ve etiketlerin varlığı doğrulanmıştır.

### Seçilen modeller

| Model | Üretici | Parametre | Boyut | Bağlam | Diller |
|---|---|---|---|---|---|
| `qwen3.5:9b` | Alibaba | 9B (dense) | 6,6 GB | 256K | 201 dil |
| `gemma4:12b` | Google | 12B (dense) | 7,6 GB | 256K | 140+ dil |
| `qwen3-embedding:0.6b` | Alibaba | 0,6B | 639 MB | 32K | 100+ dil |

### Seçim gerekçeleri

**Neden bu iki sohbet modeli?** Kıyasın anlamlı olması için modellerin
birbirinden farklı olması gerekir. Bu ikili iki eksende ayrışıyor: farklı
üretici (Alibaba / Google) ve farklı dikkat (attention) mimarisi —
qwen3.5 ağırlıklı olarak doğrusal dikkat kullanan hibrit bir yapıya sahipken
gemma4 tam dikkat kullanıyor. İkisi toplamda ~14 GB tuttuğu için 48 GB bellekte
aynı anda ayakta tutulup canlı A/B karşılaştırması yapılabiliyor.

**Neden bu embedding modeli?** Sistem Türkçe dokümanlar üzerinde çalışacağı
için çok dilli destek zorunluydu. Adaylar arasında:

| Model | Boyut | Bağlam | MMTEB |
|---|---|---|---|
| `qwen3-embedding:0.6b` | 639 MB | **32K** | **64,3** |
| `bge-m3:567m` | 1,2 GB | 8K | 59,6 |
| `embeddinggemma:300m` | 622 MB | 2K | 65,4 |
| `nomic-embed-text` | 274 MB | 8K | (yalnızca İngilizce) |

`embeddinggemma`'nın 2K bağlam sınırı Türkçe için ciddi bir kısıttır: Türkçe
sondan eklemeli bir dil olduğu için aynı içerik İngilizceye göre yaklaşık
%20–30 daha fazla token tutar. `qwen3-embedding:0.6b` hem geniş bağlam hem
yüksek çok dilli skor sunduğu için seçildi; ayrıca `bge-m3` ile aynı 1024
boyutu ürettiğinden, vektör şeması değiştirilmeden değiştirilebilir.

> **Dürüstlük notu:** MMTEB skorları çok dilli ortalamalardır. Türkçe'ye özel
> ayrıştırılmış kamuya açık bir sıralama bulunamamıştır; seçim çok dilli
> ortalamalardan çıkarım yoluyla yapılmıştır. Türkçeye özel eğitilmiş
> `TurkEmbed4Retrieval` gibi modeller literatürde mevcuttur ancak Ollama
> kütüphanesinde bulunmadığından bu çalışmanın kapsamı dışında kalmıştır.

---

## 8. Benchmark Metodolojisi

Bir kıyaslamanın değeri, ölçümün adil olmasına bağlıdır. Uygulanan kurallar:

1. **Aynı prompt'lar.** Her model, baytı baytına aynı soruları alır
   (`bench/prompts.yaml`).
2. **Aynı üretim ayarları.** `temperature=0`, `seed=42`, `num_predict=1024`
   her çağrıda açıkça gönderilir.
3. **Reasoning modu açıkça kapatılır.** Bu kritik bir noktadır: `qwen3.5`
   varsayılan olarak "düşünme" modunu **açık**, `gemma4` ise **kapalı**
   getirir. Varsayılana güvenilseydi, biri düşünme token'ları üretirken diğeri
   üretmeyecek ve hız ölçümü geçersiz olacaktı. Her iki modele de `think=false`
   açıkça gönderilir.
4. **Aynı quantization.** İkisi de `q4_K_M`.
5. **Isınma turu.** İlk çağrı modeli belleğe yükler; bu maliyet ayrı ölçülür ve
   ilk soruya yüklenmez.
6. **Ölçüm kaynağı.** token/s değeri Ollama'nın `eval_count`/`eval_duration`
   sayaçlarından hesaplanır — duvar saatinden değil.
7. **Bellek izolasyonu.** Her modelden önce Ollama'ya hangi modellerin bellekte
   olduğu sorulur ve yabancı olanlar boşaltılır (embedding modeli hariç; RAG'ın
   ona her soruda ihtiyacı var). Koşu boyunca bir başkasının yüklenip
   yüklenmediği izlenir ve yüklenirse çıktıya uyarı basılır. Bu önlemin neden
   şart olduğu Bölüm 9.4'te anlatılmaktadır.
8. **Tekrarlanan koşu, bir kısmı ters sırada.** Tek koşu, koşular arası
   değişkenliği göremediği için "A modeli B'den hızlı" iddiasını taşıyamaz.
   Ayrıca modeller her koşuda aynı sırada ölçülürse ikinci model sistematik
   olarak dezavantajlı duruma düşebilir; iki koşuda sıra ters çevrilerek bu
   sınanmıştır. Toplam altı koşu alındı, dördü temiz (Bölüm 9.1).
9. **Token ağırlıklı ortalama.** Hız `Σ token / Σ süre` olarak hesaplanır,
   vaka başına tok/s değerlerinin düz ortalaması olarak değil (gerekçe:
   Bölüm 9.2).

### Ölçülen metrikler

| Metrik | Tanım |
|---|---|
| **token/s** | `Σ token / Σ üretim süresi` — hızın ana göstergesi. Vaka ortalaması, standart sapma, en düşük ve en yüksek değer ayrıca kaydedilir |
| **TTFT** | İlk token'a kadar geçen süre — algılanan gecikme (medyan) |
| **Bellek** | Ollama'nın `/api/ps` raporundaki model boyutu. Süreç RSS'i de kaydedilir ama karşılaştırmada kullanılmaz (Bölüm 9.4) |
| **Retrieval** | Belge arama süresi — koşu genelinde tek sayı; embedding modeline aittir |
| **Kalite** | Cevapta beklenen bilgilerin bulunup bulunmadığı |
| **Kaynağa sadakat** | Bilgi dokümanda yokken uydurmama oranı |

### Kalite neden anahtar kelime ile ölçülüyor?

Cevap kalitesini bir başka LLM'e puanlatmak (LLM-as-judge) yaygın bir
yöntemdir, ancak burada tercih edilmedi: hakem modelin kendi değişkenliği,
zaten modelleri ölçmeye çalışan bir deneye ikinci bir belirsizlik kaynağı
ekler. Bunun yerine her sorunun cevabında bulunması gereken somut değerler
(`16`, `750`, `2 katı`) önceden tanımlandı. Ölçüm mekanik ve tekrarlanabilir;
üçüncü bir kişi aynı sayıları yeniden üretebilir.

Türkçe için normalizasyon gerekti: `İzmir` ve `izmir` karşılaştırması Türkçe
nokta problemi nedeniyle özel işlem ister; ayrıca aksan katlama uygulanır.

---

## 9. Sonuçlar

### 9.1 Ölçüm düzeni

Her koşu 14 soru içerir: 3 saf üretim (RAG'sız) ve 11 RAG sorusu — bunların 3'ü
cevabı dokümanlarda bilinçli olarak **bulunmayan** kontrol sorularıdır. Toplam
altı koşu alındı:

| Koşu | Dosya | Model sırası | Eşik | Durum |
|---|---|---|---|---|
| 1 | `run4-clean.json` | qwen3.5 → gemma4 | 0,52 | temiz |
| 2 | `run5-reversed.json` | gemma4 → qwen3.5 | 0,52 | temiz |
| 3 | `run6-repeat.json` | qwen3.5 → gemma4 | 0,52 | temiz |
| 4 | `run8-clean.json` | qwen3.5 → gemma4 | 0,46 | **kirlendi** |
| 5 | `run9-reversed.json` | gemma4 → qwen3.5 | 0,46 | gemma yarısı kirlendi |
| 6 | `run10-repeat.json` | qwen3.5 → gemma4 | 0,46 | temiz |

Koşular arasında benzerlik eşiği 0,52'den 0,46'ya çekildi; gerekçesi proje
raporunun 5. bölümündedir. Eşik **üretim hızını etkilemez** — yalnızca sorunun
modele ulaşıp ulaşmayacağına karar verir, model çağrıldıktan sonraki hiçbir
şeye dokunmaz. Bu nedenle rapor iki kaynağı ayırır:

- **Hız, gecikme ve bellek** sayıları 1-3 numaralı temiz koşulardan gelir.
- **Reddetme davranışı** sistemin bugünkü ayarını taşıyan 4-6 numaralı
  koşulardan gelir (Bölüm 9.6).

6. koşu — eşik değişikliğinden sonraki tek tamamen temiz koşu — hız sayılarını
bağımsız olarak doğruladı: `qwen3.5` 38,07 ve `gemma4` 27,79 tok/s, yani ilk üç
koşunun aralığında. 4 ve 5 numaralı koşuların neden kullanılamadığı Bölüm
9.4'te anlatılıyor; kısaca, ölçüm sırasında makinede başka bir uygulama
23 GB'lık bir model çalıştırdı ve harness bunu kendi uyarı mekanizmasıyla
yakaladı.

İkinci ve beşinci koşularda model sırası bilinçli olarak ters çevrildi.
Gerekçe: iki model art arda ölçüldüğünde ikinci model, birincinin bıraktığı
bellek baskısı ve yaklaşık beş dakikalık kesintisiz üretimin ısıttığı bir
işlemci üzerinde çalışır. Sıra bir yanlılık üretiyorsa, ters koşuda işaret
değiştirmesi gerekir.

**Üretmedi.** `qwen3.5` ilk sırada 37,19 ve 36,99 tok/s, ikinci sırada 37,86
tok/s ölçüldü; `gemma4` ikinci sırada 27,58 ve 27,94, ilk sırada 27,42 tok/s.
Fark her iki modelde de %2'nin altında ve yönü tutarsız — yani sıra etkisi bu
ölçüm düzeninde gürültünün içinde kalıyor.

### 9.2 Üretim hızı

| Model | tok/s (koşu 1-3 ort.) | Standart sapma | Koşu değerleri |
|---|---|---|---|
| **`qwen3.5:9b`** | **37,35** | 0,46 | 37,19 · 37,86 · 36,99 |
| `gemma4:12b` | 27,65 | 0,27 | 27,58 · 27,42 · 27,94 |

`qwen3.5:9b`, `gemma4:12b`'den **yaklaşık %35 daha hızlı** üretim yapıyor.
Üç koşunun standart sapması modeller arası farkın 1/20'sinden küçük olduğu
için bu fark ölçüm gürültüsüyle açıklanamaz.

Farkın iki kaynağı var: parametre sayısı (9,7B'ye karşı 12B) ve mimari —
`qwen3.5` ağırlıklı olarak doğrusal dikkat kullanan hibrit bir yapıya sahip,
`gemma4` ise tam dikkat kullanıyor.

Bu üç koşunun ardından, eşik değişikliğiyle birlikte alınan dördüncü temiz koşu
(6 numaralı, `run10-repeat.json`) aynı sayıları bağımsız olarak yeniden üretti:
`qwen3.5` 38,07 ± 0,32 ve `gemma4` 27,79 ± 0,62 tok/s. Dört temiz koşuda
`qwen3.5` 36,99–38,07, `gemma4` 27,42–27,94 aralığında kaldı; iki modelin
aralıkları hiçbir noktada kesişmiyor.

> **Metodoloji notu — token ağırlıklı ortalama.** tok/s değeri
> `Σ üretilen token / Σ üretim süresi` olarak hesaplanır; her sorunun tok/s
> değerinin düz ortalaması alınmaz. Düz ortalama, 19 token'lık kısa bir cevapla
> 400 token'lık uzun bir cevabı eşit ağırlıklandırır — oysa kısa cevapta ölçülen
> hız büyük ölçüde prompt işleme maliyetidir, kullanıcının deneyimlediği üretim
> hızı değil. Bu ölçüm setinde iki yöntem arasındaki fark %2'nin altında kaldı,
> ancak kısa cevapların ağırlıklı olduğu bir sette ikisi ciddi biçimde ayrışır.

### 9.3 Gecikme ve yükleme

| Model | TTFT (medyan) | TTFT aralığı | Model yükleme |
|---|---|---|---|
| **`qwen3.5:9b`** | **1.941 ms** | 1.761 – 1.951 ms | 3,8 – 11,7 s |
| `gemma4:12b` | 2.978 ms | 2.504 – 3.356 ms | 5,0 – 8,9 s |

TTFT (ilk token'a kadar geçen süre), kullanıcının "sistem dondu mu?" diye
düşündüğü süredir ve algılanan hız açısından toplam süreden daha belirleyicidir.
`qwen3.5` burada da yaklaşık bir saniye öndedir.

Yükleme süreleri koşudan koşuya iki-üç kat değişkenlik gösterdi. Bu modelin
değil işletim sisteminin özelliğidir: model dosyası dosya sistemi önbelleğinde
sıcaksa yükleme hızlı, disk'ten okunuyorsa yavaştır. Uç örnek 6 numaralı
koşudur: model bir önceki koşudan hâlâ bellekte olduğu için yükleme 0,57
saniyeye indi — yani ölçülen şeyin modelle ilgisi yok, makinenin o anki
hâliyle ilgisi var. Bu nedenle yükleme süresi
model karşılaştırmasında kullanılmamıştır; yalnızca "ilk soru pahalıdır,
sonrakiler değildir" gözlemini belgelemek için raporlanmaktadır.

### 9.4 Bellek — ve ölçümün nasıl yanıltabildiği

| Model | Ollama'nın bildirdiği | Model dosyası |
|---|---|---|
| **`qwen3.5:9b`** | **6,29 GB** | 6,6 GB |
| `gemma4:12b` | 7,85 GB | 7,6 GB |

Bu bölüm, çalışmanın en öğretici ölçüm hatasını içerdiği için ayrıntılı
yazılmıştır.

Bellek başlangıçta, komut satırında "ollama" geçen tüm süreçlerin RSS
(resident set size) değerleri toplanarak ölçülüyordu. Sonuç: 9B model için
34,29 GB, 12B model için 34,58 GB. **48 GB'lık bir makinede iki farklı boyutta
model neredeyse aynı sayıyı veremez** — metrik modeli değil, o anki sistemin
halini ölçüyordu. İki ayrı kusur vardı:

1. **Yabancı süreçler sayılıyordu.** Ollama 0.32 her modeli ayrı bir
   `llama-server` alt sürecinde çalıştırır. Filtre, o sırada başka bir
   uygulamanın yüklediği 23,6 GB'lık bir modelin runner'ını da, `ollama serve`
   ve masaüstü uygulamasını da toplama dahil ediyordu.
2. **RSS bu iş için yanlış metriktir.** Model ağırlıkları `mmap` ile
   eşlenir; RSS'e yansıyan miktar işletim sisteminin o an kaç sayfayı fiziksel
   bellekte tuttuğuna bağlıdır. Düzeltilmiş filtreyle bile RSS 13–22 GB
   arasında salınırken Ollama'nın kendi raporu sabit 6,29 / 7,85 GB verdi.

Düzeltme iki yönlüydü: ölçüm öncesi Ollama'ya *hangi modellerin bellekte
olduğu* sorulup yabancı modeller boşaltılıyor (embedding modeli hariç — ona
RAG'ın her soruda ihtiyacı var), ve bellek rakamı Ollama'nın kendi `/api/ps`
raporundan alınıyor. Harness ayrıca koşu boyunca kaç sohbet modelinin bellekte
olduğunu sayar; sayı 1'i aşarsa çıktıya uyarı basar. İlk üç koşuda da 1 kaldı.

**Önlemin işe yaradığının kanıtı: aynı olay ikinci kez yaşandı.** Eşik
değişikliğinden sonraki 4 ve 5 numaralı koşular sırasında makinedeki başka bir
uygulama (bir Docker konteyneri) Ollama'ya 23 GB'lık bir modeli yükledi.
Harness bunu yakaladı ve her iki model için de şu uyarıyı bastı: *"2 chat models
were resident during this run."* Sonuç verilerde açıkça görünüyor:

| | Temiz koşu (6) | Kirlenen koşu (4) |
|---|---|---|
| `qwen3.5` token ağırlıklı | 38,07 tok/s | 23,57 tok/s |
| `qwen3.5` vaka sapması | ± 0,32 | ± 9,31 |
| `gemma4` token ağırlıklı | 27,79 tok/s | 23,22 tok/s |
| `gemma4` vaka sapması | ± 0,62 | ± 7,86 |

Vaka bazında bakıldığında olayın niteliği netleşiyor: kirlenen koşuda
`qwen3.5`'in **medyanı 38,0 tok/s** ile temiz koşuyla aynıydı; ortalamayı
düşüren şey üç vakanın 14–24 tok/s'ye çökmesiydi. Yani model yavaşlamadı,
makine belirli anlarda başka bir işin altında kaldı. Standart sapmanın
ortalamanın dörtte birine çıkması bu tür bir kirlenmenin en güvenilir
işaretidir — ve tek bir ortalama sayı raporlansaydı bu işaret görünmez olurdu.
Bölüm 8'de "varyans raporla" kuralının somut karşılığı budur.

**Pratik sonuç:** her iki model de 16 GB belleğe sahip bir makinede rahatça
çalışır. Belirleyici kısıt bellek değil, üretim hızıdır. Ölçüm açısından
sonuç ise şudur: bir benchmark harness'ı yalnızca ölçmekle kalmamalı, **ölçüm
koşullarının bozulduğunu da tespit edebilmelidir.** Aksi hâlde kirlenmiş bir
koşu, temiz bir koşudan ayırt edilemez.

### 9.5 Retrieval (belge arama) süresi

Retrieval, kullanıcının sorusunu vektöre çevirip ChromaDB'de en yakın parçaları
bulma adımıdır. Bu iş **embedding modeline** aittir ve hangi sohbet modelinin
cevap vereceğinden bağımsızdır. Önceki ölçümde retrieval süresi model bazında
raporlanıyordu; bu, "qwen'de arama daha hızlı" gibi anlamsız bir çıkarıma davet
ettiği için tek bir sayıya indirildi.

| Metrik | Değer (6 koşu, n=132) |
|---|---|
| Koşu medyanı | 82 – 102 ms |
| En düşük | 77 ms |
| İlk sorgu | 0,1 – 8,6 s |

İlk sorgunun uç değeri gerçek bir maliyettir, gürültü değil: embedding modeli o
anda belleğe yükleniyordu. Ondan sonraki her sorgu ~90 ms'de tamamlandı. Bunun
kanıtı 6 numaralı koşudur: embedding modeli önceki koşudan hâlâ bellekte olduğu
için ilk sorgu cezası hiç oluşmadı ve 22 sorgunun tamamı 77–96 ms bandında
kaldı.

Yani retrieval, 1,8–3,0 saniyelik TTFT'nin yanında toplam gecikmenin ihmal
edilebilir bir parçasıdır — RAG'ın gecikmesi arama değil, üretimdir.

### 9.6 Cevap kalitesi ve kaynağa sadakat

| Model | Kalite | Kaynağa sadakat | Halüsinasyon |
|---|---|---|---|
| `qwen3.5:9b` | 14/14 | 11/11 | 0 |
| `gemma4:12b` | 14/14 | 11/11 | 0 |

Her iki model de **altı koşunun tamamında** tüm soruları geçti; cevabı
dokümanlarda bulunmayan üç kontrol sorusunun hiçbirinde uydurma yapmadı.

> **Bu tablo "iki model aynı kalitede" demek değildir.** Doğru okuma şudur:
> **bu test setinin iki modeli birbirinden ayırt etme gücü yoktur.** Her iki
> model de tavana vurmuştur ve tavana vuran bir test, üstündeki farkı ölçemez.
> Ayırt edici bir ölçüm için soru setinin çok adımlı çıkarım, çelişen kaynaklar
> ve tablo okuma gibi zor vakalarla genişletilmesi gerekir. Bu, mevcut
> prototipin değil ölçüm setinin sınırıdır ve devam çalışması olarak
> kaydedilmiştir.

#### İkinci savunma katmanı ilk kez sınandı

Kaynağa sadakattaki 11/11 sonucunun tek başına modellerin başarısı olmadığını
eklemek gerekir: kapsam dışı sorular çoğunlukla benzerlik eşiğine takılıp
modele hiç ulaşmaz. Reddetme o durumda modelin erdemi değil, mimarinin
garantisidir.

Eşiğin 0,52'den 0,46'ya çekilmesi bu tabloyu ilginç biçimde değiştirdi. Üç
kontrol sorusundan biri — *"Çalışanlara hisse senedi opsiyonu veriliyor mu,
vesting süresi kaç yıl?"* — 0,501 benzerlik skoruyla artık eşiği **geçiyor**
ve modele ulaşıyor. Yani bu soruda birinci savunma katmanı devre dışı; cevabı
yalnızca sistem prompt'u belirliyor.

Sonuç: 4, 5 ve 6 numaralı koşularda, iki model × üç koşu = **altı denemenin
altısında da model soruyu reddetti** ve sistem prompt'unda tanımlanan cümleyi
kelimesi kelimesine üretti:

> "Bu bilgi elimdeki İK dokümanlarında yer almıyor. İK ekibine
> ik@novatek.example adresinden ulaşabilirsiniz."

Bu, ölçümün en değerli tek sonucudur. Önceki koşularda ikinci katman hiç
sınanmamıştı — her kapsam dışı soru eşikte durduğu için sistem prompt'unun
işe yarayıp yaramadığı bilinmiyordu, yalnızca varsayılıyordu. Artık iki
katmanın da bağımsız olarak çalıştığı ölçülmüş durumda. Modele ulaşan zayıf
bağlam (0,501 benzerlikli, konuyla ilgisiz bir parça) modeli boşluk doldurmaya
kışkırtmadı.

Yine de dürüst sınır şudur: bu, tek bir soru üzerinde iki model ve üç koşuluk
bir gözlemdir. "Sistem prompt'u her zaman tutar" demek için yeterli değildir —
bu yüzden mimari, garantiyi hâlâ eşiğe yaslar ve prompt'u ikinci katman olarak
konumlandırır.

#### Yan gözlem: cevap uzunluğu

Aynı 14 soruya `qwen3.5` toplam 2.236, `gemma4` 1.646 token'lık cevap üretti
(6 numaralı temiz koşu). `qwen3.5` hem daha hızlı üretiyor hem de yaklaşık
%36 daha uzun yazıyor. Kalite skoru ikisinde de tam olduğuna göre bu bir
doğruluk farkı değil, üslup farkıdır — kısa ve öz cevap tercih ediliyorsa
`gemma4` sistem prompt'u değişmeden daha yakın duruyor.

### 9.7 Reasoning (düşünme) modu — ikincil bulgu

Yukarıdaki bütün ölçümler `think=false` ile yapıldı. İki ayrı tur da düşünme
modu açık koşuldu (`run7-think.json` ve — sistemin bugünkü eşiğiyle —
`run11-think.json`). Soru şuydu: model cevaplamadan önce "düşünürse" Türkçe
RAG doğruluğu artar mı? Aşağıdaki tablo, aynı eşikte (0,46) alınan
`run10` (kapalı) ve `run11` (açık) koşularını karşılaştırıyor:

| | `qwen3.5:9b` | `gemma4:12b` |
|---|---|---|
| TTFT — kapalı → **açık** | 1,8 s → **25,1 s** (14×) | 2,7 s → **24,6 s** (9×) |
| Üretilen token — kapalı → açık | 2.236 → 12.110 (5,4×) | 1.646 → 7.550 (4,6×) |
| Kalite — kapalı → açık | 14/14 → **5/14** | 14/14 → **14/14** |
| Kaynağa sadakat — kapalı → açık | 11/11 → **4/11** | 11/11 → **11/11** |

**Sayıların okunması iki uyarı gerektiriyor.**

*Birincisi:* `qwen3.5`'in 5/14'e düşmesi bir kalite kaybı değil, **ölçüm
bütçesinin sınırıdır.** Başarısız vakaların tamamında `eval_count` tam olarak
1.024'tür — yani `num_predict` tavanı — ve **dokuz vakada cevap metni tamamen
boştur** (sıfır karakter). Modelin ürettiği 1.024 token'ın tamamı düşünme
aşamasına gitti, kullanıcıya tek kelime kalmadı. Model yanlış cevap vermedi,
cevaba **hiç sıra gelmedi**. Kaynağa sadakatin 4/11'e düşmesi de aynı
nedenledir: boş cevap, ne doğru ne yanlış — ölçülebilir bir cevap değildir.

Bu, reasoning modunu açan herkes için pratik bir uyarıdır: düşünme token'ları
da aynı bütçeden harcanır, dolayısıyla `num_predict` birlikte yükseltilmelidir.
İki bağımsız turda (`run7` ve `run11`) aynı davranış görüldü, yani bulgu tek
koşuluk bir tesadüf değil.

*İkincisi:* İlk turda (`run7`) başka bir uygulama Ollama'ya 23 GB'lık bir model
yükledi ve harness bunu uyarı olarak kaydetti. Bu nedenle o turun bellek ve hız
sayıları ana karşılaştırmaya alınmamıştır; yalnızca kalite ve token tüketimi
yorumlanmaktadır.

**Sonuç:** Bu iş yükü için düşünme modu açmanın karşılığı yok. Kapalıyken zaten
14/14 alınıyordu — artırılacak yer bulunmuyor. Karşılığında ilk cevap süresi
9-14 kat, token tüketimi 4,6-5,4 kat artıyor. Bir İK asistanında kullanıcı
sorusuna 25 saniye sessiz kalmak kabul edilemez bir maliyettir.

Düşünme modunun değerli olacağı yer başka: çok adımlı çıkarım, çelişen
kaynakları uzlaştırma, hesap gerektiren sorular. Bu soru setinde böyle bir vaka
yok — ki bu da 9.6'daki tespiti destekliyor: **test setinin ayırt etme gücü
yetersiz.**

### 9.8 Determinizm

`temperature=0` ve `seed=42` sabitlendiğinde `gemma4` her iki yapılandırmada da
üretimini birebir tekrarladı: ilk üç koşuda 1.493, sonraki üç koşuda 1.646
token (fark eşik değişikliğinden gelir — artık bir kapsam dışı soru daha modele
ulaşıp cevap üretiyor). `qwen3.5` son üç koşuda 2.236 token ile birebir aynı
kaldı, ancak ilk üç koşunun ikisinde 2.185, birinde 2.170 token üretti —
%0,7'lik bir sapma.

Yani sıcaklık sıfırlanmış olsa bile üretim bit düzeyinde **garanti** biçimde
tekrarlanabilir değildir; GPU'da kayan nokta işlemlerinin sıralaması koşudan
koşuya değişebilir ve tek bir token'ın farklı seçilmesi cevabın geri kalanını
kaydırır. Altı koşuda bir kez gözlendi. Ölçüm bu yüzden tek koşuya değil
tekrarlanan koşulara dayandırılmıştır.

### 9.9 Benchmark'ın göremediği bir kusur

Ölçümler bittikten sonra sistem, kalibrasyon setindeki kısa sorularla elle
denendi. Bir soru beklenmedik biçimde başarısız oldu:

**"Babalık izni kaç gün?"** — cevap bilgi tabanında birebir yazıyor:
`| Eş doğumu (babalık izni) | 10 iş günü |`. Sistem yine de "bu bilgi
dokümanlarda yok" dedi.

Sebep eşik değil, **sıralamadır.** O satırı içeren parça (*2. Mazeret
İzinleri*) 37 parça içinde **12. sırada** ve 0,419 skorla geliyor; ilk sıraya
ise soruyla hiçbir somut bilgi paylaşmayan *6. İzin Bakiyesi Sorgulama* parçası
0,510 ile yerleşiyor. Doğru parça top-4'e hiç giremiyor, modele hiç ulaşmıyor —
ve model, elindeki bağlamda cevap olmadığı için **doğru davranarak** reddediyor.
Hata modelde değil, aramada.

Aynı soru dokümanın kendi sözcükleriyle sorulduğunda ("Eş doğumu izni kaç
gün?") doğru parça 2. sıraya çıkıyor. Bu, yoğun (dense) vektör aramasının
bilinen zayıflığıdır: **kelime dağarcığı uyuşmazlığı.** Kısa soruda bağlam az
olduğu için embedding konuyu değil yalnızca "izin" temasını yakalıyor.

Buradan iki sonuç çıkar ve ikisi de bu raporun ölçüm iddialarını sınırlar:

1. **Benchmark bu kusuru göremezdi.** Ölçüm setindeki 11 RAG sorusunun hepsi
   uzun ve düzgün kurulmuştur; hepsi doğru parçayı ilk sırada getiriyor. 14/14
   kalite skoru bu yüzden sistemin gerçek doğruluğunu değil, **test setinin
   kolaylığını** ölçüyor. Bölüm 9.6'daki "tavan etkisi" uyarısının somut
   kanıtı budur — soyut bir çekince değil, gerçekleşmiş bir kör nokta.
2. **Kalibrasyon metriği de eksiktir.** `calibrate_threshold.py`'nin
   "kaçırılan doğru soru" sütunu, sorunun eşiği geçip geçmediğini ölçer;
   getirilen parçanın cevabı *içerip içermediğini* ölçmez. 0,46 eşiğinde
   "0/19 kaçırma" raporlanmıştı — gerçekte en az bir soru cevaba ulaşamıyor.

**Önerilen düzeltme** bu prototipin kapsamı dışında bırakılmıştır, çünkü
indeksi yeniden kurmak altı koşuluk ölçümü ve kalibrasyonu birden geçersiz
kılardı: hibrit arama — vektör benzerliğinin yanına BM25 gibi kelime tabanlı
bir skor eklemek. "Babalık" kelimesini birebir içeren bir parça lexical skorla
üst sıraya taşınır. İkinci seçenek, parçaları gömerken başlık hiyerarşisini
metnin içine dahil etmektir.

### 9.10 Değerlendirme

| Kriter | Kazanan | Fark |
|---|---|---|
| Üretim hızı | `qwen3.5:9b` | %35 |
| İlk cevap süresi | `qwen3.5:9b` | ~1,0 s |
| Bellek | `qwen3.5:9b` | 1,6 GB az |
| Kalite / sadakat | — | ayırt edilemedi |
| Cevap uzunluğu | duruma göre | `qwen3.5` %36 daha uzun |

Bu iş yükü için `qwen3.5:9b` net biçimde daha uygun modeldir: ölçülebilir her
performans ekseninde önde ve kalitede ölçülebilir bir kaybı yok. `gemma4:12b`
ikinci model olarak sistemde tutulmalıdır — hem farklı bir üreticinin
modeliyle çapraz doğrulama imkânı verir hem de kısa cevap tercih edilen
senaryolarda daha iyi eşleşir.

---

## 10. Anahtar Kelimeler

Aşağıdaki terimler, staj dokümanının 8. bölümünde istendiği üzere kendi
cümlelerimle tanımlanmış ve mümkün olduğunca bu projeden örneklenmiştir.

**1. LLM (Large Language Model)**
Çok büyük metin yığınları üzerinde, bir metnin devamını tahmin etmeyi
öğrenerek eğitilmiş sinir ağı. *Örnek:* `gemma4:12b` bu projede kullanıcı
sorusunu, kendisine verilen İK dokümanı parçalarına dayanarak yanıtlar.

**2. Embedding & Vector Space (Gömme ve Vektör Uzayı)**
Metnin anlamını sayı dizisine dönüştürme ve bu sayıların oluşturduğu çok
boyutlu uzay. Anlamca yakın metinler bu uzayda birbirine yakın konumlanır.
*Örnek:* Bu projede her doküman parçası 1024 boyutlu bir vektöre çevrilir;
"harcırah" sorusu ile "günlük yemek ödemesi" bölümü kelime örtüşmesi olmadan
eşleşir.

**3. Inference (Çıkarım)**
Eğitilmiş bir modeli kullanarak yeni girdiye cevap üretme işlemi. Eğitimden
farkı: model ağırlıkları değişmez, yalnızca okunur. *Örnek:* Kullanıcı soru
sorduğunda gerçekleşen işlem çıkarımdır.

**4. Training vs Fine-Tuning vs Instruction-Tuning**
*Training (eğitim):* Modeli sıfırdan, devasa veri ve maliyetle oluşturmak.
*Fine-tuning (ince ayar):* Hazır modeli görece küçük, alana özgü veriyle
uzmanlaştırmak. *Instruction-tuning (talimat ayarı):* Modelin talimatları
izleyip sohbet edebilmesi için özel olarak ayarlanması.
*Örnek:* Bu projede hiçbiri yapılmadı; şirket bilgisi modele RAG ile çalışma
anında verildi. Bu, ince ayardan hem çok daha ucuz hem de doküman
güncellendiğinde anında yansıyan bir yöntemdir.

**5. Parameter (Model Ağırlıkları)**
Modelin eğitim sırasında öğrendiği sayısal katsayılar. Model boyutunun ölçüsü.
*Örnek:* `qwen3.5:9b` yaklaşık 9 milyar parametre içerir.

**6. Quantization & Dequantization**
Ağırlıkları daha az bitle saklayarak bellek ve hesaplama ihtiyacını düşürme;
dequantization ise işlem anında geçici olarak geri açma.
*Örnek:* `q4_K_M` sayesinde 9 milyar parametreli bir model 18 GB yerine
6,6 GB disk alanı kaplar.

**7. Prompt Engineering (System Prompt vs User Prompt)**
Modelden istenen davranışı elde etmek için girdi metnini tasarlama.
*System prompt* modelin kalıcı rol ve kurallarını, *user prompt* anlık soruyu
taşır. *Örnek:* Bu projede sistem prompt'u modele "yalnızca verilen kaynaklara
dayan, bilmiyorsan uydurma" kuralını dayatır ve ayrı bir dosyada
(`app/prompts/system_tr.txt`) tutulur.

**8. Token & Tokenizer (BPE, WordPiece)**
Model metni kelime kelime değil, "token" adı verilen parçalar hâlinde işler.
Tokenizer metni bu parçalara ayıran bileşendir; BPE (Byte-Pair Encoding) sık
görülen karakter dizilerini birleştirerek sözlük oluşturur.
*Örnek:* Türkçe sondan eklemeli olduğu için "çalışabileceğiniz" gibi bir kelime
birden fazla token'a bölünür — bu yüzden Türkçe metin, aynı içerikteki
İngilizce metne göre daha fazla token tüketir.

**9. FP16, INT8, 4-bit (Veri Tipleri)**
Sayıların bellekte kaç bitle temsil edildiği. Bit azaldıkça bellek ihtiyacı ve
hassasiyet birlikte düşer. *Örnek:* Bölüm 4.1'deki tabloda 9B model FP16'da
~18 GB, 4-bit'te ~5 GB yer kaplar.

**10. CUDA / ROCm / Apple Metal (GPU Hızlandırıcı Katmanları)**
Ekran kartının hesaplama gücüne erişmeyi sağlayan yazılım katmanları: CUDA
NVIDIA, ROCm AMD, Metal ise Apple donanımı içindir.
*Örnek:* Bu proje macOS'ta Metal üzerinden çalıştı; `ollama ps` çıktısı model
için `100% GPU` gösterdi.

**11. VRAM vs System RAM (ve Unified Memory)**
VRAM ekran kartının kendi belleği, RAM sistem belleğidir. Apple Silicon'da bu
ikisi ayrı değildir; tek havuz CPU ve GPU tarafından paylaşılır.
*Örnek:* 48 GB'lık test makinesinde ayrı bir VRAM yoktur; `qwen3.5:9b` bu
havuzdan 6,29 GB kullanmış ve `ollama ps` çıktısında `100% GPU` görünmüştür.

**12. Model Checkpoint & Weights**
Eğitilmiş model ağırlıklarının diske kaydedilmiş hâli. *Örnek:*
`ollama pull qwen3.5:9b` komutunun indirdiği 6,6 GB'lık dosya bir checkpoint'tir.

**13. Latency vs Throughput (Gecikme ve İş Hacmi)**
*Latency:* Tek bir isteğin ne kadar sürede yanıtlandığı. *Throughput:* Birim
zamanda kaç iş yapıldığı. İkisi her zaman birlikte iyileşmez; toplu işleme
throughput'u artırırken latency'yi kötüleştirebilir.
*Örnek:* Bu raporda TTFT gecikmeyi, token/s iş hacmini temsil eder.

**14. Context Window / Context Length (Bağlam Penceresi)**
Modelin aynı anda "görebildiği" toplam token sayısı; sistem prompt'u, sohbet
geçmişi ve verilen dokümanlar bu bütçeye dâhildir.
*Örnek:* Her iki model de 256K token bağlam destekliyor. Bu projede her soruda
yalnızca en alakalı 4 doküman parçası gönderilir — bağlam yettiği için değil,
alakasız metin eklemek cevap kalitesini düşürdüğü için.

**15. Open Source Model vs Proprietary Model**
Ağırlıkları indirilebilen ve kendi donanımınızda çalıştırabildiğiniz modeller
açık kaynak; yalnızca API üzerinden erişilenler kapalı devredir.
*Örnek:* Bu projenin tamamen çevrimdışı çalışabilmesinin nedeni açık ağırlıklı
modeller kullanılmasıdır.

**16. API vs Local Deployment**
Modeli bir sağlayıcının sunucusunda çağırmak ile kendi donanımınızda
çalıştırmak arasındaki tercih. İlki hızlı başlar ve veriyi dışarı gönderir;
ikincisi donanım gerektirir ve veriyi içeride tutar.

**17. LoRA (Low-Rank Adaptation)**
Modelin tüm ağırlıklarını yeniden eğitmek yerine, yanına eklenen küçük matris
çiftlerini eğiterek uyarlama yöntemi. Maliyeti tam ince ayarın çok altındadır
ve üretilen adaptör dosyası birkaç yüz MB'tır.

**18. PEFT (Parameter-Efficient Fine-Tuning)**
LoRA gibi, modelin yalnızca küçük bir bölümünü eğiterek uyarlama yapan
yöntemlerin genel adı.

**19. Hugging Face Model Hub & Repositories**
Açık kaynak modellerin, veri kümelerinin ve demoların yayınlandığı merkezi
platform. Ollama kütüphanesindeki modellerin büyük çoğunluğu kaynağını buradan
alır.

**20. GGUF, Safetensors, ONNX (Model Formatları)**
*GGUF:* llama.cpp ekosisteminin quantize model formatı; Ollama ve LM Studio
bunu kullanır. *Safetensors:* Hugging Face'in güvenli ağırlık formatı — kod
çalıştırma riski taşımaz. *ONNX:* Farklı çalışma zamanları arasında
taşınabilirlik için tasarlanmış format.

**21. Serving (Modeli Servis Etme) & Batch Inference**
*Serving:* Modeli bir API arkasında sürekli erişilebilir tutmak.
*Batch inference:* Birden fazla isteği tek seferde işleyerek verimi artırmak.
*Örnek:* Bu projede Ollama serving katmanıdır; ingest sırasında doküman
parçaları 16'lı gruplar hâlinde gömülerek batch işlemden yararlanılır.

**22. Zero-shot, One-shot, Few-shot Learning**
Modele hiç örnek vermeden (*zero-shot*), tek örnek vererek (*one-shot*) veya
birkaç örnek vererek (*few-shot*) görev tanımlama.
*Örnek:* Bu projedeki İK asistanı zero-shot çalışır: modele örnek soru-cevap
çiftleri verilmez, yalnızca kurallar ve ilgili doküman parçaları iletilir.
