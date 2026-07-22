# Lokalde LLM Çalıştırma — Araştırma Raporu

## 1. Giriş

Bu çalışmanın sorusu basit: bir dil modelini internete hiç çıkmadan, tamamen
kendi makinemizde çalıştırabilir miyiz? Çalıştırabiliyorsak bunun için ne kadar
donanım ve ne kadar para gerekiyor, karşılığında ne kadar performans alıyoruz?
Rapor bu üç soruya tahminlerle değil, ölçülmüş sayılarla cevap veriyor.

Çalışmadan iki somut çıktı kaldı:

1. **Ölçüm altyapısı** — iki güncel orta boy modeli birebir aynı koşullarda
   karşılaştıran, istendiğinde tekrar çalıştırılabilen bir benchmark aracı.
2. **Çalışan prototip** — şirket içi İK dokümanlarını yanıtlayan, dışarıya veri
   göndermeyen bir soru-cevap (RAG) sistemi.

Rapordaki bütün performans sayıları bu altyapıyla, aşağıdaki test makinesinde
ölçüldü; hiçbiri üretici broşüründen alınmış ya da tahmin edilmiş değil.

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

Devasa miktarda metinle eğitilmiş, kendisine verilen bir metnin devamında hangi
kelimenin gelmesinin en olası olduğunu tahmin eden bir yapay sinir ağı. Kulağa
mütevazı gelen bu iş yeterince büyük ölçekte öğrenildiğinde model soru
yanıtlamaya, özetlemeye, çeviri yapmaya ve kod yazmaya başlıyor.

*Örnek:* Qwen3.5, Gemma 4, Llama, Mistral.

### Embedding Modeli

Metni, anlamını sayılarla temsil eden bir vektöre (uzun bir sayı dizisine)
çeviren model. Anlamca yakın iki metin bu sayı uzayında da birbirine yakın
düşüyor. Böylece kelimeler birebir tutmasa bile anlam üzerinden arama
yapılabiliyor.

*Örnek:* Bu projede kullanılan `qwen3-embedding:0.6b` her metin parçasını
**1024 boyutlu** bir vektöre çeviriyor. "Yıllık izin hakkım kaç gün?" sorusuyla
dokümandaki "Hizmet süresi 1–5 yıl arası: 16 iş günü" satırının tek bir ortak
kelimesi yok, ama vektör uzayında yan yana düşüyorlar ve doğru parça bulunuyor.

### Neden ikisi birlikte?

RAG (Retrieval-Augmented Generation) mimarisinde iş bölümü net: embedding
modeli **doğru bilgiyi bulur**, LLM ise **bulunan bilgiyi anlaşılır bir cevaba
çevirir**. Hiçbir LLM'in eğitim verisinde sizin şirketinizin izin politikası
yok; embedding katmanı olmadan modelin elinde uydurmaktan başka seçenek
kalmıyor.

---

## 3. Lokal Kurulumun Avantajları ve Dezavantajları

### Avantajlar

**Veri gizliliği.** Şirket içi doküman, müşteri verisi ya da personel bilgisi
hiçbir aşamada cihazdan çıkmıyor. Bu projede bütün trafik `localhost` üzerinde
akıyor; internet kapalıyken de sistem eksiksiz çalışıyor. KVKK ve benzeri
düzenlemeler karşısında en sağlam savunma, veriyi hiç göndermemek.

**Kesintisiz erişim.** Sağlayıcının çökmesi, kotanın dolması veya fiyatların
zamlanması sizi hiç ilgilendirmiyor.

**Maliyet öngörülebilirliği.** İstek başına ödeme yerine bir kereye mahsus
donanım masrafı var; gider, kullanım hacminden bağımsız sabit bir kaleme
dönüşüyor. Ama dikkat: öngörülebilir olması *ucuz* olması demek değil — bu
çalışmanın fiyat araştırması, bu ölçekte lokal kurulumun buluttan ucuz
**olmadığını** gösterdi (Bölüm 4.4).

**Düşük gecikme ve kolay entegrasyon.** Yerel ağdaki veri tabanları ve iç
sistemlerle, araya internet girmeden konuşabiliyor.

### Dezavantajlar — dürüst değerlendirme

Raporun dengeli olması için madalyonun diğer yüzü de yazılmalı:

- **Kalite farkı.** 9–12 milyar parametreli lokal modeller, en büyük ticari
  modellerin karmaşık akıl yürütme performansına yetişemiyor. Basit soru-cevap
  ve doküman özetlemede aradaki fark küçük; çok adımlı analize geçince
  belirginleşiyor.
- **Başlangıç maliyeti.** Donanımın parası peşin çıkıyor ve bu ölçekte geri
  dönmüyor. Aynı sınıf bir modeli bulutta çalıştırmak yılda ~600 TL tutarken,
  lokal kurulumun sadece elektriği bunun birkaç katı (Bölüm 4.4). Lokalin
  gerekçesi tasarruf değil, verinin dışarı çıkmaması.
- **Bakım yükü.** Model güncellemesi, sürüm uyumluluğu, donanım arızası —
  hepsi kurumun kendi sorumluluğunda.
- **Eşzamanlılık sınırı.** Tek bir iş istasyonu aynı anda ancak sınırlı sayıda
  kullanıcıya yetişebiliyor.

---

## 4. Donanım Gereksinimleri

### 4.1 Model Boyutu ve Bellek İlişkisi

Model çalışacaksa ağırlıklarının belleğe sığması gerekiyor. Kaba hesabı şöyle:

```
Gereken bellek (GB) ≈ Parametre sayısı (milyar) × Byte/parametre
```

| Veri tipi | Parametre başına | 9B model için |
|---|---|---|
| FP32 | 4 byte | ~36 GB |
| FP16 / BF16 | 2 byte | ~18 GB |
| INT8 (q8_0) | ~1 byte | ~9 GB |
| 4-bit (q4_K_M) | ~0,5 byte | ~5 GB |

Bunun üstüne bir de **KV cache** (bağlam belleği) biniyor; uzun bağlam
pencerelerinde birkaç GB'ı bulabiliyor. Pratikte model boyutunun üzerine
%20–40 pay bırakmakta fayda var.

Ölçüm bu payın gerçekte ne kadar olduğunu gösteriyor. 32K bağlam penceresiyle
çalışırken Ollama'nın bildirdiği bellek şöyle:

| Model | Disk boyutu | Bellekte (32K bağlam) | Fark |
|---|---|---|---|
| `qwen3.5:9b` | 6,6 GB (6,14 GiB) | 6,29 GiB | +%2 |
| `gemma4:12b` | 7,6 GB (7,08 GiB) | 7,85 GiB | +%11 |

Yani 4-bit quantize edilmiş bu modellerde ek yük, yukarıdaki kaba hesabın
öngördüğünden daha düşük çıktı. Ama bu rakam **bağlam uzunluğuna bağlı**: KV
cache, pencere büyüdükçe onunla birlikte doğrusal büyüyor. Her iki model de
256K bağlamı destekliyor; ölçümü Ollama'nın varsayılanı olan 32K ile yaptık.
Pencereyi sonuna kadar açsaydık KV cache payı kat kat artardı. Bu yüzden donanım
planlarken sorulacak soru "model kaç GB?" değil, **"model + kullanacağım bağlam
kaç GB?"**

> Buradaki bellek sayıları Ollama'nın kendi raporundan geliyor. Süreç RSS'inin
> neden bu iş için yanıltıcı olduğunu Bölüm 9.4'te uzun uzun anlattık.

### 4.2 Apple Silicon: Unified Memory

NVIDIA sistemlerinde VRAM (ekran kartının kendi belleği) ile sistem RAM'i ayrı
iki havuz; model VRAM'e sığmazsa performans ciddi biçimde düşüyor. Apple
Silicon'da ise **tek bir bellek havuzu** var ve GPU bu havuzun büyük kısmına
doğrudan uzanabiliyor.

Pratikte ne demek bu? 48 GB RAM'li bir Mac, 24 GB VRAM'li bir RTX 4090'ın
sığdıramadığı modelleri belleğe alabiliyor. Buna karşılık ham hesap gücü
(FLOPS) ve bellek bant genişliği tarafında üst seviye NVIDIA kartlarının
gerisinde kalıyor.

> **Kural:** Apple Silicon'da "kaç GB VRAM var?" diye sorulmaz; "toplam RAM'in
> ne kadarını modele ayırabiliyorum?" diye sorulur. macOS varsayılan olarak
> toplam belleğin yaklaşık %75'ini GPU'ya verebiliyor.

### 4.3 Sadece CPU ile çalıştırma

Çalışır, ama token üretim hızı GPU'ya göre tipik olarak 5–10 kat düşüyor.
Karşılıklı konuştuğunuz bir asistan için kullanılabilir değil; gece boyunca
toplu iş (batch) döndürmek içinse gayet kabul edilebilir.

### 4.4 Tedarik ve maliyet analizi (Türkiye pazarı)

Donanım fiyatları çok oynak olduğu için maliyet araştırması hazır bir listeye
yaslanmak yerine **çalışmanın yapıldığı sırada** baştan yapıldı. Aşağıdaki
fiyatlar **22 Temmuz 2026** tarihinde toplandı ve gün içinde bile
değişebilir. Kur: TCMB 21.07.2026 bülteni, 1 USD = 47,20 TL.

> **2026'nın kendine has hâli.** Küresel bir DRAM/GDDR7 kıtlığı sürüyor ve bu
> bölümdeki her sayıya dokunuyor: 32 GB'lık bir DDR5 kiti 2024–25'te
> 4.500–5.500 TL bandındayken bugün 22.000–27.500 TL; Apple, Mac mini'nin
> 32 GB ve 64 GB seçeneklerini Türkiye'de satıştan kaldırdı; NVIDIA'nın
> 24 GB'lık RTX 50 Super serisi süresiz ertelendi. Kısacası "bari biraz daha
> bellek alayım" demek bu yıl alışılmadık derecede pahalı.

#### Apple Silicon — Apple Türkiye mağazası

| Ürün | Bellek | Fiyat |
|---|---|---|
| Mac mini M4 | 16 GB | 64.499 TL |
| Mac mini M4 | 24 GB | 76.999 TL |
| Mac mini M4 Pro | 24 GB | 104.999 TL |
| Mac mini M4 Pro | 48 GB | 161.249 TL |
| Mac Studio M4 Max | 36 GB | 159.999 TL |

Yetkili perakendecilerde (epey/Akakçe taraması) aynı ürünler belirgin biçimde
daha ucuza görünüyor — örneğin Mac mini M4 24 GB için 50.500–64.600 TL. Bu fark
Türkiye'de olağan bir durum, ama **garanti ve fatura koşulları farklı olduğu
için iki fiyatı aynı satırda yan yana koymak doğru olmaz.** Perakendede hâlâ M4
nesli MacBook stoğu var; fiyat/performans açısından ciddiye alınacak bir
seçenek.

#### NVIDIA

| Kart | VRAM | Fiyat |
|---|---|---|
| RTX 5060 Ti 16GB | 16 GB | 36.000 – 40.000 TL |
| RTX 5090 | 32 GB | 172.800 – 200.000 TL |
| RTX 5070 Ti Super / 5080 Super | 24 GB | **piyasada yok** — süresiz ertelendi |

Buradaki en dikkat çekici bulgu bir fiyat değil, bir **boşluk**: Temmuz 2026
itibarıyla tüketici tarafında 16 GB ile 32 GB arası bir seçenek yok. 16 GB'ın
yetmediği bir iş yükü, doğrudan 172.000 TL'lik karta atlamak zorunda kalıyor.
Apple tarafında ise 24, 36 ve 48 GB gibi ara basamaklar var — unified memory
mimarisinin bu ölçekteki en somut pratik faydası da tam olarak bu.

#### Ekran kartı tek başına bir sistem değildir

Maliyet hesaplarken en sık düşülen hata bu. 38.000 TL'lik bir RTX 5060 Ti tek
başına çalışan bir bilgisayar değil; yanına işlemci, 32 GB sistem RAM'i (bu yıl
tek başına 22.000 TL+), anakart, güç kaynağı, kasa ve NVMe SSD gerekiyor.
Türkiye'de hazır satılan sistemlere bakınca tablo şöyle:

| Konfigürasyon | Fiyat |
|---|---|
| Ryzen 5 7500F + RTX 5060 Ti 16GB | 48.329 TL |
| Ryzen 7 5700X + 32 GB RAM + RTX 5060 Ti 16GB | 61.499 TL |

Yani ekran kartı toplam sistemin ancak %60–75'ini oluşturuyor. Aynı bellek
sınıfındaki Mac mini M4 24 GB ile yan yana koyunca ikisi benzer bantta kalıyor;
fark kurulum emeğinde ve elektrik tüketiminde çıkıyor.

#### Elektrik

Mac mini M4 Pro için Apple'ın resmî rakamı: boşta 5 W, azami 140 W. NVIDIA
sistemlerinin tüketimini **ölçmedik**, kart incelemelerinden tahmin ettik
(5060 Ti sistemi ~90 W boşta / ~300 W yükte). Günün %30'unda cevap ürettiğini
varsayarsak, 7/24 açık duran bir kurulumun yıllık gideri şöyle:

| Sistem | Yıllık tüketim | Yıllık maliyet |
|---|---|---|
| Mac mini M4 Pro | ~399 kWh | ~1.700 TL |
| RTX 5060 Ti sistemi | ~1.340 kWh | ~5.800 TL |
| RTX 5090 sistemi | ~2.444 kWh | ~10.600 TL |

kWh birim fiyatı 4,32 TL alındı (mesken, ikinci kademe). **Uyarı:** EPDK'nın
kendi tarife tablosuna doğrudan ulaşılamadı; ikincil kaynaklar 2,59–4,32 TL/kWh
arasında birbiriyle çelişiyor. Tarife üç ayda bir güncellendiği için bu tabloya
dayanarak karar verilmeden önce `epdk.gov.tr` üzerinden teyit edilmeli.

#### Bulut API ile karşılaştırma — ve rahatsız edici sonuç

Amortisman hesabı ancak **doğru şeyi doğru şeyle** karşılaştırırsanız bir anlam
taşıyor. Lokalde çalıştırdığımız model 9–12B sınıfında; onu piyasanın en pahalı
ticari modelleriyle kıyaslamak lokal kurulumu haksız yere kârlı gösterir. Adil
kıyas şu: *aynı sınıftaki açık model bulutta çalıştırılsa kaça patlardı?*

50 kişilik bir şirket, günde 200 soru (ayda 4.400 sorgu), sorgu başına ~4.000
girdi + ~500 çıktı token varsayımıyla:

| Seçenek | Aylık | Yıllık |
|---|---|---|
| **Gemma 3 12B — bulut (DeepInfra)** | ~47 TL | ~564 TL |
| Ucuz sınıf ticari model (Flash/Lite sınıfı) | ~360–730 TL | ~4.400–8.700 TL |
| Orta sınıf ticari model (Haiku/Luna sınıfı) | ~1.350–1.450 TL | ~16.000–17.500 TL |
| Üst sınıf ticari model (Sonnet/Opus sınıfı) | ~4.000–6.800 TL | ~48.000–81.000 TL |

Sonuç net ve işin başındaki beklentiyi ters yüz ediyor: **aynı sınıf bir model
bulutta yılda ~600 TL tutuyor; lokal kurulumun sadece elektriği yılda
1.700–5.800 TL.** Donanımın 48.000–160.000 TL'lik peşin parasını hiç hesaba
katmasak bile lokal kurulum bu kıyasta **hiçbir kullanım yoğunluğunda kendini
ödemiyor** — hacim arttıkça aradaki fark kapanmıyor, açılıyor.

Lokal kurulumun maliyet açısından anlam kazandığı tek senaryo, *üst sınıf* bir
ticari modelin yerine geçtiğini varsaymak (o durumda RTX 5060 Ti sistemi
~1,5 yılda, Mac mini M4 Pro 48 GB ~3,5 yılda başa baş geliyor). Ama 12B'lik bir
modelin en büyük ticari modellerin yerini tuttuğunu söylemek, Bölüm 3'teki
kalite farkı yüzünden teknik olarak savunulabilir değil.

#### Karar: gerekçe maliyet değil, veri egemenliğidir

Bu çalışmanın maliyet tarafındaki en dürüst çıktısı şu: **lokal LLM kurulumu bu
ölçekte bir tasarruf kalemi değil, fiyatı belli bir gizlilik primi.** Kurumsal
karar da bu cümleyle kurulmalı. "Lokal daha ucuz" değil; "İK dokümanlarının,
özlük dosyalarının ve maaş verilerinin şirket dışına hiç çıkmaması için yılda
birkaç bin TL fazladan ödüyoruz." KVKK yükümlülüğü olan bir veri kümesi için bu
primi savunmak kolay; genel amaçlı bir sohbet asistanı için mümkün değil.

Maliyet gerekçesi ancak şu üç koşul aynı anda sağlanırsa ayakta durur:
(a) kıyaslanan bulut modeli üst sınıfsa, (b) donanım en az üç yıl
kullanılacaksa, (c) donanım sadece bu iş için değil, başka yükler için de
kullanılacaksa.

> **Kaynak ve güven notu.** Apple fiyatları `apple.com/tr` mağazasından, NVIDIA
> fiyatları Akakçe/Cimri listelerinden, hazır sistem fiyatları
> Trendyol/Technopat ilanlarından alındı (hepsi 22.07.2026). Şu kalemleri
> **doğrulayamadık ve tahmin de etmedik:** RTX 5090'lı tam sistemin maliyeti,
> tekil CPU/anakart/PSU fiyatları, NVIDIA sistemlerinin gerçek güç tüketimi,
> EPDK'nın kendi tarife tablosu. Bunlar rapora tahmin olarak değil,
> "doğrulanamadı" notuyla girdi.

---

## 5. Quantization (Kuantizasyon)

Model ağırlıkları normalde 16 ya da 32 bitlik ondalıklı sayılar hâlinde
saklanıyor. Quantization, bu sayıları daha az bitle ifade ederek modelin bellek
ihtiyacını ve işlem yükünü aşağı çekme işlemi. Kabaca, bir fotoğrafı biraz daha
düşük kalitede kaydedip yerden kazanmaya benziyor.

### GGUF format önekleri

| Format | Anlamı |
|---|---|
| `q8_0` | 8-bit, kaliteye en yakın, bellek kazancı sınırlı |
| `q5_K_M` | 5-bit, karma hassasiyet |
| `q4_K_M` | **4-bit karma** — kritik katmanlar daha yüksek bitte tutulur; kalite/boyut dengesinin standardı |
| `q4_0` | Düz 4-bit, daha eski ve daha kayıplı |

### Kazanç ve kayıp

- **Kazanç:** Bellek tüketimi FP16'ya göre yaklaşık dörtte bire iniyor, model
  hızlanıyor ve çok daha mütevazı donanımda çalışabiliyor.
- **Kayıp:** Dil ve mantık becerisinde küçük bir düşüş oluyor. `q4_K_M`
  seviyesinde bu kaybı günlük kullanımda çoğu zaman fark etmiyorsunuz.

**Bu projede ne yapıldı?** Her iki model de `q4_K_M` seviyesinde kullanıldı. Bu
bir zorunluluk değil, **ölçümün geçerli olması için şart** — farklı quantization
seviyelerindeki iki modeli kıyaslamak, farklı ayarlardaki iki motoru yarıştırmak
gibi; çıkan sonuç hiçbir şey anlatmaz.

### Dequantization

Quantize edilmiş ağırlıkların, işlem anında geçici olarak yüksek hassasiyete
geri açılması. Bellekte küçük duruyor, hesap yapılırken açılıyor.

---

## 6. Açık Kaynak Ekosistemi ve Araçlar

### 6.1 Çalıştırma araçları

| Araç | Tür | Değerlendirme |
|---|---|---|
| **Ollama** | CLI + HTTP API | Tek komutla model indirme, otomatik GPU/CPU dağılımı, OpenAI uyumlu API. **Bu projede kullanıldı.** |
| **LM Studio** | GUI | Grafik arayüz, model keşfi ve deneme için elverişli; yerel API sunucusu açabilir. Teknik olmayan kullanıcılar için en erişilebilir seçenek. |
| **llama.cpp** | C/C++ motor | Ollama ve LM Studio'nun altında yatan motor. Doğrudan kullanımı en fazla kontrolü verir, en fazla emek ister. |
| **vLLM** | Sunum motoru | Yüksek eşzamanlılık ve toplu çıkarım için; tek kullanıcılı senaryoda gereksiz karmaşıklık. |

**Neden Ollama?** HTTP API'si işin ölçülmesine izin veriyor — her yanıtla
birlikte `eval_count` ve `eval_duration` sayaçlarını da geri gönderiyor.
Benchmark'taki token/s değerleri bu sayaçlardan hesaplandı; elde kronometreyle
ölçseydik ağ ve akış (streaming) gecikmesi de sonuca karışırdı.

> **Not:** Ollama 0.19'dan itibaren Apple Silicon'da MLX motorunu kullanıyor.
> Bazı modellerin `-mlx` etiketli sürümleri var ama bunlar yalnızca Apple
> donanımında çalışıyor. Bu çalışmada Windows/CUDA tarafına da taşınabilsin
> diye **GGUF** sürümleri tercih edildi.

### 6.2 Uygulama geliştirme çerçeveleri

**LangChain / LlamaIndex / Haystack** — RAG hattı kurmayı kolaylaştıran
kütüphanelerdir.

Bu projede **bilerek kullanılmadılar.** Sebebi şu: parçalama, embedding, vektör
arama ve prompt kurulumu topu topu ~400 satır kod tutuyor. Hazır bir kütüphane
bu satırları gözden gizler, ama aynı zamanda ölçüm noktalarına ulaşmayı
zorlaştırır ve öğrenme amacını zayıflatır. Üretim ölçeğinde (birden çok veri
kaynağı, ajan davranışı) tercih farklı olurdu.

### 6.3 Model kaynakları

Modellerin orijinalleri **Hugging Face** üzerinde yayımlanıyor. Quantize
edilmiş GGUF sürümler içinse `bartowski`, `unsloth` ve `lmstudio-community`
hesapları takip ediliyor. Ollama kütüphanesi (`ollama.com/library`) bu işi
tamamen üstünüzden alıyor; model etiketini `ollama pull` ile doğrudan
çekiyorsunuz.

---

## 7. Model Seçimi

Çalışmanın başında aday listesinde **Llama 3 8B** ve **Gemma 2 9B** vardı. Bu
modeller 2024 çıkışlı ve Temmuz 2026 itibarıyla epey geride kalmış durumda;
Ollama kütüphanesinde aynı boyut sınıfında çok daha güçlü halefleri var. Rapora
güncelliğini yitirmiş bir kıyas koymamak için model listesi baştan araştırıldı
ve etiketlerin gerçekten var olduğu tek tek doğrulandı.

### Seçilen modeller

| Model | Üretici | Parametre | Boyut | Bağlam | Diller |
|---|---|---|---|---|---|
| `qwen3.5:9b` | Alibaba | 9B (dense) | 6,6 GB | 256K | 201 dil |
| `gemma4:12b` | Google | 12B (dense) | 7,6 GB | 256K | 140+ dil |
| `qwen3-embedding:0.6b` | Alibaba | 0,6B | 639 MB | 32K | 100+ dil |

### Seçim gerekçeleri

**Neden bu iki sohbet modeli?** Bir kıyasın anlamlı olması için karşılaştırılan
şeylerin birbirinden farklı olması gerekiyor. Bu ikili iki noktada ayrışıyor:
üretici farklı (Alibaba / Google) ve dikkat (attention) mimarisi farklı —
qwen3.5 ağırlıklı olarak doğrusal dikkat kullanan hibrit bir yapıdayken gemma4
tam dikkat kullanıyor. İkisi toplamda ~14 GB tuttuğu için 48 GB bellekte aynı
anda açık tutulup canlı A/B karşılaştırması yapılabiliyor.

**Neden bu embedding modeli?** Sistem Türkçe dokümanlarla çalışacağı için çok
dilli destek şarttı. Adaylar şunlardı:

| Model | Boyut | Bağlam | MMTEB |
|---|---|---|---|
| `qwen3-embedding:0.6b` | 639 MB | **32K** | **64,3** |
| `bge-m3:567m` | 1,2 GB | 8K | 59,6 |
| `embeddinggemma:300m` | 622 MB | 2K | 65,4 |
| `nomic-embed-text` | 274 MB | 8K | (yalnızca İngilizce) |

`embeddinggemma`'nın 2K bağlam sınırı Türkçe için ciddi bir dezavantaj: Türkçe
sondan eklemeli bir dil olduğundan aynı içerik İngilizceye kıyasla yaklaşık
%20–30 daha fazla token tutuyor. `qwen3-embedding:0.6b` hem geniş bağlam hem de
yüksek çok dilli skor sunduğu için seçildi. Bir bonusu daha var: `bge-m3` ile
aynı 1024 boyutu ürettiği için, ileride vektör şemasına dokunmadan yerine başka
model konabilir.

> **Dürüstlük notu:** MMTEB skorları çok dilli ortalamalar. Türkçeye özel,
> ayrıştırılmış ve kamuya açık bir sıralama bulunamadı; seçim bu ortalamalardan
> yola çıkarak yapıldı. Türkçe için özel eğitilmiş `TurkEmbed4Retrieval` gibi
> modeller literatürde var, ancak Ollama kütüphanesinde bulunmadıkları için bu
> çalışmanın dışında kaldılar.

---

## 8. Benchmark Metodolojisi

Bir kıyaslama ancak ölçüm adil yapıldığı kadar değerli. Uyulan kurallar:

1. **Sorular birebir aynı.** Her model harfi harfine aynı soruları alıyor
   (`bench/prompts.yaml`).
2. **Üretim ayarları aynı.** `temperature=0`, `seed=42`, `num_predict=1024`
   her çağrıda açıkça gönderiliyor.
3. **Düşünme modu açıkça kapatılıyor.** Burası kritik: `qwen3.5` "düşünme"
   modunu varsayılan olarak **açık**, `gemma4` ise **kapalı** getiriyor.
   Varsayılanlara güvenseydik biri düşünme token'ları üretirken diğeri
   üretmeyecek, hız ölçümü de baştan çöpe gidecekti. Bu yüzden iki modele de
   `think=false` açıkça gönderiliyor.
4. **Quantization aynı.** İkisi de `q4_K_M`.
5. **Isınma turu var.** İlk çağrı modeli belleğe yüklüyor; bu maliyet ayrıca
   ölçülüyor ve ilk sorunun sırtına yıkılmıyor.
6. **Sayı nereden geliyor?** token/s değeri Ollama'nın
   `eval_count`/`eval_duration` sayaçlarından hesaplanıyor, duvar saatinden
   değil.
7. **Bellek temizliği.** Her modelden önce Ollama'ya "bellekte kim var?" diye
   soruluyor ve yabancı modeller boşaltılıyor (embedding modeli hariç; RAG'ın
   ona her soruda ihtiyacı var). Koşu boyunca araya birinin girip girmediği
   izleniyor, girerse çıktıya uyarı basılıyor. Bu önlemin neden şart olduğunu
   Bölüm 9.4 anlatıyor.
8. **Koşular tekrarlanıyor, bir kısmı da ters sırada.** Tek bir koşu, koşular
   arasındaki oynamayı göremediği için "A modeli B'den hızlı" iddiasını
   taşıyamaz. Ayrıca modeller her seferinde aynı sırada ölçülürse ikinci model
   sistematik olarak dezavantajlı duruma düşebilir; iki koşuda sıra tersine
   çevrilerek bu sınandı. Toplamda altı koşu alındı, dördü temiz (Bölüm 9.1).
9. **Ortalama token ağırlıklı alınıyor.** Hız `Σ token / Σ süre` olarak
   hesaplanıyor; her sorunun tok/s değerinin düz ortalaması alınmıyor (nedeni
   Bölüm 9.2'de).

### Ölçülen metrikler

| Metrik | Tanım |
|---|---|
| **token/s** | `Σ token / Σ üretim süresi` — hızın ana göstergesi. Vaka ortalaması, standart sapma, en düşük ve en yüksek değer ayrıca kaydedilir |
| **TTFT** | İlk token'a kadar geçen süre — algılanan gecikme (medyan) |
| **Bellek** | Ollama'nın `/api/ps` raporundaki model boyutu. Süreç RSS'i de kaydedilir ama karşılaştırmada kullanılmaz (Bölüm 9.4) |
| **Retrieval** | Belge arama süresi — koşu genelinde tek sayı; embedding modeline aittir |
| **Kalite** | Cevapta beklenen bilgilerin bulunup bulunmadığı |
| **Kaynağa sadakat** | Bilgi dokümanda yokken uydurmama oranı |

Bu tablonun tamamı **cevabı** ölçüyor; hiçbiri aramanın doğru parçayı getirip
getirmediğine bakmıyor. O boşluk ayrı bir araçla (`bench/eval_retrieval.py`)
kapatıldı — modelden bağımsız çalışıyor ve Recall@4, MRR ile "cevabın modele
ulaşıp ulaşmadığını" ölçüyor. Neden gerektiği ve ne bulduğu Bölüm 9.10'da.

### Kalite neden anahtar kelime ile ölçülüyor?

Cevap kalitesini başka bir LLM'e puanlatmak (LLM-as-judge) yaygın bir yöntem,
ama burada tercih edilmedi: hakemlik yapan modelin kendi oynaklığı, zaten
modelleri ölçmeye çalışan bir deneye ikinci bir belirsizlik katardı. Onun
yerine her sorunun cevabında geçmesi gereken somut değerler (`16`, `750`,
`2 katı`) önceden yazıldı. Ölçüm bu hâliyle mekanik ve tekrarlanabilir; üçüncü
bir kişi aynı sayılara yeniden ulaşabilir.

Türkçe için ufak bir normalizasyon gerekti: `İzmir` ile `izmir` karşılaştırması
Türkçedeki noktalı/noktasız i meselesi yüzünden özel işlem istiyor; ayrıca
aksanlar da sadeleştiriliyor.

---

## 9. Sonuçlar

### 9.1 Ölçüm düzeni

Her koşuda 14 soru var: 3 tanesi düz üretim (RAG'sız), 11 tanesi RAG sorusu —
bu 11'in 3'ü de cevabı dokümanlarda bilerek **bulunmayan** kontrol soruları.
Toplamda altı koşu alındı:

| Koşu | Dosya | Model sırası | Eşik | Durum |
|---|---|---|---|---|
| 1 | `run4-clean.json` | qwen3.5 → gemma4 | 0,52 | temiz |
| 2 | `run5-reversed.json` | gemma4 → qwen3.5 | 0,52 | temiz |
| 3 | `run6-repeat.json` | qwen3.5 → gemma4 | 0,52 | temiz |
| 4 | `run8-clean.json` | qwen3.5 → gemma4 | 0,46 | **kirlendi** |
| 5 | `run9-reversed.json` | gemma4 → qwen3.5 | 0,46 | gemma yarısı kirlendi |
| 6 | `run10-repeat.json` | qwen3.5 → gemma4 | 0,46 | temiz |

Koşular arasında benzerlik eşiği 0,52'den 0,46'ya çekildi; gerekçesi proje
raporunun 5. bölümünde. Eşik **üretim hızına dokunmuyor** — yalnızca sorunun
modele ulaşıp ulaşmayacağına karar veriyor, model çağrıldıktan sonrasına hiç
karışmıyor. Bu yüzden rapor iki kaynağı ayrı tutuyor:

- **Hız, gecikme ve bellek** sayıları 1-3 numaralı temiz koşulardan geliyor.
- **Reddetme davranışı** ise sistemin bugünkü ayarıyla alınan 4-6 numaralı
  koşulardan geliyor (Bölüm 9.6).

6. koşu — eşik değişikliğinden sonraki tek tam temiz koşu — hız sayılarını
bağımsız olarak doğruladı: `qwen3.5` 38,07, `gemma4` 27,79 tok/s; yani ilk üç
koşunun bandında. 4 ve 5 numaralı koşuların neden kullanılamadığı Bölüm 9.4'te
anlatılıyor; kısacası ölçüm sürerken makinede başka bir uygulama 23 GB'lık bir
model çalıştırdı ve harness bunu kendi uyarı mekanizmasıyla yakaladı.

İkinci ve beşinci koşularda model sırası bilerek ters çevrildi. Sebebi şu: iki
model art arda ölçüldüğünde ikinci sıradaki model, birincinin bıraktığı bellek
baskısının ve yaklaşık beş dakikalık kesintisiz üretimin ısıttığı bir işlemcinin
üzerinde çalışıyor. Sıra bir yanlılık yaratıyorsa, ters koşuda bunun yön
değiştirmesi gerekirdi.

**Değiştirmedi.** `qwen3.5` ilk sıradayken 37,19 ve 36,99 tok/s, ikinci
sıradayken 37,86 tok/s ölçüldü; `gemma4` ikinci sıradayken 27,58 ve 27,94, ilk
sıradayken 27,42 tok/s. Fark her iki modelde de %2'nin altında ve yönü tutarsız
— yani sıra etkisi bu ölçüm düzeninde gürültünün içinde kaybolup gidiyor.

### 9.2 Üretim hızı

| Model | tok/s (koşu 1-3 ort.) | Standart sapma | Koşu değerleri |
|---|---|---|---|
| **`qwen3.5:9b`** | **37,35** | 0,46 | 37,19 · 37,86 · 36,99 |
| `gemma4:12b` | 27,65 | 0,27 | 27,58 · 27,42 · 27,94 |

`qwen3.5:9b`, `gemma4:12b`'ye göre **yaklaşık %35 daha hızlı** yazıyor. Üç
koşunun standart sapması, iki model arasındaki farkın yirmide birinden bile
küçük; dolayısıyla bu farkı "ölçüm gürültüsüdür" diye geçiştirmek mümkün değil.

Farkın iki kaynağı var: parametre sayısı (9,7B'ye karşı 12B) ve mimari —
`qwen3.5` ağırlıklı olarak doğrusal dikkat kullanan hibrit bir yapıda, `gemma4`
ise tam dikkat kullanıyor.

Bu üç koşunun ardından, eşik değişikliğiyle birlikte alınan dördüncü temiz koşu
(6 numaralı, `run10-repeat.json`) aynı sayıları bağımsız olarak tekrar üretti:
`qwen3.5` 38,07 ± 0,32 ve `gemma4` 27,79 ± 0,62 tok/s. Dört temiz koşu boyunca
`qwen3.5` 36,99–38,07, `gemma4` ise 27,42–27,94 bandında kaldı; iki modelin
aralıkları hiçbir yerde birbirine değmiyor.

> **Metodoloji notu — neden token ağırlıklı ortalama?** tok/s değeri
> `Σ üretilen token / Σ üretim süresi` şeklinde hesaplanıyor; her sorunun tok/s
> değerinin düz ortalaması alınmıyor. Düz ortalama, 19 token'lık kısacık bir
> cevapla 400 token'lık uzun bir cevabı eşit sayardı — oysa kısa cevapta
> ölçtüğünüz şey büyük ölçüde prompt'u işleme maliyeti, kullanıcının hissettiği
> yazma hızı değil. Bu sette iki yöntem arasındaki fark %2'nin altında kaldı,
> ama kısa cevapların ağır bastığı bir sette ikisi ciddi biçimde ayrışır.

### 9.3 Gecikme ve yükleme

| Model | TTFT (medyan) | TTFT aralığı | Model yükleme |
|---|---|---|---|
| **`qwen3.5:9b`** | **1.941 ms** | 1.761 – 1.951 ms | 3,8 – 11,7 s |
| `gemma4:12b` | 2.978 ms | 2.504 – 3.356 ms | 5,0 – 8,9 s |

TTFT (ilk token'ın ekrana düşmesine kadar geçen süre), kullanıcının "acaba
dondu mu?" diye düşündüğü süre. Hissedilen hız açısından toplam süreden bile
daha belirleyici. `qwen3.5` burada da yaklaşık bir saniye önde.

Yükleme süreleri koşudan koşuya iki-üç kat oynadı. Bu modelin değil, işletim
sisteminin işi: model dosyası dosya sistemi önbelleğinde sıcaksa yükleme hızlı
oluyor, diskten okunuyorsa yavaş. En uç örnek 6 numaralı koşu: model bir önceki
koşudan hâlâ bellekte olduğu için yükleme 0,57 saniyeye indi — yani burada
ölçülen şeyin modelle bir ilgisi yok, tamamen makinenin o anki hâliyle ilgili.
Bu yüzden yükleme süresini model karşılaştırmasına hiç katmadık; sadece "ilk
soru pahalı, sonrakiler değil" gözlemini belgelemek için tabloda duruyor.

### 9.4 Bellek — ve ölçümün nasıl yanıltabildiği

| Model | Ollama'nın bildirdiği | Model dosyası |
|---|---|---|
| **`qwen3.5:9b`** | **6,29 GB** | 6,6 GB |
| `gemma4:12b` | 7,85 GB | 7,6 GB |

Bu bölüm biraz uzun, çünkü çalışmanın en öğretici ölçüm hatası burada yaşandı.

Bellek başlangıçta, komut satırında "ollama" geçen bütün süreçlerin RSS
(resident set size) değerleri toplanarak ölçülüyordu. Çıkan sonuç: 9B model
için 34,29 GB, 12B model için 34,58 GB. **48 GB'lık bir makinede iki farklı
boyuttaki model neredeyse aynı sayıyı veremez** — belli ki metrik modeli değil,
o anki sistemin genel hâlini ölçüyordu. İki ayrı kusur vardı:

1. **Yabancı süreçler de toplamaya giriyordu.** Ollama 0.32 her modeli ayrı bir
   `llama-server` alt sürecinde çalıştırıyor. Filtre; o sırada başka bir
   uygulamanın yüklediği 23,6 GB'lık modelin runner'ını da, `ollama serve`'ü de,
   masaüstü uygulamasını da toplama dahil ediyordu.
2. **RSS zaten bu iş için yanlış metrik.** Model ağırlıkları `mmap` ile
   eşleniyor; RSS'e yansıyan miktar, işletim sisteminin o an kaç sayfayı
   fiziksel bellekte tuttuğuna bağlı. Filtreyi düzelttikten sonra bile RSS
   13–22 GB arasında gidip gelirken Ollama'nın kendi raporu sabit sabit
   6,29 / 7,85 GB dedi.

Düzeltme iki koldan yapıldı: ölçümden önce Ollama'ya *bellekte hangi modeller
var* diye soruluyor ve yabancı olanlar boşaltılıyor (embedding modeli hariç —
ona RAG'ın her soruda ihtiyacı var); bellek rakamı da Ollama'nın kendi `/api/ps`
raporundan alınıyor. Harness bunun üstüne, koşu boyunca kaç sohbet modelinin
bellekte olduğunu sayıyor; sayı 1'i geçerse çıktıya uyarı basıyor. İlk üç koşuda
bu sayı hep 1 kaldı.

**Önlemin işe yaradığının kanıtı: aynı olay bir kez daha yaşandı.** Eşik
değişikliğinden sonraki 4 ve 5 numaralı koşular sırasında makinedeki başka bir
uygulama (bir Docker konteyneri) Ollama'ya 23 GB'lık bir model yükledi. Harness
bunu anında yakaladı ve iki model için de şu uyarıyı bastı: *"2 chat models were
resident during this run."* Etkisi verilerde çıplak gözle görülüyor:

| | Temiz koşu (6) | Kirlenen koşu (4) |
|---|---|---|
| `qwen3.5` token ağırlıklı | 38,07 tok/s | 23,57 tok/s |
| `qwen3.5` vaka sapması | ± 0,32 | ± 9,31 |
| `gemma4` token ağırlıklı | 27,79 tok/s | 23,22 tok/s |
| `gemma4` vaka sapması | ± 0,62 | ± 7,86 |

Soru soru bakınca olayın niteliği netleşiyor: kirlenen koşuda `qwen3.5`'in
**medyanı 38,0 tok/s** ile temiz koşuyla aynıydı; ortalamayı aşağı çeken şey,
üç sorunun 14–24 tok/s'ye çakılmasıydı. Yani model yavaşlamadı; makine belirli
anlarda başka bir işin altında kaldı. Standart sapmanın ortalamanın dörtte
birine fırlaması bu tür bir kirlenmenin en güvenilir işareti — ve tek bir
ortalama sayı raporlansaydı bu işaret hiç görünmeyecekti. Bölüm 8'deki "varyansı
da yaz" kuralının somut karşılığı işte bu.

**Pratikte ne anlama geliyor?** Her iki model de 16 GB belleği olan bir makinede
rahat rahat çalışır; asıl darboğaz bellek değil, üretim hızı. Ölçüm tarafındaki
ders ise şu: bir benchmark harness'ı sadece ölçmekle yetinmemeli, **ölçüm
koşullarının bozulduğunu da fark edebilmeli.** Yoksa kirlenmiş bir koşuyla temiz
bir koşuyu birbirinden ayırmanın hiçbir yolu kalmıyor.

### 9.5 Retrieval (belge arama) süresi

Retrieval, kullanıcının sorusunu vektöre çevirip ChromaDB'de en yakın parçaları
bulma adımı. Bu işi **embedding modeli** yapıyor ve cevabı hangi sohbet
modelinin vereceğiyle hiçbir ilgisi yok. Önceki ölçümde arama süresi model
bazında raporlanıyordu; bu da "demek ki qwen'de arama daha hızlıymış" gibi
anlamsız bir çıkarıma kapı araladığı için tek bir sayıya indirildi.

| Metrik | Değer (6 koşu, n=132) |
|---|---|
| Koşu medyanı | 82 – 102 ms |
| En düşük | 77 ms |
| İlk sorgu | 0,1 – 8,6 s |

İlk sorgudaki o uç değer gürültü değil, gerçek bir maliyet: embedding modeli tam
o sırada belleğe yükleniyordu. Ondan sonraki her sorgu ~90 ms'de bitti. Kanıtı
6 numaralı koşu: embedding modeli önceki koşudan hâlâ bellekte olduğu için ilk
sorgu cezası hiç yaşanmadı ve 22 sorgunun tamamı 77–96 ms bandında kaldı.

Kısacası arama süresi, 1,8–3,0 saniyelik TTFT'nin yanında görünmez kalıyor —
RAG'ı yavaşlatan şey arama değil, modelin cevabı yazması.

Bölüm 9.10'da aramaya eklenen kelime tabanlı ikinci kol bu tabloyu değiştirmiyor:
BM25 indeksi 37 parça için bir kez 1,6 ms'de kuruluyor, sorgu başına maliyeti
0,027 ms. Yukarıdaki ~90 ms'nin neredeyse tamamı embedding çağrısında geçtiği
için, eklenen kol ölçüm gürültüsünün altında kalıyor.

### 9.6 Cevap kalitesi ve kaynağa sadakat

| Model | Kalite | Kaynağa sadakat | Halüsinasyon |
|---|---|---|---|
| `qwen3.5:9b` | 14/14 | 11/11 | 0 |
| `gemma4:12b` | 14/14 | 11/11 | 0 |

Her iki model de **altı koşunun tamamında** bütün soruları geçti; cevabı
dokümanlarda olmayan üç kontrol sorusunun hiçbirinde bir şey uydurmadı.

> **Bu tablo "iki model aynı kalitede" demek değil.** Doğru okuması şu: **bu
> test setinin iki modeli birbirinden ayırt edecek gücü yok.** İkisi de tavana
> vurmuş durumda; tavana vuran bir test de tavanın üstündeki farkı ölçemez.
> Ayırt edici bir ölçüm için soru setine çok adımlı çıkarım, birbiriyle çelişen
> kaynaklar ve tablo okuma gibi zor vakalar eklenmeli. Bu, prototipin değil
> ölçüm setinin sınırı ve devam çalışması olarak not edildi.

Tablonun bir körlüğü daha var ve o körlük sonradan ölçüldü: bu puan **cevabı**
değerlendiriyor, cevaba giden **aramayı** değil. Doğru parça hiç getirilemediğinde
model doğru davranıp reddediyor ve tabloya bir hata olarak yansımıyor. Aramanın
kendisini ölçen ayrı bir metrik Bölüm 9.10'da kuruldu; iki kusuru da o buldu.

#### İkinci savunma katmanı ilk kez sınandı

Kaynağa sadakatteki 11/11'in tek başına modellerin marifeti olmadığını da
söylemek gerek: kapsam dışı sorular çoğu zaman benzerlik eşiğine takılıp modele
hiç ulaşmıyor. O durumda reddetme modelin erdemi değil, mimarinin garantisi.

Eşiğin 0,52'den 0,46'ya çekilmesi bu tabloyu ilginç biçimde değiştirdi. Üç
kontrol sorusundan biri — *"Çalışanlara hisse senedi opsiyonu veriliyor mu,
vesting süresi kaç yıl?"* — 0,501 benzerlik skoruyla artık eşiği **geçiyor** ve
modele ulaşıyor. Yani o soruda birinci savunma hattı devre dışı; ne cevap
verileceğine tek başına sistem prompt'u karar veriyor.

Sonuç: 4, 5 ve 6 numaralı koşularda, iki model × üç koşu = **altı denemenin
altısında da model soruyu geri çevirdi** ve sistem prompt'unda tarif edilen
cümleyi kelimesi kelimesine yazdı:

> "Bu bilgi elimdeki İK dokümanlarında yer almıyor. İK ekibine
> ik@novatek.example adresinden ulaşabilirsiniz."

Bu, bütün ölçümün en değerli tek sonucu. Önceki koşularda ikinci katman hiç
sınanmamıştı — her kapsam dışı soru eşikte takıldığı için sistem prompt'unun
işe yarayıp yaramadığını bilmiyorduk, öyle olduğunu varsayıyorduk. Artık iki
katmanın da tek başına çalıştığı ölçülmüş durumda. Modele ulaşan zayıf bağlam
(0,501 benzerlikli, konuyla ilgisiz bir parça) modeli boşluk doldurmaya
kışkırtmadı.

Yine de sınırı dürüstçe koyalım: bu, tek bir soru üzerinde iki model ve üç
koşuluk bir gözlem. "Sistem prompt'u her koşulda tutar" demek için yeterli
değil — bu yüzden mimari garantiyi hâlâ eşiğe yaslıyor, prompt'u da ikinci
katman olarak arkada tutuyor.

#### Yan gözlem: cevap uzunluğu

Aynı 14 soruya `qwen3.5` toplam 2.236, `gemma4` ise 1.646 token'lık cevap yazdı
(6 numaralı temiz koşu). `qwen3.5` hem daha hızlı yazıyor hem de yaklaşık %36
daha uzun. Kalite puanı ikisinde de tam olduğuna göre bu bir doğruluk farkı
değil, üslup farkı — kısa ve öz cevap isteniyorsa `gemma4` sistem prompt'una
hiç dokunmadan bu tercihe daha yakın duruyor.

### 9.7 Reasoning (düşünme) modu — ikincil bulgu

Yukarıdaki ölçümlerin hepsi `think=false` ile yapıldı. İki tur da düşünme modu
açık koşuldu (`run7-think.json` ve — sistemin bugünkü eşiğiyle —
`run11-think.json`). Merak ettiğimiz şey basitti: model cevaplamadan önce
"düşünürse" Türkçe RAG doğruluğu artıyor mu? Aşağıdaki tablo, aynı eşikte
(0,46) alınan `run10` (kapalı) ve `run11` (açık) koşularını yan yana koyuyor:

| | `qwen3.5:9b` | `gemma4:12b` |
|---|---|---|
| TTFT — kapalı → **açık** | 1,8 s → **25,1 s** (14×) | 2,7 s → **24,6 s** (9×) |
| Üretilen token — kapalı → açık | 2.236 → 12.110 (5,4×) | 1.646 → 7.550 (4,6×) |
| Kalite — kapalı → açık | 14/14 → **5/14** | 14/14 → **14/14** |
| Kaynağa sadakat — kapalı → açık | 11/11 → **4/11** | 11/11 → **11/11** |

**Bu sayıları okurken iki şeye dikkat etmek gerekiyor.**

*Birincisi:* `qwen3.5`'in 5/14'e düşmesi bir kalite kaybı değil, **ölçüm
bütçesinin duvara toslaması.** Başarısız vakaların hepsinde `eval_count` tam
tamına 1.024 — yani `num_predict` tavanı — ve **dokuz vakada cevap metni
tamamen boş** (sıfır karakter). Modelin ürettiği 1.024 token'ın tamamı düşünme
aşamasında harcandı, kullanıcıya tek kelime kalmadı. Yani model yanlış cevap
vermedi; cevap yazmaya **hiç sıra gelmedi**. Kaynağa sadakatin 4/11'e inmesi de
aynı sebepten: boş cevap ne doğru ne yanlış, ölçülecek bir şey değil.

Bu, düşünme modunu açacak herkes için pratik bir uyarı: düşünme token'ları da
aynı bütçeden yeniyor, dolayısıyla `num_predict` de birlikte yükseltilmeli. Aynı
davranış iki bağımsız turda (`run7` ve `run11`) görüldü; yani tek seferlik bir
tesadüf değil.

*İkincisi:* İlk turda (`run7`) başka bir uygulama Ollama'ya 23 GB'lık bir model
yükledi, harness da bunu uyarı olarak kaydetti. Bu yüzden o turun bellek ve hız
sayıları ana karşılaştırmaya girmedi; sadece kalite ve token tüketimi
yorumlandı.

**Sonuç:** Bu iş için düşünme modunu açmanın bir karşılığı yok. Kapalıyken zaten
14/14 alınıyordu; yükseltilecek yer kalmamıştı. Buna karşılık ilk cevabın gelmesi
9-14 kat, token tüketimi 4,6-5,4 kat artıyor. Bir İK asistanının, sorulan soruya
25 saniye boyunca sessiz kalması kabul edilebilir bir maliyet değil.

Düşünme modunun asıl işe yarayacağı yerler başka: çok adımlı çıkarım, çelişen
kaynakları uzlaştırma, hesap isteyen sorular. Bu soru setinde bunlardan hiç yok
— ki bu da 9.6'daki tespiti doğruluyor: **test setinin ayırt etme gücü
yetersiz.**

### 9.8 Determinizm

`temperature=0` ve `seed=42` sabitlendiğinde `gemma4` her iki yapılandırmada da
kendini birebir tekrarladı: ilk üç koşuda 1.493, sonraki üç koşuda 1.646 token
(aradaki fark eşik değişikliğinden geliyor — artık bir kapsam dışı soru daha
modele ulaşıp cevap üretiyor). `qwen3.5` son üç koşuda 2.236 token ile tıpatıp
aynı kaldı, ama ilk üç koşunun ikisinde 2.185, birinde 2.170 token üretti —
%0,7'lik bir sapma.

Yani sıcaklığı sıfıra çekmek üretimin bit düzeyinde tekrarlanacağını
**garanti etmiyor**; GPU'da kayan nokta işlemlerinin sırası koşudan koşuya
değişebiliyor ve tek bir token'ın farklı seçilmesi cevabın geri kalanını
kaydırıyor. Altı koşuda bir kez karşımıza çıktı. Ölçümün tek bir koşuya değil,
tekrarlanan koşulara dayanmasının sebeplerinden biri de bu.

### 9.9 Benchmark'ın göremediği bir kusur

Ölçümler bittikten sonra sistemi kalibrasyon setindeki kısa sorularla elle
denedik. Bir soru hiç beklemediğimiz şekilde patladı:

**"Babalık izni kaç gün?"** — cevap bilgi tabanında birebir yazıyor:
`| Eş doğumu (babalık izni) | 10 iş günü |`. Sistem yine de "bu bilgi
dokümanlarda yok" dedi.

Sebebi eşik değil, **sıralama.** O satırı içeren parça (*2. Mazeret İzinleri*)
37 parça arasında **12. sırada**, 0,419 skorla geliyor; ilk sıraya ise soruyla
tek bir somut bilgi paylaşmayan *6. İzin Bakiyesi Sorgulama* parçası 0,510 ile
kuruluyor. Doğru parça ilk 4'e hiç giremiyor, dolayısıyla modele hiç ulaşmıyor —
model de elindeki metinde cevap olmadığı için **doğru davranıp** soruyu
reddediyor. Yani hata modelde değil, aramada.

Aynı soruyu dokümanın kendi sözcükleriyle sorduğunuzda ("Eş doğumu izni kaç
gün?") doğru parça 2. sıraya çıkıyor. Bu, yoğun (dense) vektör aramasının
bilinen zayıf noktası: **kelime dağarcığı uyuşmazlığı.** Kısa soruda ortada
yeterince bağlam olmadığı için embedding, konuyu değil sadece "izin" temasını
yakalıyor.

Buradan iki sonuç çıkıyor ve ikisi de bu raporun ölçüm iddialarına sınır
koyuyor:

1. **Benchmark bu kusuru zaten göremezdi.** Ölçüm setindeki 11 RAG sorusunun
   hepsi uzun ve düzgün yazılmış; hepsinde doğru parça ilk sırada geliyor.
   Dolayısıyla 14/14'lük kalite puanı sistemin gerçek doğruluğunu değil,
   **test setinin kolaylığını** ölçüyor. Bölüm 9.6'daki "tavan etkisi"
   uyarısının somut kanıtı bu — teorik bir çekince değil, başımıza gelmiş bir
   kör nokta.
2. **Kalibrasyon metriği de eksik.** `calibrate_threshold.py`'nin "kaçırılan
   doğru soru" sütunu yalnızca sorunun eşiği geçip geçmediğine bakıyor; gelen
   parçanın içinde cevap *var mı yok mu* ona bakmıyor. 0,46 eşiğinde "0/19
   kaçırma" raporlanmıştı — oysa gerçekte en az bir soru cevaba ulaşamıyor.

**Çözümü belli:** hibrit arama — vektör benzerliğinin yanına BM25 gibi kelime
tabanlı bir skor eklemek. Böylece içinde birebir "babalık" geçen bir parça
lexical skor sayesinde üst sıraya çıkar.

Bu düzeltme önce "prototipin dışında" bırakılmıştı; gerekçe, indeksi yeniden
kurmanın altı koşuluk ölçümü ve kalibrasyonu tek hamlede geçersiz kılacağıydı.
**O gerekçe yanlıştı.** BM25 aynı parçalar üzerinde çalışan bir kelime skoru;
yeniden gömme (embedding) gerektirmiyor, dolayısıyla parçalar da vektörleri de
olduğu gibi kalıyor. Kusur kapatıldı ve nasıl kapatıldığı bir sonraki bölümde.

### 9.10 Kusurun kapatılması: iki kollu arama

Bir önceki bölümdeki kusur kapatıldı. Ama asıl anlatılmaya değer olan düzeltmenin
kendisi değil, düzeltmeden **önce** yapılması gereken işti.

#### Önce ölçüm aleti

Bölüm 9.9'un kendi itirafı şuydu: `calibrate_threshold.py` yalnızca sorunun eşiği
geçip geçmediğine bakıyor, gelen parçanın içinde cevap **var mı yok mu** ona
bakmıyor. Yani sistemin bu hatayı görecek bir ölçüm aracı yoktu. Böyle bir araç
olmadan yapılacak her düzeltme, işe yarayıp yaramadığı bilinmeden yapılmış olurdu.

Bu yüzden ilk iş `bench/eval_retrieval.py` oldu. Kalibrasyon setindeki her kapsam
içi soruya, cevabı fiilen içeren parça **gold** etiketi olarak eklendi (37 bölümün
tamamı benzersiz olduğu için parça, dosya adı + başlık yolu ile tanımlanıyor;
`chunk_id` bir içerik özeti olduğundan doküman her düzenlendiğinde kayardı). Betik
üç sayı üretiyor:

| Sayı | Ne ölçüyor |
|---|---|
| **sıra** | Gold parça 37 parçalık tam sıralamada kaçıncı. Teşhis: hatanın *nedenini* söyler |
| **Recall@4** | Gold parça, sıralamanın devrettiği ilk 4'ün içinde mi. Yalnızca sıralayıcıyı ölçer |
| **ulaştı** | Gold parça sıralamayı *ve* eşiği geçip modele fiilen vardı mı. Kullanıcının yaşadığı sayı |

Bu üçü tam da ilginç hataların olduğu yerde ayrışıyor. Betik ayrıca gold etiketi
indekste karşılığı olmayan bir bölümü gösterirse **çalışmayı reddediyor**; yanlış
bir alet, alet olmamasından kötüdür.

Baz çizgi ölçümü aleti doğruladı: babalık sorusu **sıra 12, skor 0,419** çıktı —
Bölüm 9.9'da elle bulunmuş sayıların birebir aynısı.

#### Aletin bulduğu ikinci kusur

Baz çizgi beklenmeyen bir şey daha gösterdi:

> **"İzin devri var mı?"** — cevabı `1.4 Devir Kuralı` bölümünde yazıyor. O parça
> 5. sırada ve skoru **0,520**, yani eşiğin *üstünde*. İlk 4'e giremediği için
> modele hiç ulaşmıyor; model de elindeki dört parçada cevap olmadığı için doğru
> davranıp reddediyor.

Bu vaka babalıktan daha sinsi: skoru eşiğin üstünde olduğu için **hiçbir eşik
ayarı onu bulamazdı.** Yalnızca sıraya duyarlı bir metrik görebilirdi. Bölüm 9.9
"en az bir soru cevaba ulaşamıyor" derken temkinli davranmıştı; ölçülen sayı iki.

#### Tasarım: ekleyen bir kol, değiştiren değil

Aramaya BM25 tabanlı ikinci bir kol eklendi. Kritik olan nasıl birleştirildikleri:

| Kol | Kural |
|---|---|
| Yoğun (vektör) | Kosinüs ≥ 0,46, en fazla `top_k` — **aynen korundu**, sırası dahil |
| Sözcüksel (BM25) | Nadirlik tabanını aşan tek bir parça, yoğun kolun **arkasına eklenir** |

İki karar bu değişikliği tasarım gereği regresyonsuz yapıyor:

**RRF gibi bir füzyon kullanılmadı.** Füzyon, yoğun kolun ilk 4'ünü kendi arasında
yeniden sıralar; `temperature=0` ve sabit seed altında bu, zaten doğru cevaplanan
soruların üretilen metnini kaydırır. BM25'in o sıralamayı iyileştirdiğine dair
elimizde kanıt yoktu — yani bedeli olan, faydası ölçülmemiş bir risk olurdu.
Arkaya ekleme ise bir özelliği koruyor: **sözcüksel kol ateşlemediğinde sonuç
öncekiyle birebir aynı.**

**Sözcüksel kol, sert reddi yumuşak redde çeviremiyor.** Yoğun kol hiçbir parça
bulamadıysa kol hiç çalışmıyor. Kapsam dışı 9 sorunun 7'si bu durumda: üretim
çağrısı harcanmadan reddediliyorlar. Kelime eşleşmesinin bunu "modelin takdirine"
çevirmesi, raporun Bölüm 9.6'da savunduğu iki bağımsız savunma yapısını zayıflatırdı.

#### Kapıyı neyin açtığı — ve ilk yanlış metrik

Ham BM25 skoru kapı olarak kullanılamaz: sınırsızdır ve sorgu uzunluğuyla büyür,
dolayısıyla sabit bir taban dört kelimelik soruyla yirmi kelimelik soruda aynı
şeyi ifade etmez. Bunun yerine **nadirlik** ölçülüyor: parçanın içerdiği en nadir
sorgu kelimesinin IDF'i, korpustaki azami IDF'e bölünmüş hâli. 1,0 demek, parçanın
sorudaki kelimelerden birini içerdiği ve o kelimenin 37 parçanın **tam olarak
birinde** geçtiği demek. "babalık" tam da böyle bir kelime.

İlk tasarlanan metrik bu değildi ve yanlış olduğu ölçümle ortaya çıktı. Başlangıçta
**kapsama** ölçülüyordu: sorgunun IDF ağırlığının ne kadarının parçada bulunduğu.
Payda, korpusta *var olan* sorgu kelimelerinden hesaplanıyordu — yani hiçbir yerde
geçmeyen kelimeler atılıyordu. Sonuç ters yönde çalıştı:

> **"Kreş yardımı var mı?"** (kapsam dışı) kapsama **1,000** aldı. Çünkü "kreş"
> korpusta hiç geçmiyor, atılıyor; geriye kalan "yardımı / var / mı" ise alakasız
> bir bölümde tamamen bulunuyor.

Sorunun kapsam dışı olduğunun en güçlü kanıtı — o tek ayırt edici kelimenin hiçbir
yerde geçmemesi — metrik tarafından çöpe atılıyordu. Nadirlik bu şekilde
kandırılamıyor, çünkü parçanın **ne içerdiğini** soruyor. Hata bir regresyon
testiyle kilitli.

#### Sonuç

19 gold etiketli soru üzerinde, düzeltmeden önce ve sonra:

| | baz çizgi | iki kollu |
|---|---|---|
| Recall@4 (yalnız sıralama) | 0,895 | 0,895 |
| MRR | 0,857 | 0,857 |
| **Cevabına ulaşan** | **0,895** (17/19) | **0,947** (18/19) |
| Teslim edilen parça kümesi değişen soru | — | 1/19 |
| Kapsam dışına eklenen parça | — | 0 |
| Sert red (kapsam dışı) | 7/9 | 7/9 |

Recall@4'ün değişmemesi bir eksiklik değil, kanıt: o metrik yalnızca yoğun kolu
ölçüyor ve yoğun kola hiç dokunulmadı. Değişen tek şey, sıralamanın altında kalan
doğru parçanın artık modele ulaşabilmesi.

Babalık sorusu hâlâ 12. sırada ve hâlâ 0,419 skorlu — ama artık cevaplanıyor:
*"Babalık izni için verilen süre 10 iş gündür."* Arayüzde o parça "kelime
eşleşmesi" rozetiyle işaretleniyor, çünkü skoru düşük göründüğü hâlde oraya
zayıf olduğu için değil, kelimesi birebir geçtiği için gelmiş durumda.

**Ölçüm iddiaları açısından en önemli sonuç:** benchmark'ta kullanılan 11 RAG
sorusunun tamamının teslim edilen parça kümesi değişmedi. Dolayısıyla Bölüm
9.1–9.8'deki üretim sayıları — hız, gecikme, bellek, kalite — olduğu gibi geçerli.

#### Açık kalan

"İzin devri var mı?" hâlâ cevaplanamıyor ve BM25 bunu düzeltmiyor: soru "devri"
diyor, doküman "devredilir". Türkçe'nin sondan eklemeli yapısı birebir kelime
eşleşmesini yeniyor. Sabit uzunlukta önek kırpması da temiz ayrılmıyor —
"izni"/"izin" çiftinde ünlü düşmesi devreye giriyor. Doğru çözüm bir Türkçe
gövdeleyici (stemmer); kanıtsız eklenmedi, çünkü kapsam dışı sızmaya ne yaptığı
ölçülmeden eklenmiş bir gövdeleyici bu bölümün anlattığı hatanın aynısı olurdu.

### 9.11 Değerlendirme

| Kriter | Kazanan | Fark |
|---|---|---|
| Üretim hızı | `qwen3.5:9b` | %35 |
| İlk cevap süresi | `qwen3.5:9b` | ~1,0 s |
| Bellek | `qwen3.5:9b` | 1,6 GB az |
| Kalite / sadakat | — | ayırt edilemedi |
| Cevap uzunluğu | duruma göre | `qwen3.5` %36 daha uzun |

Bu iş için `qwen3.5:9b` açık ara daha uygun model: ölçülebilen her performans
ekseninde önde ve kalitede ölçülebilir bir kaybı yok. `gemma4:12b` ise sistemde
ikinci model olarak kalmalı — hem başka bir üreticinin modeliyle çapraz kontrol
imkânı veriyor hem de kısa cevap istenen durumlara daha iyi oturuyor.

---

## 10. Anahtar Kelimeler

Bu bölüm, rapor boyunca geçen 22 temel terimi kendi cümlelerimle tanımlıyor ve
her birini bir örnekle somutlaştırıyor. Örnekler öncelikle bu projeden verildi;
projede bilerek kullanılmayan terimlerde (LoRA, PEFT, Safetensors, ONNX) bu
durum açıkça belirtilip örnek ekosistemden ya da tercihin gerekçesinden
kuruldu.

**1. LLM (Large Language Model)**
Devasa metin yığınları üzerinde, bir metnin devamını tahmin etmeyi öğrenerek
eğitilmiş sinir ağı. *Örnek:* `gemma4:12b` bu projede kullanıcının sorusunu,
kendisine verilen İK dokümanı parçalarına bakarak yanıtlıyor.

**2. Embedding & Vector Space (Gömme ve Vektör Uzayı)**
Metnin anlamını sayı dizisine dönüştürme ve bu sayıların oluşturduğu çok
boyutlu uzay. Anlamca yakın metinler bu uzayda birbirine yakın konumlanır.
*Örnek:* Bu projede her doküman parçası 1024 boyutlu bir vektöre çevrilir;
"harcırah" sorusu ile "günlük yemek ödemesi" bölümü kelime örtüşmesi olmadan
eşleşir. Bu gücün bedeli de var: aynı yöntem, kelimesi birebir geçen bir parçayı
kaçırabiliyor. Ölçülmüş örneği ve çözümü Bölüm 9.9 ve 9.10'da.

**3. Inference (Çıkarım)**
Eğitimi bitmiş bir modeli kullanarak yeni bir girdiye cevap üretme işi.
Eğitimden farkı: model ağırlıkları değişmiyor, sadece okunuyor. *Örnek:*
Kullanıcı soruyu yazıp gönderdiğinde olan şey çıkarımdır.

**4. Training vs Fine-Tuning vs Instruction-Tuning**
*Training (eğitim):* Modeli sıfırdan, devasa veri ve devasa maliyetle
oluşturmak. *Fine-tuning (ince ayar):* Hazır bir modeli görece küçük, alana
özgü veriyle uzmanlaştırmak. *Instruction-tuning (talimat ayarı):* Modelin
verilen talimatları izleyip sohbet edebilmesi için özel olarak ayarlanması.
*Örnek:* Bu projede bunların hiçbiri yapılmadı; şirket bilgisi modele RAG ile,
tam soru sorulduğu anda verildi. Bu yol ince ayardan hem çok daha ucuz hem de
doküman güncellendiğinde değişiklik anında yansıyor.

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
Model metni kelime kelime değil, "token" denen parçalar hâlinde işliyor.
Tokenizer da metni bu parçalara ayıran bileşen; BPE (Byte-Pair Encoding) sık
rastlanan karakter dizilerini birleştirerek kendine bir sözlük çıkarıyor.
*Örnek:* Türkçe sondan eklemeli olduğu için "çalışabileceğiniz" gibi bir kelime
birkaç token'a bölünüyor — aynı içerikteki İngilizce metne kıyasla Türkçenin
daha fazla token yemesinin sebebi bu.

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
*Latency:* Tek bir isteğin kaç saniyede yanıtlandığı. *Throughput:* Birim
zamanda toplam kaç iş çıktığı. İkisi her zaman birlikte iyileşmiyor; toplu
işleme throughput'u yükseltirken latency'yi kötüleştirebiliyor.
*Örnek:* Bu raporda TTFT gecikmeyi, token/s ise iş hacmini temsil ediyor.

**14. Context Window / Context Length (Bağlam Penceresi)**
Modelin aynı anda "gözünün önünde tutabildiği" toplam token sayısı; sistem
prompt'u, sohbet geçmişi ve verilen dokümanların hepsi bu bütçeden yiyor.
*Örnek:* Her iki model de 256K token bağlam destekliyor. Buna rağmen bu projede
her soruda yalnızca en alakalı 4 doküman parçası gönderiliyor — bağlam yetmediği
için değil, alakasız metin eklemek cevabın kalitesini düşürdüğü için.

**15. Open Source Model vs Proprietary Model**
Ağırlıklarını indirip kendi donanımınızda çalıştırabildiğiniz modeller açık
kaynak; yalnızca API üzerinden ulaşabildikleriniz kapalı devre.
*Örnek:* Bu projenin tamamen çevrimdışı çalışabilmesinin sebebi, ağırlıkları
açık modeller kullanılması.

**16. API vs Local Deployment**
Modeli bir sağlayıcının sunucusunda çağırmakla kendi donanımınızda çalıştırmak
arasındaki tercih. İlki hemen başlıyor ama veriyi dışarı gönderiyor; ikincisi
donanım istiyor ama veriyi içeride tutuyor. *Örnek:* Bu projede lokal kurulum
seçildi ve Bölüm 4.4'te bunun **daha pahalı** olduğu ölçüldü — aynı sınıf bir
model bulutta yılda ~564 TL tutarken lokal kurulumun sadece elektriği
1.700–5.800 TL. Tercihin gerekçesi maliyet değil, İK verisinin şirket dışına
hiç çıkmaması.

**17. LoRA (Low-Rank Adaptation)**
Modelin bütün ağırlıklarını yeniden eğitmek yerine, yanına eklenen küçük matris
çiftlerini eğiterek uyarlama yöntemi. Maliyeti tam ince ayarın çok altında ve
ortaya çıkan adaptör dosyası birkaç yüz MB'ta kalıyor. *Örnek:* Bu projede
kullanılmadı — şirket bilgisi modele RAG ile verildi. Kullanılsaydı, 6,6 GB'lık
`qwen3.5:9b` checkpoint'i olduğu gibi kalır, yanına birkaç yüz MB'lık bir
adaptör dosyası eklenirdi.

**18. PEFT (Parameter-Efficient Fine-Tuning)**
LoRA gibi, modelin sadece küçük bir bölümünü eğiterek uyarlama yapan
yöntemlerin genel adı. *Örnek:* Bu projede hiçbir PEFT yöntemi uygulanmadı;
madde 4'teki gerekçeyle RAG tercih edildi — İK dokümanı güncellendiğinde
değişikliğin anında yansıması için modelin hiç eğitilmemesi gerekiyordu.

**19. Hugging Face Model Hub & Repositories**
Açık kaynak modellerin, veri kümelerinin ve demoların yayımlandığı merkezi
platform. Ollama kütüphanesindeki modellerin büyük çoğunluğu aslında buradan
geliyor. *Örnek:* Bu projedeki modellerin orijinalleri de Hugging Face'te
yayımlanıyor; quantize edilmiş GGUF sürümleri için `bartowski`, `unsloth` ve
`lmstudio-community` hesapları takip ediliyor (Bölüm 6.3). `ollama pull` bu
adımı kullanıcının üstünden alıyor.

**20. GGUF, Safetensors, ONNX (Model Formatları)**
*GGUF:* llama.cpp ekosisteminin quantize model formatı; Ollama ve LM Studio
bunu kullanıyor. *Safetensors:* Hugging Face'in güvenli ağırlık formatı —
açarken kod çalıştırma riski taşımıyor. *ONNX:* Farklı çalışma zamanları
arasında taşınabilirlik için tasarlanmış format. *Örnek:* Bu projede yalnızca
GGUF kullanıldı; `ollama pull qwen3.5:9b` komutunun indirdiği dosya bu
formatta. Apple'a özel MLX sürümleri yerine GGUF'un seçilmesinin sebebi
kurulumun Windows/CUDA tarafına da taşınabilmesiydi (Bölüm 6.1).

**21. Serving (Modeli Servis Etme) & Batch Inference**
*Serving:* Modeli bir API arkasında sürekli açık ve erişilebilir tutmak.
*Batch inference:* Birden fazla isteği tek seferde işleyip verimi artırmak.
*Örnek:* Bu projede serving katmanı Ollama; ingest sırasında doküman parçaları
16'lı gruplar hâlinde gömülerek batch işlemeden faydalanılıyor.

**22. Zero-shot, One-shot, Few-shot Learning**
Modele hiç örnek vermeden (*zero-shot*), tek örnek vererek (*one-shot*) ya da
birkaç örnek vererek (*few-shot*) görev tarif etme.
*Örnek:* Bu projedeki İK asistanı zero-shot çalışıyor: modele örnek soru-cevap
çiftleri verilmiyor, sadece kurallar ve ilgili doküman parçaları gönderiliyor.
