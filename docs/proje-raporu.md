# İK Asistanı — Proje Raporu

**Hazırlayan:** Yiğit Erdoğan
**Tarih:** 22 Temmuz 2026
**Kapsam:** Şirket dokümanlarındaki soruları yanıtlayan, dışarıya tek bir veri
bile göndermeyen lokal bir soru-cevap prototipi.

Ölçüm metodolojisi ve donanım araştırması ayrı dosyadadır:
[`arastirma-raporu.md`](./arastirma-raporu.md). Kurulum adımları için
[`README.md`](../README.md).

---

## 1. Problem

Bir çalışan "babalık izni kaç gün?" diye merak ettiğinde bugün iki seçeneği
var: İK'ya sormak ya da doğru PDF'i bulup içinde aramak. Üçüncü bir yol gibi
duran "ChatGPT'ye soruvereyim" ise aslında şirketin iç yönetmeliğini başka
birinin sunucusuna göndermek demek — o dokümanların içinde maaş bantları, izin
hakları ve prosedürler duruyor.

Kurulan sistem bu ikilemi ortadan kaldırıyor: ne soru ne de doküman makinenin
dışına çıkıyor. İnternet kablosunu çekseniz bile çalışmaya devam ediyor.

## 2. Sistem nasıl çalışıyor?

```
Soru
 └─> Embedding (qwen3-embedding:0.6b)     ~90 ms
      └─> ChromaDB · kosinüs benzerliği · en iyi 4 parça
           └─> Benzerlik eşiği (0,46)
                ├─ hiçbiri geçemedi ──> "Bu bilgi dokümanlarda yok"  (model hiç çağrılmaz)
                └─ geçenler
                     └─> BM25 kelime araması (+0,03 ms)
                          nadir bir kelime birebir eşleşiyorsa 1 parça daha ekler
                           └─> Sistem prompt'u + bulunan parçalar
                                └─> LLM (qwen3.5:9b | gemma4:12b)
                                     └─> Akışlı cevap + kaynak kartları
                                          └─> Takip sorusu çipleri (ek model çağrısı yok)
```

İşin can alıcı ayrıntısı şu: **eşik kontrolü modelden önce geliyor.** Cevabı
dokümanlarda olmayan bir soru geldiğinde modele "lütfen uydurma" diye rica
edilmiyor; model o soruyu zaten hiç görmüyor. Böylece uydurma riski modelin iyi
niyetine değil, sistemin kurgusuna bağlanmış oluyor.

İkinci savunma hattı sistem prompt'u (`app/prompts/system_tr.txt`): eşiği kıl
payı geçen zayıf bir bağlam geldiğinde modelin boşlukları kendi kafasından
doldurmasını engelliyor.

Şemadaki BM25 kolu sonradan eklendi ve bilinçli olarak **dar** tutuldu. Yalnızca
vektör araması zaten en az bir parça bulduğunda çalışıyor; hiçbir parça eşiği
geçemediyse hiç devreye girmiyor. Yani yukarıdaki "model hiç çağrılmaz" garantisi
aynen duruyor — kelime eşleşmesi sert reddi yumuşak redde çeviremiyor. Neden
böyle kurulduğu ve neyi düzelttiği araştırma raporu Bölüm 9.10'da.

## 3. Bileşenler ve seçim gerekçeleri

| Katman | Seçim | Neden |
|---|---|---|
| Çalıştırma | Ollama 0.32 (HTTP API) | Tek komutla model yönetimi, harici API yok, `/api/ps` ile ölçülebilir bellek raporu |
| Sohbet modeli | `qwen3.5:9b` (+ `gemma4:12b`) | Ölçümle seçildi — bkz. araştırma raporu Bölüm 9 |
| Embedding | `qwen3-embedding:0.6b` | 1024 boyut, 32K bağlam, çok dilli; Türkçe sondan eklemeli olduğu için geniş bağlam gerekli |
| Vektör DB | ChromaDB (kalıcı, gömülü) | Docker/servis gerektirmez, diske yazar, metadata filtreleme var |
| Backend | Python 3.12 + FastAPI | Async streaming (SSE), Pydantic doğrulama, otomatik OpenAPI |
| Frontend | Vite + React + TypeScript (`strict`) | Tip güvenliği; SSE akışını doğrudan tüketir |
| Arayüz | Tailwind + shadcn/ui | Erişilebilir bileşen tabanı, tema desteği |

### Neden LangChain / LlamaIndex kullanılmadı?

Bütün RAG hattı topu topu 400 satır. Hazır bir kütüphane bu satırları gözden
gizlerdi, ama karşılığında iki şeyi de götürürdü: ölçüm noktalarına doğrudan
erişim (TTFT, `eval_count`, arama süresi) ve prompt'un modele tam olarak hangi
biçimde gittiğini görebilmek. Benchmark'ın bütün geçerliliği "her modele harfi
harfine aynı istek gidiyor" varsayımına dayandığı için bu görünürlük pazarlık
konusu değildi.

Gerçek bir üretim sisteminde — birden fazla veri kaynağı, yeniden sıralama
(reranking), araç kullanımı — tercih büyük ihtimalle farklı olurdu.

### Neden ChromaDB?

Kurumsal ölçekte Qdrant ya da pgvector daha doğru tercihler. Ama burada elimizde
37 parçalık bir indeks var; ayrı bir servis veya Docker katmanı, karşılığında
hiçbir şey kazandırmadan kurulum yükü getirirdi. Vektör şeması standart olduğu
için ileride taşımak da zor değil.

## 4. Bilgi tabanı ve parçalama

Bilgi tabanı, kurgusal bir şirket (NovaTek Yazılım A.Ş.) için yazılmış dört
Türkçe İK dokümanıdır — izin politikası, çalışma düzeni, masraf ve yan haklar,
işe giriş ve oryantasyon. Gerçek bir şirketin belgesi kullanılmamıştır.

Parçalama (`app/chunking.py`) markdown başlık hiyerarşisini takip ediyor: her
parça hangi belgenin hangi başlığından geldiğini yanında taşıyor ve tablolar
ortadan bölünmüyor. Sebebi gayet pratik: "harcırah günlük 750 TL" bilgisi bir
tablo satırında duruyor; o satır ikiye bölünürse parça anlamını tamamen
kaybediyor.

| Ayar | Değer |
|---|---|
| Parça boyutu | ~500 token |
| Örtüşme | 75 token (%15) |
| İndekslenen parça | 37 |
| Getirilen parça (top-k) | 4 |

## 5. Benzerlik eşiğinin kalibrasyonu

Eşik değeri kafadan atılmadı, ölçülerek bulundu
(`bench/calibrate_threshold.py`). 28 etiketli soru — 19 tanesinin cevabı
dokümanlarda var, 9 tanesinin yok — sisteme tek tek soruluyor ve her birinin en
yüksek benzerlik skoru kaydediliyor.

Bu kalibrasyon iki kez yapıldı ve ikincisi birincisini çürüttü. İlk turda
yalnızca benchmark setindeki uzun, düzgün kurulmuş sorular kullanıldı; sonuç
0,52 çıktı ve iki küme arasında 0,061'lik tertemiz bir boşluk vardı. Ama o
boşluk sorunun yazılış biçiminden geliyordu: gerçek bir çalışan "Yurt içi
seyahatte günlük yemek harcırahı ne kadar?" diye yazmıyor, **"Harcırah ne
kadar?"** diye yazıyor. Kısa sorular sete eklenince tablo değişti:

| | Aralık |
|---|---|
| Kapsam içi (19 soru) | **0,468** – 0,708 |
| Kapsam dışı (9 soru) | 0,278 – **0,501** |

İki küme artık iç içe geçmiş durumda. En düşük kapsam içi soru ("Harcırah ne
kadar?" — 0,468), en yüksek kapsam dışı sorunun (hisse opsiyonu — 0,501)
*altında* kalıyor. Yani tek bir eşikle bu ikisini temiz biçimde ayırmak mümkün
değil; geriye "hangi hatayı göze alıyoruz?" sorusu kalıyor:

| Eşik | Kaçırılan doğru soru | Sızan kapsam dışı |
|---|---|---|
| 0,44 | 0/19 | 3/9 |
| **0,46 — seçilen** | **0/19** | **2/9** |
| 0,48 | 2/19 | 1/9 |
| 0,52 | 4/19 | 0/9 |

Seçim 0,46 oldu: hiçbir doğru soruyu kaçırmayan **en yüksek** değer. Gerekçesi
şu: bu iki hatanın ağırlığı aynı değil. Cevabı dokümanda yazan bir soruyu
reddetmek geri dönüşü olmayan bir hata — kullanıcı "bu sistem bilmiyormuş" diye
öğreniyor ve bir daha sormuyor. Eşiği geçip içeri sızan kapsam dışı bir soru
ise henüz kaybedilmiş değil; ikinci savunma hattına, sistem prompt'una düşüyor
ve orada reddedilebiliyor. Kısacası eşik, sonradan yakalanabilecek hatayı
yakalanamayacak hataya tercih edecek şekilde ayarlandı.

Somut örnek: 0,52'deyken sistem şu dört soruyu — dördünün de cevabı
dokümanlarda yazdığı hâlde — geri çeviriyordu: *"Harcırah ne kadar?"*,
*"Haftada kaç gün ofisteyim?"*, *"Babalık izni kaç gün?"*, *"Eğitim bütçesi ne
kadar?"*.

Buradan çıkan asıl ders eşiğin kendisi değil: **bir kalibrasyon seti, sistemin
gerçekte karşılaşacağı soruları temsil etmiyorsa ölçtüğü şey de gerçek
değildir.** Doküman seti ya da kullanıcıların soru sorma biçimi değişirse
kalibrasyon yeniden çalıştırılmalı; 0,46 bu veri setine ait bir sayı, evrensel
bir sabit değil.

> **Bu metriğin göremediği şey.** Tablodaki "kaçırılan doğru soru" sütunu
> yalnızca sorunun eşiği geçip geçmediğine bakıyor; gelen parçanın içinde cevap
> *var mı yok mu* ona bakmıyor. Aradaki fark önemli: 0,46'da "0/19 kaçırma"
> yazıyor, ama gerçekte iki soru doğru parçaya ulaşamıyordu.

Bu körlük sonradan kapatıldı. `bench/eval_retrieval.py` her kapsam içi soruya
cevabı fiilen içeren parçayı **gold** olarak etiketliyor ve üç şeyi ayrı ayrı
ölçüyor: gold parçanın sıralamadaki yeri, ilk 4'e girip girmediği (Recall@4) ve
modele fiilen ulaşıp ulaşmadığı. Bu araç kurulur kurulmaz eşiğin hiç göremediği
bir vaka çıktı — *"İzin devri var mı?"*, skoru 0,520 ile eşiğin **üstünde** ama
gold parça 5. sırada olduğu için modele hiç varmıyor.

Eşik değeri (0,46) bu çalışmada **değiştirilmedi.** Değiştirmek, ölçümle gelmiş
bir sayıyı ve yukarıdaki bütün gerekçeyi geçersiz kılardı. Bunun yerine aramaya
kelime tabanlı ikinci bir kol eklendi; kol yalnızca *ekleme* yaptığı için
tablodaki iki hata sütunu da olduğu gibi geçerli kalıyor. Ayrıntı ve öncesi/
sonrası sayıları araştırma raporu Bölüm 9.10'da.

## 6. Performans özeti

Ayrıntılar ve nasıl ölçüldüğü araştırma raporunun 9. bölümünde. Birbirinden
bağımsız üç temiz koşunun özeti şöyle:

| Metrik | `qwen3.5:9b` | `gemma4:12b` |
|---|---|---|
| Üretim hızı (token ağırlıklı) | **37,35 ± 0,46** tok/s | 27,65 ± 0,27 tok/s |
| İlk cevap (TTFT, medyan) | **1.941 ms** | 2.978 ms |
| Bellek | **6,29 GB** | 7,85 GB |
| Kalite | 14/14 | 14/14 |
| Kaynağa sadakat | 11/11 | 11/11 |

Eşik 0,46'ya çekildikten sonra alınan dördüncü temiz koşu aynı sayıları
bağımsız olarak doğruladı: 38,07 ve 27,79 tok/s. Toplamda altı koşu alındı;
ikisi, makinede başka bir uygulama 23 GB'lık bir model çalıştırdığı için
kirlendi — harness bunu kendi uyarı mekanizmasıyla yakaladı (araştırma raporu
Bölüm 9.4).

Belge arama medyanı ~90 ms. Yani kullanıcının ekrana bakıp beklediği sürenin
neredeyse tamamı modelin cevabı yazma süresi; arama kısmı fark edilmiyor bile.

İki modelin de kaliteden tam puan alması "bu modeller eşit" demek değil;
**test setinin ikisini birbirinden ayırt edemediği** anlamına geliyor
(bkz. Bölüm 8).

Yukarıdaki tablo cevabı ölçüyor; aramanın kendisi ayrı ölçülüyor ve modelden
bağımsız:

| Arama metriği (19 gold etiketli soru) | Yalnız vektör | + kelime kolu |
|---|---|---|
| Recall@4 (sıralama) | 0,895 | 0,895 |
| **Cevabına ulaşan soru** | **17/19** | **18/19** |
| Kapsam dışına eklenen parça | — | 0 |

Kelime kolu yoğun aramaya hiç dokunmadığı için Recall@4 aynı kalıyor; değişen
tek şey, sıralamanın altında kalan doğru parçanın artık modele ulaşabilmesi.

**Eşik düşünce ne oldu?** 0,46'da kapsam dışı kontrol sorularından biri artık
eşiği geçip modele ulaşıyor — o soruda birinci savunma hattı devrede değil.
Üç koşuda, iki modelde, yani altı denemenin altısında da model soruyu reddetti
ve sistem prompt'undaki cümleyi kelimesi kelimesine yazdı. İkinci katman ilk
kez gerçekten sınandı ve tuttu.

**Peki düşünme (reasoning) modu açılsın mı? Hayır.** Ayrı bir turda açık hâliyle
de ölçtük: ilk cevabın gelmesi 1,9 saniyeden 28,5 saniyeye çıktı, token tüketimi
4-5 katına fırladı, kalite ise yerinde saydı — zaten 14/14'tü, yükseltecek yer
yoktu. Üstelik `qwen3.5` varsayılan 1.024 token bütçesinde düşünmeyi bitiremeyip
cevaba hiç başlayamadı. Karşılıklı konuştuğunuz bir asistanda bu takas
savunulacak gibi değil; ayrıntısı araştırma raporu Bölüm 9.7'de.

## 7. Gizlilik ve güvenlik

- **Dışarıya giden tek bir istek yok.** Bütün trafik `localhost:11434` (Ollama)
  ile `localhost:8000` (API) arasında gidip geliyor. Ağ kablosunu çekseniz
  sistem çalışmaya devam eder.
- **Gizli bilgi kodun içinde durmuyor.** Bütün ayarlar ortam değişkeninden
  okunuyor (`app/config.py`, `pydantic-settings`); depoda sadece
  `.env.example` var.
- **Prompt'lar koda gömülü değil.** `app/prompts/` altında ayrı metin
  dosyalarında duruyorlar; birini değiştirmek için kodu yeniden dağıtmaya gerek
  yok.
- **Hata mesajları ham API cevabını sızdırmıyor.** Ollama tarafında bir sorun
  olduğunda kullanıcı "model çekili mi?" tarzında bir mesaj görüyor; isteğin
  gövdesi (içinde doküman metni olabilir) hiçbir zaman ekrana yansımıyor.

## 8. Sınırlar

Prototipin bugün yapamadıkları, olduğu gibi:

1. **Yetkilendirme yok.** Sisteme giren herkes her dokümanı sorgulayabiliyor.
   Gerçek bir İK kurulumunda parça bazlı erişim kontrolü (departman, kademe)
   şart olur.
2. **Türkçe ekler kelime aramasını hâlâ yenebiliyor.** *"Babalık izni kaç gün?"*
   vakası kapatıldı: aramaya eklenen BM25 kolu, cevabı birebir içeren parçayı
   (12. sıra, 0,419 skor) modele ulaştırıyor ve soru artık doğru cevaplanıyor.
   Ama aynı yöntem *"İzin devri var mı?"* sorusunu kurtaramıyor — soru "devri"
   diyor, doküman "devredilir". Birebir token eşleşmesi bunları eşleştiremiyor,
   sabit uzunlukta önek kırpması da temiz ayırmıyor ("izni"/"izin" çiftinde ünlü
   düşmesi var). Doğru çözüm bir Türkçe gövdeleyici; kapsam dışı sızmaya ne
   yaptığı ölçülmeden eklenmedi. Sayılar araştırma raporu Bölüm 9.9 ve 9.10'da.
3. **Test seti tavana vurdu.** 14 soru iki modeli birbirinden ayıramıyor ve
   yukarıdaki kusurların ikisini de fark edemedi, çünkü setteki soruların hepsi
   uzun ve düzgün yazılmış. Bu körlüğün bir kısmı kapandı: `eval_retrieval.py`
   artık aramayı cevaptan ayrı ölçüyor ve gizli kalmış ikinci kusuru o buldu.
   Ama **cevap** kalitesi tarafı hâlâ tavanda. Sete çok adımlı çıkarım,
   birbiriyle çelişen kaynaklar ve tablo okuma eklenmeli.
4. **Kalite ölçümü anahtar kelimeye bakıyor.** Mekanik ve tekrarlanabilir
   olması iyi, ama cevabın akıcı olup olmadığını ya da gereksiz uzadığını
   ölçmüyor.
5. **Doküman güncelleme işi elle yapılıyor.** Yeni doküman eklendiğinde
   `python -m app.ingest` komutunu kendiniz çalıştırmanız gerekiyor; otomatik
   izleme veya yeniden indeksleme yok.
6. **Sistem tek kullanıcıya göre kurgulandı.** Aynı anda birden fazla istek
   gelirse Ollama bunları sıraya alıyor; çok kullanıcılı kullanım hiç
   ölçülmedi.

## 9. Kurumsal öneri

**Donanım.** Asıl darboğaz bellek değil, modelin yazma hızı. Seçilen model
6,3 GB yer kaplıyor; 16 GB unified memory'li bir Mac ya da 16 GB VRAM'li bir
NVIDIA kartı bu iş için fazlasıyla yeter. Türkiye fiyatlarıyla (22.07.2026):
Mac mini M4 24 GB ≈ 77.000 TL, RTX 5060 Ti 16GB'lı hazır bir sistem ≈
48.000–62.000 TL. Bütçeyi sadece ekran kartı üzerinden kurmak yanıltıcı olur —
kart, çalışır bir sistemin ancak %60–75'i.

**Bu işin maliyet gerekçesi tutmuyor — ve bunu bilerek yazıyoruz.** Fiyat
araştırması (araştırma raporu Bölüm 4.4) beklediğimizin tam tersini gösterdi:
aynı sınıftaki bir modeli bulutta çalıştırmak bu kullanım hacminde yılda ~600 TL
tutuyor; lokal kurulumun sadece elektriği 1.700–5.800 TL. Donanımın peşin
parasını hiç saymasak bile lokal kurulum bu kıyasta kendini amorti etmiyor.
Dolayısıyla öneri şu cümle üzerine kurulmalı: **lokal çözüm bir tasarruf kalemi
değil, parası ölçülebilen bir gizlilik primi.** Özlük dosyaları, maaş bantları
ve performans verileri KVKK kapsamında; bu veri için o primi ödemek kolayca
savunulur. Genel amaçlı bir sohbet asistanı için savunulmaz.

**Model.** Birincil model `qwen3.5:9b` olsun; `gemma4:12b` de sistemde ikinci
model olarak kalsın — farklı bir üreticiden geliyor, çapraz kontrol imkânı
veriyor ve kısa cevap istenen durumlara daha iyi oturuyor.

**Dağıtım.** Şirket içinde tek bir sunucuda Ollama + FastAPI çalışsın,
kullanıcılar tarayıcıdan bağlansın. Böylece model tek bir yerde durur, her
masaüstüne ayrı ayrı kurulmaz.

**Peki lokal çözüm nerede yanlış tercih olur?** Karmaşık, çok adımlı akıl
yürütme; uzun kod üretimi; aynı anda çok sayıda kullanıcı — bu işlerde 10 GB
sınıfı bir modelin sınırı çok çabuk görünür. Lokal kurulum, kapsamı belli ve
verisi hassas işler için doğru araç; İK soru-cevabı da tam olarak bu tarife
uyuyor.

## 10. Doğrulama

```bash
# Backend
cd backend
uv run ruff check . && uv run ruff format --check .   # temiz
uv run pytest                                          # 50/50

# Frontend
cd frontend
npm run typecheck && npm test && npm run build         # temiz · 19/19 · temiz

# Ölçüm
cd backend
uv run python -m bench.run_bench --output run-yeni.json   # model karşılaştırması
uv run python -m bench.eval_retrieval                     # arama kalitesi
uv run python -m bench.calibrate_threshold                # eşik taraması
```

Takip sorusu çipleri değiştirilecekse sıra şu: `uv run python -m
app.gen_suggestions` taslakları `data/suggested-questions.yaml`'a yazar, dosya
**elle gözden geçirilip** commit edilir, sonra `uv run python -m app.ingest`
indeksi yeniden kurar. Ara adım atlanabilir değil: bu çalışmada üretilen 74
sorunun 20'si yeniden yazıldı, biri tamamen atıldı (*"Nöbet primi ne kadar?"* —
doküman primi vaat ediyor ama tutarı hiç yazmıyor, yani asistan kendi önerdiği
soruyu reddederdi).
