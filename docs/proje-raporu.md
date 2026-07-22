# İK Asistanı — Proje Raporu

**Hazırlayan:** Yiğit Erdoğan
**Tarih:** 22 Temmuz 2026
**Kapsam:** Staj dokümanının 11. bölümünde istenen bitirme görevi — şirket
dokümanlarını yanıtlayan, dışarıya hiçbir veri göndermeyen lokal soru-cevap
prototipi.

Ölçüm metodolojisi ve donanım araştırması ayrı dosyadadır:
[`arastirma-raporu.md`](./arastirma-raporu.md). Kurulum adımları için
[`README.md`](../README.md).

---

## 1. Problem

Bir çalışanın "babalık izni kaç gün?" sorusuna cevap bulması bugün İK'ya
sorması ya da doğru PDF'i açıp taraması demektir. Bu soruyu ChatGPT'ye sormak
ise şirketin iç yönetmeliğini üçüncü bir tarafa göndermek anlamına gelir —
İK dokümanları maaş bantları, izin hakları ve prosedürler içerir.

Kurulan sistem bu ikilemi ortadan kaldırır: sorular da dokümanlar da makineden
çıkmaz. Ağ bağlantısı kesilse bile sistem çalışmaya devam eder.

## 2. Sistem nasıl çalışıyor?

```
Soru
 └─> Embedding (qwen3-embedding:0.6b)     ~90 ms
      └─> ChromaDB · kosinüs benzerliği · en iyi 4 parça
           └─> Benzerlik eşiği (0,46)
                ├─ hiçbiri geçemedi ──> "Bu bilgi dokümanlarda yok"  (model hiç çağrılmaz)
                └─ geçenler ──> Sistem prompt'u + bulunan parçalar
                                 └─> LLM (qwen3.5:9b | gemma4:12b)
                                      └─> Akışlı cevap + kaynak kartları
```

Sistemin ayırt edici tarafı **eşik kontrolünün modelden önce** gelmesidir.
Cevabı dokümanlarda olmayan bir soru sorulduğunda modelden "uydurma" diye rica
edilmez; model o soruyu hiç görmez. Bu, halüsinasyon riskini modelin iyi
niyetine değil mimariye bağlar.

İkinci savunma katmanı sistem prompt'udur (`app/prompts/system_tr.txt`):
eşiği kıl payı geçen zayıf bir bağlamda modelin boşluk doldurmasını engeller.

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

Bütün RAG hattı yaklaşık 400 satır. Bir çerçeve bu satırları gizlerdi ama
karşılığında iki şey kaybettirirdi: ölçüm noktalarına doğrudan erişim (TTFT,
`eval_count`, retrieval süresi) ve prompt'un modele tam olarak hangi biçimde
gittiğinin görünürlüğü. Benchmark'ın geçerliliği "her modele baytı baytına aynı
istek" varsayımına dayandığı için bu görünürlük pazarlık konusu değildi.

Üretim ölçeğinde — çok kaynaklı ingest, yeniden sıralama (reranking), araç
kullanımı — tercih farklı olurdu.

### Neden ChromaDB?

Qdrant ve pgvector kurumsal ölçekte daha uygun seçeneklerdir. Bu prototipte
37 parçalık bir indeks var; ayrı bir servis veya Docker katmanı, kazandırdığı
hiçbir şey karşılığında kurulum yükü getirirdi. Vektör şeması standart olduğu
için taşınma maliyeti düşüktür.

## 4. Bilgi tabanı ve parçalama

Bilgi tabanı, kurgusal bir şirket (NovaTek Yazılım A.Ş.) için yazılmış dört
Türkçe İK dokümanıdır — izin politikası, çalışma düzeni, masraf ve yan haklar,
işe giriş ve oryantasyon. Gerçek bir şirketin belgesi kullanılmamıştır.

Parçalama (`app/chunking.py`) markdown başlık hiyerarşisini takip eder:
her parça hangi belgenin hangi başlığından geldiğini metadata olarak taşır ve
tablolar bölünmez. Bunun nedeni pratik: "harcırah günlük 750 TL" bilgisi bir
tablo satırındadır ve satır ortadan bölünürse parça anlamını kaybeder.

| Ayar | Değer |
|---|---|
| Parça boyutu | ~500 token |
| Örtüşme | 75 token (%15) |
| İndekslenen parça | 37 |
| Getirilen parça (top-k) | 4 |

## 5. Benzerlik eşiğinin kalibrasyonu

Eşik değeri sezgiyle değil ölçümle belirlendi (`bench/calibrate_threshold.py`).
28 etiketli soru — 19 kapsam içi, 9 kapsam dışı — sisteme sorulur ve her birinin
en yüksek benzerlik skoru kaydedilir.

Bu kalibrasyon iki kez yapıldı ve ikincisi birincisini çürüttü. İlk turda
yalnızca benchmark setinin uzun, düzgün kurulmuş soruları kullanıldı; sonuç
0,52 ve iki küme arasında 0,061'lik tertemiz bir boşluktu. Ancak o boşluk
soru üslubunun bir eseriydi: gerçek bir çalışan "Yurt içi seyahatte günlük
yemek harcırahı ne kadar?" diye yazmaz, **"Harcırah ne kadar?"** diye yazar.
Kısa sorular sete eklendiğinde tablo değişti:

| | Aralık |
|---|---|
| Kapsam içi (19 soru) | **0,468** – 0,708 |
| Kapsam dışı (9 soru) | 0,278 – **0,501** |

İki küme artık örtüşüyor. En düşük kapsam içi soru ("Harcırah ne kadar?" —
0,468) en yüksek kapsam dışı sorunun (hisse opsiyonu — 0,501) *altında*
kalıyor. Bu, tek bir eşiğin ikisini birden ayıramayacağı anlamına gelir; geriye
hangi hatanın satın alınacağı seçimi kalır:

| Eşik | Kaçırılan doğru soru | Sızan kapsam dışı |
|---|---|---|
| 0,44 | 0/19 | 3/9 |
| **0,46 — seçilen** | **0/19** | **2/9** |
| 0,48 | 2/19 | 1/9 |
| 0,52 | 4/19 | 0/9 |

Seçim 0,46: sıfır kaçırma sağlayan **en yüksek** değer. Gerekçe, iki hatanın
simetrik olmamasıdır. Cevabı dokümanda yazan bir soruyu reddetmek nihai bir
hatadır — kullanıcı "bu sistem bilmiyor" diye öğrenir ve bir daha sormaz.
Eşiği geçen kapsam dışı bir soru ise nihai değildir; ikinci savunma katmanına,
sistem prompt'una düşer ve orada reddedilebilir. Bu nedenle eşik, yakalanabilir
hatayı yakalanamaz hataya tercih edecek biçimde ayarlanmıştır.

Somut olarak 0,52'de sistem şu dört soruyu — dördünün de cevabı dokümanlarda
olduğu hâlde — reddediyordu: *"Harcırah ne kadar?"*, *"Haftada kaç gün
ofisteyim?"*, *"Babalık izni kaç gün?"*, *"Eğitim bütçesi ne kadar?"*.

Buradan çıkan asıl ders eşik değerinin kendisi değildir: **bir kalibrasyon
setinin, ölçtüğü sistemin karşılaşacağı girdiyi temsil etmesi gerekir.**
Doküman seti veya kullanıcı üslubu değişirse kalibrasyon yeniden
çalıştırılmalıdır; eşik veri setine bağlıdır, evrensel bir sabit değildir.

> **Metriğin sınırı.** Yukarıdaki tablodaki "kaçırılan doğru soru" sütunu,
> sorunun eşiği geçip geçmediğini ölçer — getirilen parçanın cevabı *içerip
> içermediğini* ölçmez. Bu ayrım önemlidir: 0,46'da "0/19 kaçırma" yazıyor,
> ancak elle yapılan denemede bir sorunun eşiği geçtiği hâlde doğru parçaya
> ulaşamadığı görüldü (Bölüm 8, madde 2).

## 6. Performans özeti

Ayrıntı ve metodoloji araştırma raporu Bölüm 9'dadır. Üç bağımsız temiz koşunun
özeti:

| Metrik | `qwen3.5:9b` | `gemma4:12b` |
|---|---|---|
| Üretim hızı (token ağırlıklı) | **37,35 ± 0,46** tok/s | 27,65 ± 0,27 tok/s |
| İlk cevap (TTFT, medyan) | **1.941 ms** | 2.978 ms |
| Bellek | **6,29 GB** | 7,85 GB |
| Kalite | 14/14 | 14/14 |
| Kaynağa sadakat | 11/11 | 11/11 |

Eşik 0,46'ya çekildikten sonra alınan dördüncü temiz koşu bu sayıları bağımsız
olarak yeniden üretti: 38,07 ve 27,79 tok/s. Toplam altı koşu alındı; ikisi,
makinede başka bir uygulamanın 23 GB'lık bir model çalıştırması yüzünden
kirlendi ve harness bunu kendi uyarı mekanizmasıyla yakaladı (araştırma raporu
Bölüm 9.4).

Belge arama medyanı ~90 ms; yani kullanıcının beklediği sürenin neredeyse
tamamı modelin üretim süresidir, arama değil.

Kalite skorlarının ikisinde de tam çıkması "iki model eşit" demek değildir;
**test setinin ayırt etme gücünün yetmediği** anlamına gelir (bkz. Bölüm 8).

**Eşik düşünce ne oldu?** 0,46'da kapsam dışı kontrol sorularından biri artık
eşiği geçip modele ulaşıyor — yani o soruda birinci savunma katmanı devrede
değil. Üç koşuda, iki modelde, altı denemenin altısında da model soruyu
reddetti ve sistem prompt'undaki cümleyi birebir üretti. İkinci katman ilk kez
gerçekten sınandı ve tuttu.

**Reasoning modu açılmalı mı? Hayır.** Ayrı bir turda düşünme modu açık
ölçüldü: ilk cevap süresi 1,9 saniyeden 28,5 saniyeye çıktı, token tüketimi
4-5 katına ulaştı, kalite ise artmadı — zaten 14/14'tü. Üstelik `qwen3.5`
varsayılan 1.024 token bütçesinde düşünmeyi bitiremeyip cevaba hiç
başlayamadı. Etkileşimli bir asistanda bu değiş tokuş savunulamaz; ayrıntı
araştırma raporu Bölüm 9.7'dedir.

## 7. Gizlilik ve güvenlik

- **Dışarıya giden istek yok.** Tüm trafik `localhost:11434` (Ollama) ve
  `localhost:8000` (API) arasındadır. Ağ kablosu çekilse sistem çalışır.
- **Gizli bilgi kaynak kodunda tutulmaz.** Tüm ayarlar ortam değişkeninden
  okunur (`app/config.py`, `pydantic-settings`); depoda yalnızca
  `.env.example` bulunur.
- **Prompt'lar kodda gömülü değildir.** `app/prompts/` altında ayrı metin
  dosyalarındadır; değiştirmek için kod dağıtımı gerekmez.
- **Hata mesajları ham API cevabı sızdırmaz.** Ollama hatası kullanıcıya
  "model çekili mi?" biçiminde döner, istek gövdesi (doküman metni içerebilir)
  asla yankılanmaz.

## 8. Sınırlar

Prototipin bugün yapamadıkları, dürüstçe:

1. **Yetkilendirme yok.** Herkes her dokümanı sorgulayabilir. Gerçek bir İK
   kurulumunda parça bazlı erişim kontrolü (departman, kademe) gerekir.
2. **Kısa sorularda arama sıralaması yetersiz kalabiliyor — ölçülmüş bir
   örnek var.** *"Babalık izni kaç gün?"* sorusunda cevabı birebir içeren
   parça (`Eş doğumu (babalık izni) | 10 iş günü`) 37 parça arasında ancak
   12. sırada, 0,419 skorla geliyor; ilk sıraya konuyla ilgisiz bir "izin
   bakiyesi" parçası yerleşiyor. Doğru parça top-4'e giremediği için model
   cevabı hiç görmüyor ve elindeki bağlama sadık kalarak reddediyor — hata
   modelde değil, aramada. Aynı soru dokümanın kendi sözcükleriyle
   sorulduğunda ("Eş doğumu izni kaç gün?") doğru parça 2. sıraya çıkıyor.
   Bu, yoğun vektör aramasının bilinen kelime dağarcığı uyuşmazlığı
   sorunudur; standart çözümü hibrit aramadır (BM25 + vektör). Ayrıntı ve
   sayılar araştırma raporu Bölüm 9.9'da.
3. **Test seti tavana vurdu.** 14 soru iki modeli ayırt edemiyor — üstelik
   yukarıdaki kusuru da göremedi, çünkü ölçüm setindeki soruların tamamı uzun
   ve düzgün kurulmuş. Çok adımlı çıkarım, çelişen kaynaklar, tablo okuma ve
   **kısa/eksik yazılmış sorular** eklenmeli.
4. **Kalite ölçümü anahtar kelime tabanlı.** Tekrarlanabilir ve mekaniktir,
   ama cevabın akıcılığını veya gereksiz uzunluğunu ölçmez.
5. **Doküman güncelleme akışı manuel.** Yeni doküman eklenince
   `python -m app.ingest` elle çalıştırılır; izleme/otomatik yeniden indeksleme
   yoktur.
6. **Tek kullanıcı varsayımı.** Eşzamanlı yük altında Ollama sıraya alır;
   çok kullanıcılı kullanım ölçülmemiştir.

## 9. Kurumsal öneri

**Donanım.** Belirleyici kısıt bellek değil üretim hızıdır. Seçilen model
6,3 GB tutuyor; 16 GB unified memory'li bir Mac veya 16 GB VRAM'li bir NVIDIA
kartı bu iş yükü için yeterlidir. Türkiye fiyatlarıyla (22.07.2026): Mac mini
M4 24 GB ≈ 77.000 TL, RTX 5060 Ti 16GB'lı hazır bir sistem ≈ 48.000–62.000 TL.
Ekran kartının tek başına alınması yanıltıcıdır — kart, çalışır bir sistemin
ancak %60–75'idir.

**Maliyet gerekçesi kurulamaz — ve bu bilinçli bir tespittir.** Fiyat
araştırması (araştırma raporu Bölüm 4.4) beklenenin tersini gösterdi: aynı
sınıftaki bir modeli bulutta çalıştırmak bu kullanım hacminde yılda ~600 TL
tutuyor; lokal kurulumun yalnızca elektriği 1.700–5.800 TL. Donanımın peşin
maliyeti hiç sayılmasa bile lokal kurulum bu kıyasta amorti etmiyor. Öneri bu
nedenle şöyle kurulmalıdır: **lokal çözüm bir tasarruf kalemi değil,
ölçülebilir bir gizlilik primidir.** Özlük dosyaları, maaş bantları ve
performans verileri KVKK kapsamındadır; bu veri kümesinde prim kolaylıkla
haklı çıkar. Genel amaçlı bir sohbet asistanında çıkmaz.

**Model.** `qwen3.5:9b` birincil; `gemma4:12b` ikinci model olarak sistemde
kalsın — farklı üretici, çapraz doğrulama imkânı ve kısa cevap tercih edilen
senaryolar için.

**Dağıtım.** Şirket içi tek bir sunucuda Ollama + FastAPI; kullanıcılar
tarayıcıdan bağlanır. Böylece model tek yerde tutulur, her masaüstüne
kurulmaz.

**Nerede lokal çözüm doğru araç değildir?** Karmaşık çok adımlı akıl yürütme,
uzun kod üretimi veya yüksek eşzamanlılık gerektiren işlerde 10 GB sınıfı bir
modelin sınırı hızla görünür. Lokal kurulum, kapsamı belli ve verisi hassas
işler için doğru araçtır — İK soru-cevabı tam olarak bu tanıma uyar.

## 10. Doğrulama

```bash
# Backend
cd backend
uv run ruff check . && uv run ruff format --check .   # temiz
uv run pytest                                          # 19/19

# Frontend
cd frontend
npm run typecheck && npm test && npm run build         # temiz · 5/5 · temiz

# Ölçüm
cd backend
uv run python -m bench.run_bench --output run-yeni.json
```
