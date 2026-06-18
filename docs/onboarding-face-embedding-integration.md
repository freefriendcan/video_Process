# Onboarding Yüz Fotoğrafı → Embedding Entegrasyon Raporu

**Konu:** `proactive-home-agent` dashboard'undaki onboarding sürecinde toplanan yeni
kullanıcı yüz fotoğraflarının `video_Process` reposunda embedding'e dönüştürülüp
saklanması için ileri seviye entegrasyon analizi ve uygulama mimarisi.

**Önerilen mimari:** `video_Process` = tek doğruluk kaynağı (single source of truth).

**Tarih:** 2026-06-18 · **Durum:** Analiz / tasarım raporu (kod değişikliği içermez).

---

## 1. Yönetici Özeti

Bugün iki sistem yüz tanımayı **birbirinden bağımsız ve uyumsuz** şekilde yapıyor:

- **`proactive-home-agent`** (backend, Raspberry Pi): Onboarding'de toplanan yüz fotoğraflarını
  **DeepFace / GhostFaceNet** ile embed edip **SQL veritabanında** (`User.face_embedding`, JSON)
  saklıyor.
- **`video_Process`** (Mac, canlı kamera hattı): Kimlik doğrulamayı **insightface ArcFace
  (`buffalo_l` / `w600k_r50.onnx`)** ile yapıyor; galeriyi **`data/embeddings/faces.pkl`**
  pickle dosyasında tutuyor.

Her iki embedding de 512 boyutlu olmasına rağmen **farklı modellerden** geldikleri için
**aynı uzayda değiller** — birinde üretilen vektör diğerinde anlamlı bir cosine benzerliği
vermez. Sonuç: onboarding'de "kaydedilen" yüz, canlı kameranın kullandığı galeride **hiç yer
almıyor**. Dashboard'da yüzünü tanıtan kullanıcı, oturma odasındaki kamera tarafından
**"Unknown"** olarak görülür.

Bu rapor, onboarding fotoğraflarını **canlı hattın kullandığı ArcFace galerisine** akıtarak bu
kopukluğu gidermeyi öneriyor. Tasarımın çekirdeği: `video_Process` içine **inbound bir enrollment
servisi** eklemek, onboarding fotoğraflarını backend üzerinden bu servise iletmek ve
**`video_Process`'in kendi mevcut embedding boru hattını yeniden kullanarak** `faces.pkl`'i
güncellemek. Böylece **tek bir embedding uzayı** (ArcFace 512-d) hem onboarding hem de canlı
tanıma için ortak doğruluk kaynağı olur.

> **Kilit fırsat:** `video_Process`'in enrollment mantığı (`scripts/enroll_faces.py`) ve
> `proactive-home-agent`'in ayarlarındaki kullanılmayan `database_path: "data/embeddings/faces.pkl"`
> referansı, bu birleşmenin **zaten amaçlanmış** olduğunu gösteriyor. Yeni embedding kodu
> yazmaya gerek yok; mevcut fonksiyonları bir HTTP yüzeyi arkasına almak yeterli.

---

## 2. Mevcut Durum Analizi (As-Is)

### 2.1 Onboarding'de yüz fotoğrafı toplama (frontend)

`frontend/src/components/onboarding/Step3Biometrics.tsx`

- Tarayıcı `getUserMedia({ video: true })` ile kamerayı açar.
- Kullanıcıdan **5 poz** ister: `front → left → right → up → down`. Her "Capture Photo"
  tıklamasında `canvas.toBlob(..., "image/jpeg", 0.9)` ile bir JPEG üretir.
- Kayıt aşamasında (`executeFinalSave`) fotoğrafları **tek tek** ve **ayrı isteklerle**
  backend'e gönderir:

```
POST {API_URL}/users/register   (Authorization: Bearer <token>)
  form-data: image_file=front.jpg   [+ opsiyonel audio_file=voice_sample.webm]
POST {API_URL}/users/register   image_file=left.jpg
POST {API_URL}/users/register   image_file=right.jpg
POST {API_URL}/users/register   image_file=up.jpg
POST {API_URL}/users/register   image_file=down.jpg
```

- Önce `front` (+ ses) gönderilir; başarısızsa tüm akış durur. Diğer 4 açı bir döngüde
  gönderilir ve hataları **yutulur** (`console.warn("Angle sync failed")`). Yani çoklu-açı
  "best effort"tür.
- **Önemli:** Bu uçların hiçbiri `video_Process`'e gitmez; hepsi proactive backend'e gider.

### 2.2 Backend kayıt ucu

`backend/api/routers/user_router.py` → `POST /users/register`

- Kimlik **JWT'den** alınır (`current_user`); form'da kullanıcı adı taşınmaz. Yani 5 açının
  hepsi otomatik olarak aynı `current_user.username`'e iliştirilir.
- `image_file` baytları okunup `vision_service.register_face(username, image_bytes)`'a verilir.

### 2.3 Backend embedding üretimi ve saklama (mevcut)

`backend/api/services/vision_service.py` → `VisionService` (singleton)

- Model: **`GhostFaceNet`** (`self.model_name`), **DeepFace** üzerinden.
- Dedektör: **`yolov8n-face.pt`** (`FaceDetector`, CPU).
- Hizalama: **kapalı** (`alignment_backend = "none"`, `DeepFace.represent(..., align=False)`).
- Akış (`register_face`):
  1. `cv2.imdecode` → en geniş 640px'e küçült.
  2. `detector.detect` → en büyük yüz kutusu.
  3. `_extract_padded_roi` (%30 padding) → yüz ROI.
  4. `_extract_embedding` → `DeepFace.represent(model_name="GhostFaceNet")` → 512-d vektör.
  5. Saklama: `User.face_embedding` (JSON) listesine **append**; liste 5'e ulaşınca en eskiyi
     `pop(0)` (pratikte **son ~4-5 poz**lik halka tampon).
  6. `load_faces_from_db()` ile bellek içi `known_faces` tazelenir.
- Eşleştirme (`recognize`): bellek içi galeriye karşı **cosine**, eşik
  `face_recognition.threshold = 0.45` (`vision_core/config/settings.py`).

`backend/database/models.py`

```python
class User(SQLModel, table=True):
    ...
    voice_embedding: Optional[List[float]] = Field(default=None, sa_column=Column(JSON))
    face_embedding:  Optional[List[float]] = Field(default=None, sa_column=Column(JSON))
```

> `face_embedding` aslında **liste-of-liste** olarak kullanılıyor (`List[List[float]]`):
> kullanıcı başına birden çok poz embedding'i.

### 2.4 video_Process embedding hattı (canonical)

`identification/face_identifier.py` → `FaceIdentifier`

- Model: **insightface `buffalo_l` → `w600k_r50.onnx`** (`get_model` + `prepare`).
- `embed_face(roi, keypoints)`: **5-nokta hizalama** (`face_align.norm_crop`, 112×112) →
  `get_feat` → **L2-normalize** → 512-d vektör. (Canlı `identify` ve enrollment **aynı** bu
  yolu kullanır; `for_enrollment` ile galeri olmadan yalnızca model yüklenir.)
- Galeri formatı: `dict[str, np.ndarray]`, değer `(N, 512)` float32, satırlar L2-normalize.
- Eşleştirme (`_best_gallery_match`): galeri matrisi `@ embedding` → maksimum cosine; eşik
  `face_match_cosine_threshold = 0.35` (`config.py`).
- **Galeri yalnızca `__init__`'te bir kez yüklenir** (`_load_gallery`); çalışma sırasında
  **hot-reload / dosya izleme yok**.

`scripts/enroll_faces.py` (offline CLI)

- `data/enroll/<label>/*.jpg` → `FaceDetector.detect` (en büyük yüz) → `crop_keypoints`
  → `FaceIdentifier.for_enrollment().embed_face` → kişi başına `(N,512)` → `pickle.dump`
  → `data/embeddings/faces.pkl`.

`scripts/capture_enrollment.py` (offline CLI)

- Webcam'den **kalite kapısından** (`FaceQualityGate.check`) geçen kareleri
  `data/enroll/<name>/` altına yazar; `--enroll` ile sonrasında `enroll_faces.py` çalıştırır.

### 2.5 Ağ topolojisi ve servisler-arası iletişim

- `video_Process` **yalnızca dışa-doğru (outbound) istemcidir**. `events/dispatcher.py`
  `requests.Session` ile Pi backend'e POST atar:
  - `POST http://{pi_ip}:{pi_port}/vision/identify`
  - `POST .../vision/update_presence`
  - `POST .../vision/fall_alert`
  - `POST .../vision/gesture`
- Tek **inbound** yüzeyi `streaming/vision_ws_server.py` (varsayılan `ws://0.0.0.0:5003`) —
  ama bu **tek yönlü yayın**dır (dashboard'a `VisionState` JSON'u akıtır), enrollment için
  uygun değildir.
- `config.py`'de `server_host="0.0.0.0"`, `server_port=5002` **tanımlı ama kullanılmıyor**
  (ayrılmış kapasite — yeni enrollment servisi için doğal yer).
- Backend tarafı, dış kameralardan kimlik/oturum olaylarını almaya **zaten hazır**:
  `backend/api/routers/vision_router.py` → `/vision/identify`, `/vision/update_presence`,
  `/vision/identity_event` (şema: `event_type`, `track_id`, `user`, `zone`, `ts_wall`…).

### 2.6 Kritik birleşme izi

`backend/api/services/vision_core/config/settings.py`:

```python
class FaceRecognitionConfig(BaseModel):
    model: str = "GhostFaceNet"
    threshold: float = 0.45
    database_path: str = "data/embeddings/faces.pkl"   # ← KULLANILMIYOR
    detector_model: str = "yolov8n-face.pt"
    alignment_backend: Literal["fan","deepface","none"] = "none"
```

`database_path` değeri, `video_Process`'in galeri yoluyla **birebir aynı** (`data/embeddings/faces.pkl`).
Bu, iki repoyu tek galeri etrafında birleştirme niyetinin zaten var olduğunu gösterir.

---

## 3. Boşluk Analizi (Gap Analysis)

| # | Boşluk | Bugün | Etki |
|---|--------|-------|------|
| G1 | **Embedding uzayı uyumsuzluğu** | Backend GhostFaceNet, video_Process ArcFace | Onboarding embedding'i canlı tanımada kullanılamaz; skorlar taşınmaz |
| G2 | **Inbound enrollment API yok** | video_Process outbound-only | Backend, foto/enroll isteğini iletebileceği bir uç bulamıyor |
| G3 | **Galeri hot-reload yok** | `faces.pkl` yalnızca başlangıçta yüklenir | Yeni enrollment, pipeline yeniden başlatılmadan canlıya yansımaz |
| G4 | **Fotoğraflar video_Process'e ulaşmıyor** | `/users/register` → SQL DB | Canlı hattın galerisi onboarding'den haberdar değil |
| G5 | **Hizalama/kalite tutarsızlığı** | Backend `align=False`, kalite kapısı yok | Aynı fotoğraf iki tarafta farklı işlenir; düşük kaliteli enroll riski |
| G6 | **Çoklu-açı gruplama** | 5 poz 5 ayrı istek; hatalar yutuluyor | Atomik olmayan, kısmî enrollment; poz başına ayrı işleme gerek |

---

## 4. Önerilen Mimari (To-Be): video_Process = Tek Doğruluk Kaynağı

### 4.1 Hedef veri akışı

```
[Tarayıcı / Onboarding Step3Biometrics]
   │  5 JPEG (front,left,right,up,down)  (mevcut UI; değişmez)
   ▼
[proactive-home-agent backend]  POST /users/register  (JWT → username)
   │  register_face() artık YEREL DeepFace yerine FORWARD eder
   ▼  POST http://{mac_ip}:5002/vision/enroll   (label=username, images=[...], auth token)
[video_Process — YENİ Enrollment Servisi (FastAPI, :5002)]
   │  1) FaceDetector.detect  → en büyük yüz + 5 landmark
   │  2) FaceQualityGate.check → kalitesiz pozları ele
   │  3) FaceIdentifier.for_enrollment().embed_face → ArcFace 512-d (L2-norm)
   │  4) faces.pkl'e atomik append/rebuild  (label → (N,512))
   ▼  5) Galeri sürümünü artır → canlı pipeline HOT-RELOAD
[Canlı Pipeline]  identify() artık bu kişiyi tanır
   ▼  POST {pi}/vision/identify · /vision/update_presence  (mevcut akış)
[Backend]  presence/identity → dashboard
```

### 4.2 Yeni bileşen: video_Process Enrollment Servisi

**Yer:** Yeni paket, ör. `serving/enroll_api.py`; `config.py`'deki ayrılmış
`server_host`/`server_port=5002` kullanılır. Web çerçevesi olarak **FastAPI + uvicorn**
önerilir (mevcut `requirements.txt`'te HTTP sunucusu yok; tek hafif bağımlılık eklenir).
`main.py` pipeline başlatılırken bu servis **ayrı bir thread**te ayağa kalkar ve
`FaceIdentifier`/`FaceDetector`/`FaceQualityGate` örneklerini paylaşır.

**Uç noktalar (öneri):**

| Metot & yol | Amaç |
|-------------|------|
| `POST /vision/enroll` | Bir kişi için 1..N yüz görüntüsüyle galeri girdisi oluştur/güncelle |
| `POST /vision/enroll/{label}/append` | Var olan kişiye yeni poz(lar) ekle |
| `DELETE /vision/enroll/{label}` | Kişiyi galeriden çıkar |
| `GET /vision/gallery` | Etiketler + poz sayıları + galeri sürümü (gözlemlenebilirlik) |
| `GET /healthz` | Servis/galeri durum kontrolü |

### 4.3 Mevcut kodun yeniden kullanımı (yeni embedding kodu YOK)

Enrollment servisinin çekirdeği, `scripts/enroll_faces.py::enroll_person` mantığının
bayt-girişli halidir:

```
for img_bytes in images:
    bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    face = _largest_face(face_det.detect(rgb))            # detection/face_detector.py
    if face is None: continue                             # → "no_face"
    x,y,w,h,score,kps = face
    roi = bgr[max(0,y):y+h, max(0,x):x+w]
    ok,_ = quality_gate.check(roi, kps, fw, fh, (x,y,w,h), ir_mode=False)  # quality/face_quality_gate.py
    if not ok: continue                                   # → "low_quality"
    crop_kps = FaceIdentifier.crop_keypoints(kps, (x,y,w,h), (fw,fh))
    emb = identifier.embed_face(roi, crop_kps)            # identification/face_identifier.py  (ArcFace, L2-norm)
    embeddings.append(emb)
gallery[label] = np.stack(embeddings).astype(np.float32)  # (N,512)
```

Böylece onboarding embedding'i, canlı `identify`'ın kullandığı **birebir aynı** detect→align→embed
yolundan geçer; uzay tutarlılığı garanti olur.

### 4.4 Atomik galeri yazımı + hot-reload

- **Atomik yazım:** `faces.pkl.tmp`'e yaz → `os.replace` ile yer değiştir (yarım dosya riski yok).
  Yazımı bir `threading.Lock` korur (enroll API thread'i ↔ pipeline thread'i).
- **Hot-reload (G3 çözümü):** Üç seçenekten biri:
  1. **Sürüm damgalı yeniden yükleme (önerilen):** Galeri yanında `faces.version` (monotonik
     sayaç veya `mtime`); `FaceIdentifier`'a `maybe_reload()` eklenir, ana döngü her K karede
     sürümü kontrol edip değiştiyse `_load_gallery`'yi tekrar çağırır. Basit, kilitlenmesiz,
     dış bağımlılık yok.
  2. Dosya izleme (`watchdog`) — ekstra bağımlılık.
  3. Sinyal/komut (örn. enroll API içte `identifier.reload_gallery()` çağırır) — servis ile
     pipeline aynı süreçte olduğundan en doğrudan yol; (1) ile birlikte kullanılabilir.
- Yeniden yükleme **kopya-değiştir** (copy-on-write) ile yapılır: yeni sözlük hazırlanır, sonra
  referans atanır; canlı `identify` hiçbir an yarım galeri görmez.

### 4.5 Backend tarafı: forward + GhostFaceNet'in emekliliği (G1, G4)

`backend/api/services/vision_service.py::register_face` davranışı değişir:

- DeepFace/GhostFaceNet ile yerel embedding **kaldırılır**; bunun yerine ham görüntü baytları
  `video_Process` `/vision/enroll`'a iletilir (kişi etiketi = `current_user.username`).
- Backend SQL'de artık ham embedding değil, yalnızca **meta** tutar (ör. `face_enrolled: bool`,
  `enroll_count`, `last_enrolled_at`) — kişisel biyometri tek yerde (`faces.pkl`, Mac) kalır.
- `recognize()` / `/vision/identify`: Backend kendi DeepFace karşılaştırmasını yapmak yerine,
  canlı kimliği zaten `video_Process`'in `identify` → `/vision/update_presence` akışından alır.
  (İsteğe bağlı: backend `/vision/identify`'ı `video_Process`'e proxy edebilir.)
- **Çevrimdışı (Mac kapalı) senaryosu:** Backend, enroll isteklerini **kuyruğa** alır
  (ör. `pending_enrollments` tablosu) ve Mac erişilebilir olunca yeniden dener; kullanıcıya
  "biyometri senkronize edilecek" geri bildirimi verilir. Onboarding UI bloke edilmez.

> **Geçiş notu:** GhostFaceNet ve ArcFace skorları taşınmadığından, mevcut DB'deki eski
> `face_embedding` değerleri **yeniden kayıt** (re-enroll) gerektirir; tek seferlik bir
> "biyometriyi yenile" akışı ile mevcut kullanıcılar ArcFace uzayına taşınır.

---

## 5. Sözleşmeler (API Contracts)

### 5.1 `POST /vision/enroll`

İstek (çoklu-açıyı tek atomik işlemde toplamak için **multipart** önerilir):

```
POST /vision/enroll
Authorization: Bearer <servis-token>
Content-Type: multipart/form-data
  label:  "ogulcan"
  mode:   "replace" | "append"          # varsayılan: append
  images: front.jpg, left.jpg, right.jpg, up.jpg, down.jpg   # 1..N dosya
```

Başarılı yanıt:

```json
{
  "status": "ok",
  "label": "ogulcan",
  "accepted": 4,
  "rejected": [
    { "file": "down.jpg", "reason": "low_quality" }
  ],
  "embeddings_total": 4,
  "gallery_version": 7
}
```

Hata/yetersiz yanıt:

```json
{ "status": "no_face", "label": "ogulcan", "accepted": 0, "rejected": [...] }
```

**Reddedilme nedenleri** (kalite kapısı ve dedektörle hizalı): `no_face`,
`multiple_faces`, `low_quality` (boyut/parlaklık/bulanıklık/frontality), `empty_crop`,
`decode_error`.

### 5.2 Davranış kuralları

- **Çoklu-açı gruplama:** Tek `label` altında N poz → galeri satırları `(N,512)` olarak
  **biriktirilir** (ortalama alınmaz; canlı eşleştirme zaten satır-bazında maksimum cosine yapar).
- **İdempotensi / çakışma:** `mode="replace"` aynı `label`'ı sıfırdan yazar; `append` ekler.
  İstemci, yeniden denemelerde aynı sonucu almak için opsiyonel `request_id` taşıyabilir.
- **En az 1 kabul:** Tüm pozlar reddedilirse galeri **değişmez**; `gallery_version` artmaz.

---

## 6. Kenar Durumlar & Güvenlik

- **Kalitesiz / yüzsüz fotoğraf:** `FaceQualityGate` ve dedektör erken eler; kısmî kabul
  desteklenir (4/5 poz kabul yeterli). Sıfır kabul → 422 benzeri yanıt, DB meta güncellenmez.
- **Embedding uzayı tutarlılığı (kritik):** Enrollment **mutlaka** canlı hattaki aynı ArcFace
  modeli + aynı 5-nokta hizalama ile üretilmeli. Model adı/sürümü (`arcface_model_name`)
  galeri meta'sına yazılmalı; model değişirse galeri **geçersiz** sayılıp yeniden üretilmeli.
- **Mac çevrimdışı:** Backend kuyruk + yeniden deneme (üstel geri çekilme). Onboarding
  tamamlanır; biyometri "beklemede" işaretlenir.
- **Eşzamanlılık:** Galeri yazımı `Lock` ile korunur; okuma (canlı `identify`) copy-on-write
  referansla yarım veri görmez.
- **Gizlilik & yetki:** Ham yüz fotoğrafı yalnızca backend→Mac iç ağında taşınır; `faces.pkl`
  Mac'te yerelde kalır, git'e girmez (`.gitignore`: `data/embeddings/*`). Enroll ucu
  **kimlik doğrulamalı** olmalı (servisler-arası token / mTLS / Tailscale ACL). Mevcut
  `pi_ip`/`mac_ip` Tailscale adresleri bu güveni doğal sağlar.
- **Etiket güvenliği:** `label` backend tarafında `current_user.username`'den türetilir;
  istemci (tarayıcı) keyfi etiket gönderemez (kimlik JWT'den gelir).

---

## 7. Aşamalı Uygulama Yol Haritası (Öneri)

> Bu rapor kapsamında kod yazılmaz; aşağıdaki fazlar ileride uygulanacak iş kalemleridir.

| Faz | Kapsam | Dokunulacak başlıca noktalar |
|-----|--------|------------------------------|
| **1** | Enroll API + `faces.pkl` yazımı | `video_Process`: yeni `serving/enroll_api.py`, `main.py` (thread başlat), `enroll_faces.py` mantığının bayt-girişli refaktoru; FastAPI bağımlılığı |
| **2** | Galeri hot-reload | `identification/face_identifier.py` (`maybe_reload`/`reload_gallery`, sürüm damgası), `main.py` döngüsünde periyodik kontrol |
| **3** | Backend forward + GhostFaceNet emekliliği | `backend/.../vision_service.py::register_face` (forward), `database/models.py` (meta alanları), DeepFace bağımlılığının kaldırılması, çevrimdışı kuyruk |
| **4** | Çoklu-açı & kalite raporlama, mevcut kullanıcı migrasyonu | UI'da poz-bazlı kabul/ret geri bildirimi (`Step3Biometrics.tsx`), "biyometriyi yenile" akışı |

---

## 8. Doğrulama Stratejisi

- **Uçtan uca:** Onboarding'den enroll → `GET /vision/gallery`'de yeni etiket + poz sayısı
  görünür → canlı kamerada aynı kişi `identify` tarafından doğru etiketlenir (önceden "Unknown"
  iken artık tanınır).
- **Eşik kalibrasyonu:** `scripts/calibrate_threshold.py` ile `face_match_cosine_threshold`
  (varsayılan 0.35) onboarding-kaynaklı pozlarda doğrulanır; yanlış-kabul/yanlış-ret dengesi ölçülür.
- **Hot-reload testi:** Pipeline çalışırken enroll → yeniden başlatmadan tanıma; `gallery_version`
  artışı gözlenir.
- **Regresyon:** Mevcut test deseni (`tests/test_phase_a_contracts.py`, `tests/test_vision_state.py`,
  `tests/conftest.py`) izlenerek enroll servisi için kontrat testleri eklenir (sahte ArcFace ile
  hızlı, modelsiz birim testleri).
- **Atomiklik:** Yazım sırasında süreç öldürülürse `faces.pkl` bozulmamalı (`.tmp`+`os.replace`).

---

## 9. Ek: Dosya & Sembol Referans Tablosu

### proactive-home-agent

| Yol | Rol |
|-----|-----|
| `frontend/src/components/onboarding/Step3Biometrics.tsx` | 5 poz yüz yakalama + `/users/register`'a gönderim |
| `backend/api/routers/user_router.py` | `POST /users/register`, `/users/add-guest` |
| `backend/api/services/vision_service.py` | `register_face`, `recognize` (DeepFace/GhostFaceNet) |
| `backend/api/services/vision_core/detectors/face.py` | `FaceDetector` (yolov8n-face) |
| `backend/api/services/vision_core/recognizers/face_aligner.py` | FAN hizalama (ArcFace referans noktaları) |
| `backend/api/services/vision_core/config/settings.py` | `FaceRecognitionConfig` (model, threshold, `database_path`) |
| `backend/api/routers/vision_router.py` | `/vision/identify`, `/vision/update_presence`, `/vision/identity_event` |
| `backend/database/models.py` | `User.face_embedding` (JSON) |
| `backend/api/services/vector_db.py` | ChromaDB — yalnız RAG/memory, yüz için **kullanılmaz** |

### video_Process

| Yol | Rol |
|-----|-----|
| `identification/face_identifier.py` | ArcFace embed/align/match, galeri yükleme (`embed_face`, `for_enrollment`, `_load_gallery`, `_best_gallery_match`) |
| `scripts/enroll_faces.py` | Offline galeri üretimi (`enroll_person`) — enroll API'nin temeli |
| `scripts/capture_enrollment.py` | Webcam yakalama + kalite kapısı |
| `detection/face_detector.py` | `FaceDetector.detect` → `(x,y,w,h,score,5 landmark)` |
| `quality/face_quality_gate.py` | `FaceQualityGate.check` (boyut/parlaklık/bulanıklık/frontality) |
| `config.py` | `gallery_path`, `face_match_cosine_threshold`, `arcface_model_name`, ayrılmış `server_port=5002` |
| `events/dispatcher.py` | Outbound olaylar (`identify`/`presence`/`fall`/`gesture`) |
| `streaming/vision_ws_server.py` | Tek yönlü `VisionState` yayını (`:5003`) — enroll için uygun değil |
| `main.py` | Pipeline kurulumu; yeni enroll servisinin başlatılacağı yer |
| `data/embeddings/faces.pkl` | Galeri (`{label: (N,512)}`) — birleşik doğruluk kaynağı |

---

*Bu doküman bir analiz/tasarım raporudur; herhangi bir çalışma zamanı kodu değiştirmez.*
