Yönetici Özeti
Üç kaynak rapor aynı sistemi incelemektedir: M3 Pro üzerinde çalışan, yüz takibi/tanıma, jest algılama ve transformer tabanlı düşme tespiti gerçekleştiren 955 satırlık monolitik bir vision pipeline'ı — sonuçlar HTTP üzerinden Raspberry Pi 5 agent'ına iletilmektedir.
Bu hibrit rapor, üç incelemenin her birindeki en güçlü yaklaşımı seçer ve aralarındaki çatışmaları (threading vs. multiprocessing, mevcut modeli koru vs. değiştir) tutarlı bir plan halinde çözer. Ancak kritik bir zemin değişikliği vardır: src/ altındaki modüler yapı artık mevcut değildir. Bu, Rapor 2'nin temel önerisini ("var olan modüle yakınsa") geçersiz kılar ve stratejiyi iki aşamalı bir plana dönüştürür:
Aşama A — Modülerleştir. mac_camera.py'ın çalışma zamanında kanıtlanmış davranışlarını sıfır işlevsel değişiklikle temiz modüllere ayır. Hiçbir model değişmez, hiçbir yeni bağımlılık eklenmez. Bu aşamanın çıktısı: aynı şeyi yapan, ama test edilebilir, genişletilebilir ve bakımı yapılabilir bir kod tabanı.
Aşama B — Geliştir. Modüler yapının yarattığı temiz ekleme noktalarını kullanarak iyileştirmeleri sırayla uygula: RTSP/IR ingestion, ByteTrack, CoreML hızlandırma, MQTT event bus, spine-angle güvenlik kapısı.
Bu sıralama kasıtlıdır. Monolite uygulanan her iyileştirme, yapısal karmaşıklığı katlanarak artırır ve geri dönüşü zorlaştırır. Modüler yapı ise her iyileştirmeyi bağımsız, test edilebilir ve geri alınabilir bir birim haline getirir.

Alan Bazında Karşılaştırmalı Değerlendirme
Alan 1 — Pipeline Mimarisi ve Eş Zamanlılık Modeli
Seçilen Yaklaşım: Monoliti producer-consumer kalıbına göre parçala. Adanmış bir capture thread'i, en-son-kareyi-tut mantığıyla bounded queue'ya (maxsize=2) yazar. Tek bir BGR→RGB dönüşümü yapılır ve tüm tüketiciler bu immutable buffer'ı paylaşır. Yalnızca düşme tespiti — CPU-bound ve GIL darboğazı olan tek aşama — ayrı bir OS process'inde çalıştırılır; diğer tüm aşamalar threading ile kalır.
Kaynak: Birleşik — Rapor 2 (producer-consumer + tek RGB dönüşümü) ile Rapor 1 (CPU-bound worker için process izolasyonu).
Gerekçe: Raporlar arasındaki merkezi çatışma budur. Rapor 2 ve 3 saf threading önerir; Rapor 1 tam multi-process actor modeli önerir. Threading tek başına teknik olarak yetersizdir: MediaPipe'ın C++ katmanı GIL'i serbest bırakırken, çevreleyen NumPy, cv2.imencode ve dict işlemleri bırakmaz — bu da ana döngüde head-of-line blocking yaratır. Ancak her aşama için ayrı process (Rapor 1) gereksiz mühendislik karmaşıklığı taşır. Hibrit yaklaşım her aşama için en ucuz doğru çözümü seçer: I/O-bound aşamalar için thread, GIL'in gerçekten ısırdığı düşme tespiti için process.

Alan 2 — Kamera Yakalama ve Gece Görüşü (IR) Yönetimi
Seçilen Yaklaşım: Yerel cv2.VideoCapture(0) webcam'i, Tapo C225 RTSP feed'i ile değiştir. Çift akış stratejisi: 720p alt akış (/stream2) tüm ML inference için, 2K ana akış (/stream1) yalnızca düşme anında tam çözünürlük ekran görüntüsü için. Otomatik yeniden bağlanma mantığı ve CAP_PROP_BUFFERSIZE=1 ile latansı minimize et. Kritik olarak: IR/gece görüşü modunu kanal ortalamaları üzerinden standart sapma ile tespit et (R≈G≈B ⇒ gri tonlama ⇒ IR), parlaklık kapılarını IR'ye özgü eşiklere çevir ve ir_mode flag'ini downstream'e taşı.
Kaynak: Rapor 1, buffer optimizasyonu Rapor 3 tarafından doğrulanmış.
Gerekçe: En net tek-rapor kazanımı. Rapor 2 hâlâ yerel webcam'i hedefler; Rapor 3 RTSP'yi ele alır ama gece görüşünü yok sayar. Yalnızca Rapor 1, C225'in 940nm IR modunun mevcut MIN_BRIGHTNESS/MAX_BRIGHTNESS kalite kapısını sessizce bozacağını ve RGB-eğitimli yüz tanımanın doğruluğunu düşüreceğini tespit eder. Düşmelerin en yüksek riskli ve en az gözlemlenen olduğu gece saatlerinde çalışması gereken bir güvenlik sistemi için IR yönetimi opsiyonel değil, temel doğruluktur. Yüksek etki, orta fizibilite, donanım fiziğine dayalı benzersiz kanıt.
Not: Bu iyileştirme Aşama B'de gelir — Aşama A'da mevcut webcam yakalama mantığı aynen korunur, yalnızca bağımsız bir capture/ modülüne çıkarılır.

Alan 3 — Yüz Tespiti, Takibi ve Tanıma
Seçilen Yaklaşım: Eski KCF tracker'ını ByteTrack ile değiştir (ultralytics bağımlılığı zaten requirements.txt'te mevcut). Yüz tanımayı cihaz üzerinde GhostFaceNet/ArcFace ile ONNX Runtime CoreML EP üzerinden çalıştır. Gece operasyonu için IR-fine-tuned bir embedding modeline (ArcFace-IR) geçiş yolunu planla.
Kaynak: Birleşik — Rapor 2 (ByteTrack gerekçesi, ~200 satır silme) ve Rapor 3 (bağımsız ByteTrack/CoreML doğrulaması), Rapor 1 IR-tanıma uzantısı.
Gerekçe: Rapor 2 ve 3 bağımsız olarak ByteTrack'te birleşir — güçlü konsensüs sinyali. ByteTrack, detection-anchored çalışır ve track ID'lerini dahili olarak yönetir; bu, Rapor 1'in dikkatle mühendislik yapılmış TrackerStore yarış koşulu düzeltmesini gereksiz kılar — yarışa neden olan manuel dict + lock yapısı tamamen ortadan kalkar. Bir hata sınıfını ortadan kaldırmak, hatayı düzeltmekten kesinlikle daha üstündür. Cihaz üzerinde tanıma, üç raporun da vurguladığı 200–800ms ağ round-trip'ini ve 8 saniyelik HTTP timeout riskini ortadan kaldırır.
Not: Aşama A'da KCF tracker mantığı aynen kalır, yalnızca tracking/ modülüne çıkarılır. ByteTrack geçişi Aşama B'de yapılır.

Alan 4 — Düşme Tespiti
Seçilen Yaklaşım: Mevcut özel Transformer sınıflandırıcısını koru (belgelenmiş %94.9 F1), ancak (a) CoreML'e (.mlpackage) dönüştürerek Neural Engine üzerinde çalıştır, (b) Alan 1'deki izole process içinde çalıştır, (c) ham y-koordinatı hız kapısını, rotasyona bağımsız omurga açısı monitörü ile değiştir — omuz orta noktası→kalça orta noktası vektörünün açısal hızını takip eder. Opsiyonel olarak MediaPipe Pose yerine YOLOv8-Pose (CoreML) ile anahtar nokta çıkarımı yap. ST-GCN'i yalnızca uzun vadeli araştırma kanalı olarak değerlendir.
Kaynak: Birleşik — Rapor 2 (kanıta dayalı model koruma) + Rapor 1 (SpineAngleMonitor geometri düzeltmesi, CoreML dönüşümü, process izolasyonu), Rapor 3 YOLOv8-Pose seçeneği.
Gerekçe: İkinci gerçek çatışma. Rapor 1, Transformer'ı SOTA ST-GCN/AGCN ile değiştirmeyi savunur; Rapor 2, belgelenmiş %94.9 F1 puanıyla korunmasını önerir. Kanıt Rapor 2'yu destekler: doğrulanmış, güvenlik kritik bir sınıflandırıcıyı kıyaslanmamış bir alternatif için terk etmek yanlış risk duruşudur. Ancak Rapor 1, tüm raporlardaki en yüksek değerli tekil düzeltmeyi sunar — kameraya açılı monte edildiğinde yürüyen kişinin pozitif y-hızı üretip yanlış düşme tetiklemesine neden olan kamera açısı bağımlılık hatası. Omurga açısı (açısal hız) kapısı yapısal olarak rotasyona bağımsızdır ve doğrudan yanlış pozitiflere — bir düşme detektörünün birincil başarısızlık modu — saldırır.

Alan 5 — Apple Silicon Donanım Hızlandırma
Seçilen Yaklaşım: İki dalgada uygula. Hemen (saatler): MediaPipe GPU/Metal delegate'ini etkinleştir — tek satır BaseOptions(delegate=GPU) değişikliği, yeniden eğitim gerektirmez, 3–5× hızlanma (MediaPipe ≥ 0.10.9 gerekli). Daha derin (haftalar): Düşme Transformer'ını coremltools ile CoreML mlprogram formatına dönüştür (FLOAT16, ComputeUnit.ALL), ONNX modellerini CoreML EP üzerinden yönlendir.
Kaynak: Birleşik — Rapor 2 (en düşük eforlu drop-in delegate) ve Rapor 1 (detaylı coremltools dönüşüm kodu).
Gerekçe: Üç rapor da M3 Pro'nun 18 çekirdekli GPU'su ve 16 çekirdekli Neural Engine'inin boşta durduğu konusunda hemfikir — oybirliğiyle, kanıta dayalı bulgu. Rapor 2'nin GPU-delegate tek satırlık değişikliği, analiz genelindeki en yüksek ROI/saat eylemidir ve ilk gönderilmelidir. Rapor 1'in tam CoreML dönüşümü daha yüksek tavan sunar (düşme modelinde ≈3–5ms vs. 25–40ms) ancak çok haftalık bir çalışmadır, bu nedenle ikinci sırada gelir.

Alan 6 — Agentic Entegrasyon: Taşıma ve Olay Şeması
Seçilen Yaklaşım: Fire-and-forget HTTP POST'ları MQTT ile değiştir (yerel Mosquitto broker). Olay türüne göre QoS: düşmeler QoS 2 (exactly-once, güvenlik kritik), jestler/tanımlamalar QoS 1, yüksek frekanslı varlık güncellemeleri QoS 0. Son bilinen durum için retained messages, beklenmeyen bağlantı kopmasında otomatik camera_offline yayını için Last Will & Testament kullan. Tek bir versiyonlu zarf şeması (v, ts, src, loc, evt, d) taşı — evt alanına göre yönlendir, kompakt anahtarlar wire verimliliği ve LLM-context ingestion için — Rapor 1'in tanısal payload alanlarıyla (spine_angle_deg, velocity_at_fall, ir_mode, frame_id) zenginleştir.
Kaynak: Rapor 2 (QoS haritalama, LWT, retained state, birleşik kompakt zarf, taşıma karşılaştırması), Rapor 1 (zengin payload alanları, şema versiyonlama).
Gerekçe: Üç rapor da bağımsız olarak HTTP yerine MQTT'yi önerir — güçlü konsensüs. Rapor 2, QoS seviyelerini olay kritikliğine eşleyen (düşme sessizce kaybolmamalı; varlık güncellemesi kaybolabilir), LWT ile otomatik çevrimdışı tespiti belirleyen ve kapsamlı bir taşıma karşılaştırma tablosu sunan tek rapor olarak taşıma katmanını kazanır. Şemada, Rapor 2'nin kısa anahtarlı, versiyon etiketli, tek ayrıştırıcılı zarfı wire üzerinde ~%40 daha küçüktür ve agent/LLM ingestion için optimize edilmiştir. Rapor 1'in katkısı payload'un içeriğidir: zengin düşme diagnostikleri olayları agent için daha eyleme dönüştürülebilir yapar.
Not: Aşama A'da mevcut HTTP POST fonksiyonları taşıma-agnostik bir EventDispatcher arayüzünün arkasına alınır; MQTT geçişi Aşama B'de bu arayüzün yeni implementasyonudur.

Alan 7 — Kod Kalitesi ve Yeniden Yapılandırma
Seçilen Yaklaşım: Monolitin 28 modül düzeyindeki global değişkenini tutarlı sınıflara kapsülle: PipelineConfig (tüm konfigürasyon), FrameProducer (yakalama), EventDispatcher (transport-agnostik olay yayını), ve üst düzey VisionPipeline (orkestratör). SSL-bypass hack'ini kaldır. Yapılandırılmış loglama (loguru) ekle. EventDispatcher'ı transport-agnostik bir seam olarak tasarla — HTTP→MQTT geçişi drop-in swap olsun.
Kaynak: Rapor 2, her alanda en kapsamlı kod kalitesi analizi.
Gerekçe: Rapor 2 bu alanı açık ara domine eder — 28 global değişkeni (test, hot-reload ve çoklu kamera ölçeklendirmesini engelleyen) listeleyen, SSL-bypass güvenlik hijyen sorununu işaretleyen ve olay üretimi ile taşıma arasında temiz bir bağımlılık dikişi tasarlayan tek rapor. Rapor 2'nin P0/P1/P2 öncelik tablosu, üç dokümandaki en eyleme dönüştürülebilir planlama artefaktıdır.

Stratejik Sinerjiler
Seçilen yaklaşımlar yalnızca bir arada var olmak yerine birbirini güçlendirir:
IR flag'i tüm pipeline'ı diker. Yakalamada tespit edilir (Alan 2), olay zarfında taşınır (Alan 6), tanıma modelini tetikler (Alan 3). Tek bir kamera-fiziği farkındalığı, şema onu taşımak üzere tasarlandığı için temiz şekilde yayılır.
Hibrit eş zamanlılık ve düşme modeli birbirine anahtar gibi oturur. CPU-bound Transformer (Alan 4), process izolasyonunu haklı kılan tam da o aşamadır (Alan 1); spine-angle kapısı aynı izole worker içinde çalışır; CoreML hızlandırma (Alan 5) worker'ı tam kare hızında çalıştıracak kadar hafifleştirir.
MQTT QoS ve doğrulanmış düşme modeli güvenlik döngüsünü kapatır. Spine-angle doğrulamalı, CoreML-hızlandırılmış bir düşme olayı (Alan 4) exactly-once garantileriyle teslim edilir (Alan 6) — yüksek güvenilirlikli detektör ve yüksek güvenilirlikli taşıma kritiklikte eşleşir.
ByteTrack bir kuvvet çarpanıdır. Seçilmesi (Alan 3) ~200 satır siler, yarış koşulunu özel bir düzeltme olmadan ortadan kaldırır ve olay şemasının (Alan 6) jestleri ve varlığı belirli kişilere atfetmek için kullandığı stabil track ID'leri üretir.
Transport-agnostik seam, modülerleştirmeyi korur. EventDispatcher arayüzü (Alan 7) Aşama A'da HTTP ile çalışır, Aşama B'de MQTT'ye dönüşür — pipeline kodunda sıfır değişiklik.

mac_camera.py Modülerleştirme Planı
Hedef Dizin Yapısı
video_process/
├── main.py                          # Giriş noktası, VisionPipeline'ı başlatır
├── config.py                        # PipelineConfig dataclass
│
├── capture/
│   ├── __init__.py
│   ├── frame_producer.py            # FrameProducer (webcam thread)
│   └── frame_packet.py              # FramePacket dataclass
│
├── detection/
│   ├── __init__.py
│   ├── face_detector.py             # BlazeFace sarmalayıcı + sonuç normalizasyonu
│   ├── gesture_recognizer.py        # MediaPipe GestureRecognizer sarmalayıcı
│   └── fall_detector.py             # Pose çıkarımı + TFLite Transformer + hız kapısı
│
├── tracking/
│   ├── __init__.py
│   └── tracker_manager.py           # KCF tracker oluşturma/güncelleme/silme + IoU eşleme
│
├── identification/
│   ├── __init__.py
│   └── face_identifier.py           # Pi HTTP tanıma (identify_face_from_pi)
│
├── events/
│   ├── __init__.py
│   ├── dispatcher.py                # EventDispatcher arayüzü + HTTP implementasyonu
│   └── schema.py                    # Olay zarf oluşturma yardımcıları
│
├── streaming/
│   ├── __init__.py
│   └── mjpeg_server.py              # Flask app + /video_feed endpoint
│
├── quality/
│   ├── __init__.py
│   └── face_quality_gate.py         # ROI boyut, Laplacian, parlaklık, göz mesafesi kontrolleri
│
└── utils/
    ├── __init__.py
    └── constants.py                 # Sabitler (URL'ler, model yolları, eşikler)
Kaynak Eşleme: Monolitten Modüllere
Aşağıdaki tablo, mac_camera.py'ın her bölümünün hangi modüle taşınacağını gösterir. Davranış değişmez — yalnızca fiziksel konum değişir.
mac_camera.py Satırları (yaklaşık)İçerikHedef Modül1–28Import'lar, SSL bypassHer modül kendi import'ını alır; SSL bypass kaldırılır29–18228 global değişken, sabitler, eşiklerconfig.py → PipelineConfig dataclass183–280quality_check(), check_face_quality()quality/face_quality_gate.py281–380extract_and_normalize_pose(), run_fall_detection(), hız kapısı mantığıdetection/fall_detector.py381–430gesture_callback(), jest işlemedetection/gesture_recognizer.py431–480identify_face_from_pi(), HTTP gönderimiidentification/face_identifier.py481–545send_presence_json(), send_fall_alert(), send_gesture_event(), network executorevents/dispatcher.py546–636async_image_buffer, MediaPipe model başlatmadetection/ altındaki ilgili modül __init__ veya factory fonksiyonu637–923camera_processing_loop() — tüm CV mantığıParçalanır: yakalama→capture/, tracker güncelleme→tracking/, tespit çağrıları→detection/, JPEG encode→streaming/, orkestrasyon→main.py → VisionPipeline924–955Flask app, /video_feed, if __name__streaming/mjpeg_server.py + main.py
Modülerleştirme Adımları (Aşama A)
Her adımda tam fonksiyonel eşdeğerlik hedeflenir. Hiçbir adım yeni özellik veya model değişikliği içermez.
Adım 1 — Config çıkarımı (≈2 saat)
28 global değişkeni tek bir PipelineConfig dataclass'ına taşı. Bu, geri kalan tüm modüllerin bağımlılık enjeksiyonu ile yapılandırma almasını sağlar. Sabitler (PI_BASE_URL, model yolları, eşikler) burada yaşar.
python# config.py
from dataclasses import dataclass

@dataclass
class PipelineConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    face_detection_interval: float = 0.15
    fall_target_fps: float = 15.0
    fall_confidence_threshold: float = 0.90
    fall_alert_cooldown: float = 10.0
    # ... monolitteki tüm sabitler buraya taşınır
    agent_base_url: str = "http://100.105.136.5:8000"
Test: Config oluştur, değerlerin doğruluğunu assert et. Pipeline hâlâ global'leri okur — bu adımda henüz bağlanmaz.

Adım 2 — Quality gate çıkarımı (≈1 saat)
quality_check() ve check_face_quality() fonksiyonlarını quality/face_quality_gate.py'a taşı. Saf fonksiyonlardır — herhangi bir global state'e bağımlılıkları PipelineConfig'ten gelen eşiklerle değiştirilir. Bu modül diğerlerinden bağımsız test edilebilir.
python# quality/face_quality_gate.py
class FaceQualityGate:
    def __init__(self, config: PipelineConfig):
        self.min_roi_size = config.min_face_roi_size
        self.min_laplacian = config.min_laplacian_variance
        # ...

    def check(self, face_roi: np.ndarray, keypoints: list) -> tuple[bool, float]:
        """Monolitteki quality_check() + check_face_quality() birleşimi."""
        # ... mevcut mantık aynen, global yerine self.xxx
Test: Örnek yüz ROI'leri ile birim testi — geçen ve kalan senaryolar.

Adım 3 — Event dispatcher çıkarımı (≈2 saat)
Dört ayrı HTTP POST fonksiyonunu (send_presence_json, send_fall_alert, send_gesture_event, identify_face_from_pi'ın ağ kısmı) tek bir EventDispatcher sınıfının arkasına al. Bu adım transport-agnostik seam'i oluşturur — Aşama B'deki MQTT geçişinin ekleme noktası.
python# events/dispatcher.py
class EventDispatcher:
    """Transport-agnostik olay yayını. Şu an HTTP, gelecekte MQTT."""

    def __init__(self, config: PipelineConfig):
        self._executor = ThreadPoolExecutor(max_workers=config.network_workers)
        self._session = requests.Session()
        self._base_url = config.agent_base_url

    def emit(self, event_type: str, payload: dict) -> None:
        """Non-blocking olay gönderimi."""
        self._executor.submit(self._send, event_type, payload)

    def _send(self, event_type: str, payload: dict) -> None:
        url_map = { ... }  # mevcut endpoint eşlemeleri
        # ... mevcut POST mantığı
Test: Mock HTTP server'a karşı, dört olay türünün doğru endpoint'e ulaştığını doğrula.

Adım 4 — Detection sarmalayıcıları çıkarımı (≈3 saat)
Üç bağımsız modül oluştur — her biri tek bir ML görevini kapsüller:
detection/face_detector.py — BlazeFace model yükleme, detect() çağrısı, sonuç normalizasyonu.
detection/gesture_recognizer.py — MediaPipe GestureRecognizer başlatma, recognize_async() çağrısı, callback yönetimi, async_image_buffer (GC önleme hack'i şimdilik aynen korunur).
detection/fall_detector.py — MediaPipe Pose başlatma, extract_and_normalize_pose(), TFLite interpreter yükleme, run_fall_detection(), hız kapısı mantığı, feature buffer (deque).
Her sarmalayıcı kendi model instance'ını tutar, kendi state'ini yönetir, ve dışarıya tek bir temiz arayüz sunar.
python# detection/fall_detector.py
class FallDetector:
    def __init__(self, config: PipelineConfig):
        self._interpreter = TFLiteInterpreter(config.fall_model_path)
        self._pose = mp_pose.Pose(model_complexity=1, ...)
        self._feature_buffer = deque(maxlen=config.fall_input_timesteps)
        # ... monolitteki tüm fall state'i buraya

    def process_frame(self, rgb_frame: np.ndarray) -> Optional[FallEvent]:
        """Tek kare işle, düşme varsa FallEvent döndür, yoksa None."""
        # ... mevcut run_fall_detection() mantığı
Test: Kayıtlı video kareleri ile her detektörü bağımsız çalıştır.

Adım 5 — Tracker yönetimi çıkarımı (≈2 saat)
KCF tracker oluşturma/güncelleme/silme, IoU eşleme, TTL temizliği ve active_trackers dict'ini tracking/tracker_manager.py'a taşı. trackers_lock bu sınıfın dahili bir detayı olur.
python# tracking/tracker_manager.py
class TrackerManager:
    def __init__(self, config: PipelineConfig):
        self._lock = threading.Lock()
        self._trackers: dict[int, TrackerEntry] = {}
        self._next_id = 0
        # ...

    def update_all(self, frame: np.ndarray) -> list[TrackerResult]:
        """Tüm tracker'ları güncelle, başarısız olanları temizle."""

    def match_detections(self, detections: list, frame: np.ndarray) -> None:
        """IoU eşleme ile yeni tespitleri mevcut tracker'lara ata veya yeni oluştur."""

    def evict_expired(self) -> list[int]:
        """TTL'i dolmuş tracker'ları sil."""
Not: Yarış koşulu bu adımda henüz düzeltilmez — mevcut dict + lock kalıbı aynen taşınır. Aşama B'de ByteTrack'e geçişle bu sınıf tamamen değiştirilecek ve yarış koşulu sınıf olarak yok olacaktır.
Test: Sahte bbox'larla tracker oluştur/güncelle/sil döngüsü, TTL eviction doğrula.

Adım 6 — Face identifier çıkarımı (≈1 saat)
identify_face_from_pi() fonksiyonunu identification/face_identifier.py'a taşı. JPEG encode, HTTP multipart gönderimi, cevap ayrıştırma ve retry mantığı bu modülde yaşar.
python# identification/face_identifier.py
class FaceIdentifier:
    def __init__(self, config: PipelineConfig):
        self._url = f"{config.agent_base_url}/vision/identify"
        self._timeout = 8.0

    def identify(self, face_roi: np.ndarray) -> Optional[IdentifyResult]:
        """Yüz ROI'sini Pi'ye gönder, sonuç döndür."""
Test: Mock Pi endpoint'i ile başarılı/timeout/hata senaryoları.

Adım 7 — Capture ve streaming çıkarımı (≈2 saat)
capture/frame_producer.py: cv2.VideoCapture sarmalama, adanmış thread, bounded queue (maxsize=2, drop-oldest), read_latest().
streaming/mjpeg_server.py: Flask app, /video_feed endpoint, generate() jeneratörü, latest_jpeg_frame global'ini sınıf attribute'una dönüştürme.
python# capture/frame_producer.py
class FrameProducer(threading.Thread):
    def __init__(self, config: PipelineConfig):
        super().__init__(daemon=True)
        self._cap = cv2.VideoCapture(config.camera_index)
        self._queue = queue.Queue(maxsize=2)
        # ...

    def read_latest(self) -> Optional[np.ndarray]:
        """En son kareyi non-blocking oku. Yoksa None."""
Test: FrameProducer'ı başlat, birkaç kare oku, sıranın taşmadığını doğrula.

Adım 8 — Orkestratör ve giriş noktası (≈3 saat)
main.py: VisionPipeline sınıfı oluştur — tüm modülleri örneklendirir, ana işleme döngüsünü çalıştırır, graceful shutdown yönetir.
Bu, monolitteki 300 satırlık camera_processing_loop()'un yerini alır, ancak artık her CV çağrısı kendi modülündeki tek satırlık bir method call'dır.
python# main.py
class VisionPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.capture = FrameProducer(config)
        self.dispatcher = EventDispatcher(config)
        self.face_detector = FaceDetector(config)
        self.gesture = GestureRecognizer(config, self.dispatcher)
        self.fall = FallDetector(config)
        self.tracker = TrackerManager(config)
        self.identifier = FaceIdentifier(config)
        self.quality = FaceQualityGate(config)
        self.stream = MJPEGServer(config)

    def run(self):
        self.capture.start()
        self.stream.start()
        while self._running:
            frame = self.capture.read_latest()
            if frame is None:
                time.sleep(0.005)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Her modül kendi işini yapar
            self.gesture.process(rgb, timestamp)
            self.fall.process_frame(rgb)
            tracker_results = self.tracker.update_all(frame)
            # ... face detection, identification tetikleme
            self.stream.update_frame(frame)
Test: Tüm modüllerle entegrasyon testi — pipeline başlatılır, birkaç saniye çalışır, graceful shutdown yapılır. Çıktıların (HTTP POST'lar, MJPEG stream) monolitle aynı olduğu doğrulanır.

Modülerleştirme Sonrası Doğrulama
Aşama A tamamlandığında, mac_camera.py silinmeden önce şu doğrulamalar yapılır:

Fonksiyonel eşdeğerlik: Aynı video girdisiyle hem monoliti hem modüler pipeline'ı çalıştır, HTTP POST payload'larını karşılaştır.
Performans regresyon testi: Her iki versiyonun FPS'ini ölç — modüler versiyon monolitten %5'ten fazla yavaş olmamalıdır.
Graceful shutdown: Ctrl+C ile temiz kapanış, thread sızıntısı yok, kamera serbest bırakılıyor.
Edge case'ler: Kamera bağlantısız başlatma, Pi çevrimdışı senaryosu, boş kare dizisi.


Uygulama Takvimi — Revize
DalgaİçerikSüreÖn KoşulDalga 0Modülerleştirme (Aşama A, Adım 1–8)4–5 günYokDalga 1GPU delegate (1 satır), SSL bypass kaldırma, loguru ekleme1 günDalga 0Dalga 2RTSP Tapo C225 + IR tespiti (capture/ modül değişimi)3–4 günDalga 0Dalga 3ByteTrack geçişi (tracking/ modül değişimi) + cihaz üzerinde tanıma (identification/ modül değişimi)3–4 günDalga 0Dalga 4MQTT geçişi (events/dispatcher.py yeni implementasyon)2–3 günDalga 0Dalga 5Düşme tespitini ayrı process'e taşı + SpineAngleMonitor3–4 günDalga 0 + 2Dalga 6CoreML dönüşüm (fall model → .mlpackage)1–2 haftaDalga 5
Kritik gözlem: Dalga 2–5 birbirinden bağımsızdır ve paralel çalışabilir — hepsi yalnızca Dalga 0'ın (modülerleştirme) yarattığı temiz modül sınırlarına bağımlıdır. Bu, modülerleştirmenin neden önce gelmesi gerektiğinin yapısal kanıtıdır: monolite uygulanan her iyileştirme seridir ve birbirine dolanır; modüler yapıya uygulanan her iyileştirme paraleldir ve izole kalır.

Sonuç
Bu planın belirleyici önerisi: monoliti iyileştirme, önce parçala. mac_camera.py'ın 955 satırı, çalışma zamanında kanıtlanmış, değerli davranışlar içerir — ancak bu davranışlar global state, seri çalışma ve sıkı bağlantı tarafından rehin alınmıştır. Modülerleştirme bu rehineyi çözer ve her sonraki iyileştirmeyi bağımsız, test edilebilir ve geri alınabilir bir birime dönüştürür.
mac_camera.py'ı silmek son adımdır, ilk değil.
