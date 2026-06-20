# Handoff — Gesture Pipeline Güvenilirlik Düzeltmeleri (3 sorun)

> **Bağlam:** Canlı testte kullanıcı defalarca jest yaptığı halde Mac tarafında **tek bir
> `Sustained gesture` / `Gaze Lock` log'u bile çıkmadı** → Pi'ye hiçbir gesture gitmedi. Kök neden
> analizi 3 yapısal sorun ortaya çıkardı. Bunlar **kimlik (person-track) fix'inden bağımsızdır**;
> o fix MediaPipe'a giden görüntüyü ve `_on_result`'ın tespit mantığını değiştirmiyor.
>
> **Rol:** Codex implement eder. Kapsam ve doğrulama bağlayıcı. Surgical kal.

---

## Sorunların özeti (analiz)

1. **(Birincil) `_is_processing` tek-uçuş kilidi kalıcı takılıyor.** `process()` yalnızca
   `if not self._is_processing:` iken `recognize_async` çağırıyor; bayrak **sadece** `_on_result`
   içinde sıfırlanıyor, recovery/timeout yok. MediaPipe `LIVE_STREAM`, grafik meşgulken kareyi
   FlowLimiterCalculator ile **düşürür ve düşen kare için callback çağırmaz**. Ağır yükte (kişi
   oturur → fall detector her karede Pose çalıştırıyor) uçuştaki tek kare bir kez düşerse →
   `_on_result` gelmez → `_is_processing` sonsuza dek `True` → gesture tanıma tamamen ölür.
   "Tam sessizlik + hiç Gaze Lock log'u yok" bunu birebir açıklıyor.

2. **(İkincil) `is_frontal` donuyor.** `main.py` içinde `gesture_rec.is_frontal` **yalnızca
   identified bir face track varken** güncelleniyor. Face track expire olup bir daha gelmeyince
   gaze-lock durumu son değerinde donuyor → `Thumb_Up`/`Victory` belirsiz şekilde bastırılıyor.

3. **(Üçüncül) El ROI'si fall detector'ın merkez-4:3 pose'una asalak.** Gesture'ın el ROI'si
   `pose_tracks ← fall_result.pose`'dan geliyor; bu pose `fall_region_px` (karenin ortasındaki 4:3
   crop) üzerinden çıkıyor, rate-limit'li ve yalnız belirli fall state'lerinde üretiliyor; wrist'ler
   `None` olabiliyor. Kişi merkez bölge dışında/kambur otururken el ROI bozuluyor → MediaPipe el
   görmüyor.

---

## Kapsam (sadece bunlar)

1. `detection/gesture_recognizer.py` — Fix 1 (kilidi kaldır) + Fix 3 (ROI'yi person-track'e bağla).
2. `main.py` — Fix 2 (`is_frontal` her kare reset) + Fix 3 (gesture'a person-track region'larını besle).
3. `tests/test_phase_a_contracts.py` — yeni davranışlar için testler.

**Kapsam dışı:** Pi tarafı, fall detector mantığı, enrollment, kimlik çözümü (`user_for_person_track`
aynen kalır), face tanıma eşiği. Bunlara dokunma.

---

## Fix 1 — `_is_processing` kilidini kaldır (kök neden)

**Karar:** Manuel tek-uçuş kilidini tamamen kaldır. MediaPipe `LIVE_STREAM`'in **kendi
FlowLimiterCalculator'ı** zaten backpressure yapıyor (meşgulken kareyi düşürür, max 1 in-flight);
`recognize_async` non-blocking. Her kare submit etmek **dokümante edilen pattern**. Bu, gerçek
inference sıklığını artırmaz (zaten tek-uçuştu) ama deadlock riskini sıfırlar.

### `detection/gesture_recognizer.py`

**(a) `__init__`** — `self._is_processing` alanını **kaldır**. `self._pending_track_id` kalsın.

**(b) `process()`** — kilidi kaldır, her kare submit et:
```python
    def process(
        self,
        rgb_frame: np.ndarray,
        current_time: float,
        regions: Sequence[WristPose] | None = None,
    ) -> None:
        current_ms = int(current_time * 1000)
        if current_ms <= self._last_timestamp_ms:
            current_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = current_ms

        try:
            roi, track_id = self._hand_roi(rgb_frame, regions)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi.copy())
            self._async_image_buffer.append(mp_image)
            if len(self._async_image_buffer) > 30:
                self._async_image_buffer.pop(0)
            self._pending_track_id = track_id
            self._recognizer.recognize_async(mp_image, current_ms)
        except Exception:
            pass
```
> `param` adını `poses` → `regions` yaptım (Fix 3 ile tutarlı). `_async_image_buffer` GC koruması
> aynen kalsın.

**(c) `_on_result()`** — sondaki `self._is_processing = False` satırını **kaldır**. Geri kalan
(Gaze Lock, sustained, dispatch + `_get_user_for_track`) aynen kalır.

> **Alternatif (tercih edilmez):** Kilidi korumak istersen watchdog ekle — `_last_submit_time` tut,
> `process()` başında `if self._is_processing and now - self._last_submit_time > 1.0: self._is_processing = False`.
> Kaldırmak daha temiz; alternatifi yalnızca CPU'yu sıkı sınırlamak istersen kullan.

---

## Fix 2 — `is_frontal` staleness (main.py)

`main.py` kamera döngüsünde, identified face track'leri gezen `for t in active:` döngüsünden **önce**
her kare `is_frontal`'ı resetle; sonra frontal+identified yüz varsa `True` olur (mevcut atama bunu
yapıyor).

```python
            gesture_rec.is_frontal = False          # <-- EKLE: döngüden önce her kare reset
            for t in active:
                t_id = t["id"]
                x, y, w, h = t["bbox"]
                user = t["user"]
                if user not in ["Unknown", "Identifying..."]:
                    is_frontal = quality_gate.estimate_frontality(
                        t["detection_keypoints"], frame_w, frame_h, (x, y, w, h),
                    )
                    gesture_rec.is_frontal = is_frontal
                    ...
```

**Trade-off (kullanıcıya not):** Bu fix ile `Thumb_Up`/`Victory`, yalnızca **aktif + frontal bir
face track** varken çalışır (gaze-lock'un asıl amacı bu; yanlış-pozitifleri azaltır). **Closed_Fist
(emergency) gaze-lock'a tabi değil → etkilenmez**, Fix 1 ile zaten çalışır. Eğer `Thumb_Up`'ın yüz
track'i olmadan, sadece person-track ile de çalışması istenirse bu ayrı bir geliştirme: frontality'yi
fall pose landmark'larından (burun/omuz simetrisi) türetmek — **bu işin kapsamı dışı**, ayrıca ele al.

---

## Fix 3 — El ROI'sini fall pose'undan ayır, person-track'e bağla

**Karar:** Gesture ROI kaynağı artık **her kare mevcut olan person track'leri** olsun. Eşleşen bir
fall pose varsa wrist'lerle hassas el kutusu kullan; yoksa **person bbox crop**'una düş (tüm kare
değil). Böylece gesture tespiti fall state'inden ve oturma pozisyonundan bağımsızlaşır.

### `detection/gesture_recognizer.py`

**(a)** Hafif bir region tipi ekle (dosyanın üstüne, `WristPose` Protocol'ünün yanına). `WristPose`
Protocol'ü zaten `track_id`, `bbox`, `left_wrist`, `right_wrist` içeriyor — bu dataclass ona uyar:
```python
from dataclasses import dataclass

@dataclass
class GestureRegion:
    track_id: int
    bbox: tuple[int, int, int, int]
    left_wrist: tuple[float, float] | None
    right_wrist: tuple[float, float] | None
```

**(b) `_raw_hand_bbox`** — wrist varsa hassas kutu; yoksa region bbox crop'una düş (artık `continue`
ile tüm kareye düşmüyor):
```python
    def _raw_hand_bbox(
        self,
        rgb_frame: np.ndarray,
        regions: Sequence[WristPose] | None,
    ) -> TrackedHandBBox | None:
        if not regions:
            return None

        frame_h, frame_w = rgb_frame.shape[:2]
        fallback: TrackedHandBBox | None = None
        for region in regions:
            wrist_points = [
                point for point in (region.left_wrist, region.right_wrist) if point is not None
            ]
            person_x, person_y, person_w, person_h = region.bbox

            if wrist_points:
                pad = int(round(max(person_w, person_h) * self._cfg.hand_crop_pad / 2.0))
                xs = [point[0] for point in wrist_points]
                ys = [point[1] for point in wrist_points]
                x1 = max(0, int(round(min(xs) - pad)))
                y1 = max(0, int(round(min(ys) - pad)))
                x2 = min(frame_w, int(round(max(xs) + pad)))
                y2 = min(frame_h, int(round(max(ys) + pad)))
                if x2 > x1 and y2 > y1:
                    return (x1, y1, x2 - x1, y2 - y1), region.track_id

            if fallback is None:
                fx1 = max(0, min(frame_w, person_x))
                fy1 = max(0, min(frame_h, person_y))
                fw = max(1, min(frame_w - fx1, person_w))
                fh = max(1, min(frame_h - fy1, person_h))
                fallback = (fx1, fy1, fw, fh), region.track_id

        return fallback
```
> Mantık: wrist'li **ilk** region → hassas el kutusu döner. Hiçbirinde wrist yoksa **ilk** region'ın
> person bbox crop'u döner. `_hand_roi` ve EMA aynen kalır (yalnız `poses` param adını `regions` yap).

### `main.py`

Kamera döngüsünde gesture'a beslemeyi değiştir. `pose_tracks` (fall wrist kaynağı) zaten oluşuyor;
ona ek olarak **person_tracks**'ten her kare region listesi kur ve wrist'leri eşleşen pose'dan iliştir:

```python
            # mevcut: pose_tracks fall sonuçlarından dolduruluyor ...
            pose_by_track = {pose.track_id: pose for pose in pose_tracks}
            gesture_regions = [
                GestureRegion(
                    track_id=int(person_track["id"]),
                    bbox=cast(BBox, person_track["bbox"]),
                    left_wrist=(
                        pose_by_track[int(person_track["id"])].left_wrist
                        if int(person_track["id"]) in pose_by_track else None
                    ),
                    right_wrist=(
                        pose_by_track[int(person_track["id"])].right_wrist
                        if int(person_track["id"]) in pose_by_track else None
                    ),
                )
                for person_track in person_tracks
            ]
            gesture_rec.process(rgb_frame, current_time, gesture_regions)   # eski: pose_tracks
            gesture_rec.clear_stale(current_time)
```
`GestureRegion`'ı import et: `from detection.gesture_recognizer import GestureRecognizer, GestureRegion`.

> Artık person track olduğu sürece (her kare) gesture ROI mevcut; fall pose yalnız wrist hassasiyeti
> için kullanılıyor. Person track yoksa `gesture_regions` boş → `_raw_hand_bbox` None → tüm kare
> (eski güvenli davranış).

---

## Tests — `tests/test_phase_a_contracts.py`

Mevcut `test_raw_hand_bbox_returns_pose_track_id` **aynen geçmeli** (wrist'li region hâlâ
`((32, 42, 36, 36), 7)` döner). Şunları ekle:

- **`test_raw_hand_bbox_falls_back_to_person_bbox_without_wrist`**:
  ```python
  region = types.SimpleNamespace(track_id=5, bbox=(10, 20, 30, 40), left_wrist=None, right_wrist=None)
  result = recognizer._raw_hand_bbox(np.zeros((100, 100, 3), dtype=np.uint8), [region])
  assert result == ((10, 20, 30, 40), 5)
  ```
- **`test_process_submits_every_frame_without_inflight_gate`** (Fix 1 regresyon kilidi):
  ```python
  recognizer = GestureRecognizer.__new__(GestureRecognizer)
  recognizer._cfg = PipelineConfig()
  recognizer._last_timestamp_ms = 0
  recognizer._async_image_buffer = []
  recognizer._pending_track_id = None
  calls: list[int] = []
  class FakeRec:
      def recognize_async(self, image, ts): calls.append(ts)
  recognizer._recognizer = FakeRec()
  frame = np.zeros((100, 100, 3), dtype=np.uint8)
  recognizer.process(frame, 1.0, None)
  recognizer.process(frame, 1.1, None)   # arada callback YOK
  assert len(calls) == 2                  # kilit kaldırıldı → ikisi de submit
  ```
  > `_is_processing` alanı kaldırıldığından bu test onun olmadığını da doğrular. (mp.Image gerçek
  > mediapipe kullanır; ortamda kurulu.)

---

## Doğrulama (hepsi geçmeli)

```bash
uv run ruff check detection/gesture_recognizer.py main.py tests/test_phase_a_contracts.py
uv run mypy detection/gesture_recognizer.py main.py
uv run pytest -q
uv run python -c "import main, detection.gesture_recognizer; print('ok')"
git diff --check
```
Beklenen: ruff/mypy temiz, mevcut testler + yeni testler **passed**, import ok.

### Runtime doğrulama (kullanıcı)
1. Pipeline'ı başlat, kameraya gir (person track oluşsun).
2. **Otururken** ve **ayaktayken** Closed_Fist yap → Mac log'da `Sustained gesture: Closed_Fist`
   çıkmalı (artık ağır yükte sessizliğe düşmemeli), Pi'de emergency mapping tetiklenmeli.
3. Uzun süre (birkaç dk, fall-pose yükü altında) jest yapmaya devam et → gesture **ölmemeli**
   (Fix 1 doğrulaması).
4. `Thumb_Up`: yüze bakarken çalışır (Fix 2); yüz track'i yokken bilinçli olarak bastırılır.

---

## Riskler / dikkat

- **Fix 1 davranışı:** Kilit kaldırılınca `_pending_track_id` her kare set edilir; callback en son
  submit edilen kareye karşılık gelir. Jestler ~1sn sürdüğü (çok kare) ve kişi komşu karelerde ayni
  track'te kaldığı için kimlik ataması yine doğru — best-effort, kabul edilebilir.
- **`_is_processing` referanslarını TAM temizle:** `__init__`, `process` (eski `if not ...` + except
  reset), `_on_result` (son satır). Hiçbir yerde kalmasın yoksa `AttributeError`.
- **Fix 3 multi-person:** `_raw_hand_bbox` wrist'li ilk region'ı seçer; iki kişi aynı anda el
  kaldırırsa biri işlenir — mevcut sınır, genişletme.
- **Fall detector'a dokunma:** `pose_tracks` üretimi aynen kalır; sadece wrist kaynağı olarak okunur.
- **Surgical kal:** 3 dosya. `user_for_person_track` ve kimlik akışı değişmez.
