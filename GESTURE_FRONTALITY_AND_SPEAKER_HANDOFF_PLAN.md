# Handoff — Pose-tabanlı Frontality (P4) + Speaker Eşiği (P2)

> İki bağımsız hızlı fix, iki repo:
> - **P4 (video_Process):** Gesture gaze-lock'unun `is_frontal`'ı artık **kırılgan face track'ten**
>   değil, **fall detector'ın pose landmark'larından** türetilsin. Böylece Thumb_Up/Victory, yüz track'i
>   expire olsa bile (kişi kameraya bakarken) çalışır; loglardaki yüzlerce `Gaze Lock: ignoring` biter.
> - **P2 (proactive-home-agent):** Speaker doğrulama eşiği 0.65 → **0.55** (Berkay 0.61–0.62 ile "Guest"e
>   düşüyordu).
>
> **Rol:** Codex implement eder. Kapsam ve doğrulama bağlayıcı. Surgical kal.

---

## Analiz — frontality pose'dan nasıl türetilir

Mevcut yüz-tabanlı `quality_gate.estimate_frontality`: göz_mesafesi / yüz_genişliği `< min_eye_distance_ratio`
ise profil sayıyor. Bunun **pose karşılığı**: MediaPipe Pose'da NOSE, LEFT/RIGHT_EYE, LEFT/RIGHT_SHOULDER
landmark'ları var (`.x` normalize, `.visibility` 0–1). Kişi kameraya bakarken iki göz de görünür ve
yatay göz aralığı omuz genişliğinin anlamlı bir oranıdır; profile dönünce uzak göz/omuz görünürlüğü
düşer **ve** göz aralığı daralır. Kriter:

```
frontal  ⇔  min(vis(nose,le,re,ls,rs)) ≥ V   ve   |le.x - re.x| / |ls.x - rs.x| ≥ R
```
`V`, `R` config'ten (default `V=0.5`, `R=0.30`). Tüm pose üretimi `_pose_track_data(landmarks)`'tan
geçtiği için frontality'yi orada hesaplayıp `PoseTrackData`'ya bir `frontal: bool` alanı olarak taşırız.
`main.py` her kare `gesture_rec.is_frontal`'ı pose'lardan set eder.

---

## P4 — video_Process

### 1) `config.py`
Fall/pose ayarlarının yanına ekle:
```python
    pose_frontal_min_visibility: float = 0.5
    pose_frontal_eye_shoulder_ratio: float = 0.30
```

### 2) `detection/fall_detector.py`

**(a) `PoseTrackData`** dataclass'ına alan ekle (default'lu, geriye uyumlu):
```python
    frontal: bool = False
```

**(b)** Frontality helper'ı ekle (sınıf metodu):
```python
    def _pose_is_frontal(self, landmarks: list[object]) -> bool:
        lm = self._mp_pose.PoseLandmark
        nose = landmarks[lm.NOSE.value]
        le = landmarks[lm.LEFT_EYE.value]
        re = landmarks[lm.RIGHT_EYE.value]
        ls = landmarks[lm.LEFT_SHOULDER.value]
        rs = landmarks[lm.RIGHT_SHOULDER.value]

        vis_min = self._cfg.pose_frontal_min_visibility
        if min(
            nose.visibility, le.visibility, re.visibility, ls.visibility, rs.visibility
        ) < vis_min:
            return False

        shoulder_dx = abs(ls.x - rs.x)
        if shoulder_dx <= 1e-6:
            return False
        eye_dx = abs(le.x - re.x)
        return (eye_dx / shoulder_dx) >= self._cfg.pose_frontal_eye_shoulder_ratio
```

**(c) `_pose_track_data`** — dönen `PoseTrackData`'ya `frontal` ekle:
```python
        return PoseTrackData(
            track_id=track_id,
            bbox=person_bbox,
            crop_bbox=crop_bbox,
            left_wrist=left_wrist,
            right_wrist=right_wrist,
            frontal=self._pose_is_frontal(landmarks),
        )
```
> Not: `_pose_track_data` zaten hem `_run_detection` (IDLE) hem MONITORING yolundan çağrılıyor; tek
> nokta yeterli. `landmarks` = `results.pose_landmarks.landmark`.

### 3) `main.py`

Fix 2'de eklenen face-tabanlı frontality'yi **kaldır**, pose-tabanlıyla değiştir:

- `for t in active:` döngüsünden **önceki** `gesture_rec.is_frontal = False` satırını **sil**.
- Döngü **içindeki** şu iki satırı **sil** (presence dispatch'i KORU):
  ```python
  is_frontal = quality_gate.estimate_frontality(
      t["detection_keypoints"], frame_w, frame_h, (x, y, w, h),
  )
  gesture_rec.is_frontal = is_frontal
  ```
  (Geriye `if user not in ["Unknown", "Identifying..."]:` guard'ı + `last_json_time` presence bloğu kalır.)
- `pose_tracks` tamamen kurulduktan sonra, `gesture_rec.process(...)` çağrısından **hemen önce** ekle:
  ```python
  if pose_tracks:
      gesture_rec.is_frontal = any(pose.frontal for pose in pose_tracks)
  gesture_rec.process(rgb_frame, current_time, pose_tracks)
  ```

> **Sticky davranış kasıtlı:** `is_frontal` yalnız pose olan karelerde güncellenir (fall FPS ~15Hz,
> kişi varken sürekli üretiliyor). Pose'suz karelerde son pose-değeri korunur → async gesture
> callback'i False bir ana denk gelip jesti yanlışça gaze-lock'lamaz. Kişi gidince person track
> düşer, zaten gesture üretilmez.
>
> `quality_gate.estimate_frontality`'ye **dokunma** — quality gate'in 4. katmanında (yüz crop kalite)
> hâlâ kullanılıyor; o ayrı amaç.

### 4) `tests/test_phase_a_contracts.py`
`FallDetector.__new__(FallDetector)` ile instance; `_cfg = PipelineConfig()`,
`_mp_pose = mp.solutions.pose` ata. 33 elemanlı sahte landmark listesi kur (her biri
`types.SimpleNamespace(x=..., y=..., visibility=...)`), ilgili indeksleri doldur:
- **`test_pose_is_frontal_true_when_facing_camera`**: iki göz net ayrık + tüm vis yüksek → `True`.
  Örn. shoulders x=0.3/0.7 (dx=0.4), eyes x=0.45/0.55 (dx=0.10 → 0.10/0.40=0.25 < 0.30 → False!).
  Dikkat oranı sağla: eyes x=0.4/0.6 (dx=0.20 → 0.20/0.40=0.5 ≥ 0.30) → `True`.
- **`test_pose_is_frontal_false_on_profile`**: uzak gözün `visibility=0.1` (vis_min altı) → `False`.
- **`test_pose_is_frontal_false_when_eyes_too_close`**: tüm vis yüksek ama eyes dx küçük
  (0.45/0.55, shoulders 0.3/0.7 → 0.10/0.40=0.25 < 0.30) → `False`.

> mediapipe ortamda kurulu; `mp.solutions.pose.PoseLandmark.NOSE.value` vb. indeksler kullanılır.

### Doğrulama (P4)
```bash
uv run ruff check config.py detection/fall_detector.py main.py tests/test_phase_a_contracts.py
uv run mypy config.py detection/fall_detector.py main.py
uv run pytest -q
uv run python -c "import main, detection.fall_detector; print('ok')"
git diff --check
```

---

## P2 — proactive-home-agent

### `backend/api/services/speaker_service.py`
`identify_speaker` default eşiğini düşür:
```python
    def identify_speaker(self, audio_bytes: bytes, threshold=0.55):   # eski: 0.65
```
**Çağıranları kontrol et:** `grep -rn "identify_speaker(" backend/` — eğer bir çağrı eşiği **açıkça**
`threshold=0.65` geçiyorsa orayı da 0.55 yap (yoksa default yeterli). Başka davranış değiştirme.

### Doğrulama (P2)
```bash
cd backend && python -c "import ast; ast.parse(open('api/services/speaker_service.py').read()); print('ok')"
git diff --check
grep -rn "identify_speaker(" .   # 0.65 kalan çağrı olmadığını teyit
```

---

## Riskler / dikkat
- **P4 kapsamı:** Sadece `config.py`, `detection/fall_detector.py`, `main.py`, test. `quality_gate`'e,
  Fix 1/Fix 3'e, kimlik çözümüne dokunma.
- **`is_frontal` tek-flag (multi-person):** `any(pose.frontal ...)` — birden çok kişi varsa biri frontal
  olunca gaze-lock açılır. Mevcut tek-flag tasarımıyla tutarlı; jest kimliği ayrıca person-track'ten
  çözülüyor. Genişletme bu işin dışında.
- **P2:** 0.55 sahte-eşleşme riskini biraz artırır; ürün kararı kullanıcı tarafından verildi. Sadece
  default değişiyor.
- İki repo ayrı commit/iş; karıştırma.
