# Handoff — Gesture'ı Person-Track Kimliğine Bağlama

> **Sorun:** Yeni sistemde jestler `Unknown` olarak dispatch ediliyor → Closed_Fist emergency
> protokolü tetiklenmiyor. Kök neden: `TrackerManager.get_active_user()` yalnızca **face**
> track'lerine bakıyor; face track'leri hızla expire olduğu için jest anında genelde identified
> bir face track yok. Oysa kimlik, **person** track'inde uzun süre yaşıyor (`propagate_identity`).
>
> **Hedef:** Jestin ROI'si hangi person track'inin bileğinden üretildiyse (her `PoseTrackData`
> zaten `track_id` taşır), o person track'inin kimliğine çözümle. Person kayıtlı bir kullanıcıysa
> jesti ona ata; değilse `Unknown`.
>
> **Rol:** Codex implement eder. Aşağıdaki kapsam ve doğrulama bağlayıcı. Surgical kal.

---

## Kapsam (sadece bunlar)

1. `tracking/tracker_manager.py` — yeni `user_for_person_track(track_id)` metodu.
2. `detection/gesture_recognizer.py` — ROI'yi üreten pose'un `track_id`'sini yakala; async sonuçta
   o track'in kimliğine çözümle.
3. `main.py` — gesture recognizer'a yeni resolver'ı bağla.
4. `tests/test_phase_a_contracts.py` — yeni resolver + binding için testler.

**Kapsam dışı:** Pi tarafı (emergency mapping zaten kayıtlı kullanıcıyla çalışıyor — bizim işimiz
doğru kullanıcıyı göndermek), multi-hand aynı anda iki kişi (ön mevcut sınır), enrollment, face
tanıma eşiği. Bunlara dokunma.

---

## Tasarım kararları (kilit)

- **Kimlik kaynağı = person track.** Jest, ROI'yi üreten pose'un `track_id`'sine bağlanır; o person
  track'inin `emitted_user or user` değeri kullanıcıdır. Face track'i artık gesture için sorgulanmaz.
- **Async eşleşme.** MediaPipe `recognize_async`; sonuç `_on_result`'a sonradan, başka thread'de gelir.
  ROI hangi pose'dan üretildiyse, o submission'a ait `track_id`'yi **submit anında** sakla
  (`_pending_track_id`) ve `_on_result`'ta onu çöz. `_is_processing` guard'ı zaten iki submission
  arası callback'i garanti ettiği için bu track_id ile sonuç birebir eşleşir.
- **track_id yoksa fallback.** Pose hiç yoksa (ROI tüm kare) `track_id = None` → `user_for_person_track(None)`
  herhangi bir identified person track'ine düşer (tek kullanıcılı evde eski davranışla uyumlu). track_id
  **var ama** person Unknown ise → `Unknown` döndür (o kişi gerçekten tanınmıyor; uydurma yapma).
- **Geriye dönük dispatch davranışı korunur.** `Unknown` jestler hâlâ gönderilir (Pi onları sessiz
  loglar); değişen tek şey, tanınan kişinin artık doğru atanması.

---

## Adım adım

### 1) `tracking/tracker_manager.py`

`get_active_user`'ın hemen yanına yeni metodu ekle (mevcut `get_active_user`'ı **silme/değiştirme** —
başka yerde kullanılmıyorsa bile dokunma; surgical kal):

```python
    def user_for_person_track(self, track_id: int | None) -> str:
        """Resolve the identity of a person track for gesture attribution.

        Returns the named user of the given person track (preferring the
        emitted identity). Falls back to any identified person track only when
        track_id is None (ROI had no pose). Returns 'Unknown' otherwise.
        """
        with self._lock:
            if track_id is not None:
                record = self._person_tracks.get(track_id)
                if record is not None:
                    user = record.emitted_user or record.user
                    return user if user not in ("Unknown", "Identifying...") else "Unknown"
                return "Unknown"
            for record in self._person_tracks.values():
                user = record.emitted_user or record.user
                if user not in ("Unknown", "Identifying..."):
                    return user
        return "Unknown"
```

> Not: `track_id` verildi ama o track yoksa → `Unknown` (fallback'a düşme; jestin sahibi o track'ti,
> kaybolduysa atfetme). Sadece `track_id is None` iken global identified person'a düş.

### 2) `detection/gesture_recognizer.py`

**(a) Constructor** — callback adını niyeti yansıtacak şekilde değiştir:

```python
    def __init__(self, cfg: PipelineConfig, dispatcher: EventDispatcher, get_user_for_track):
        """
        Args:
            get_user_for_track: callable(track_id: int | None) -> str, thread-safe;
                returns the identified user for a person track (or 'Unknown').
        """
        self._cfg = cfg
        self._dispatcher = dispatcher
        self._get_user_for_track = get_user_for_track
        ...
        self._pending_track_id: int | None = None   # diğer init alanlarının yanına ekle
```

**(b) `_raw_hand_bbox`** — döndürdüğü bbox ile birlikte pose'un `track_id`'sini de ver. İmzayı
`tuple[int, int, int, int] | None` → `tuple[tuple[int, int, int, int], int] | None` yap ve her
`return` noktasında matched pose'un `track_id`'sini ekle. Wrist'li ilk pose seçildiği için onun
`pose.track_id`'si kullanılır. `WristPose` Protocol'üne `track_id: int` alanını ekle (PoseTrackData
zaten taşıyor).

Örnek (wrist'li dönüş):
```python
            if x2 > x1 and y2 > y1:
                return (x1, y1, x2 - x1, y2 - y1), pose.track_id
            ...
            return (
                (max(0, min(frame_w, person_x)), ...),
                pose.track_id,
            )
```
Hiç pose/wrist yoksa `return None` (eskisi gibi).

**(c) `_hand_roi`** — `track_id`'yi yukarı taşı. İmzayı `np.ndarray` → `tuple[np.ndarray, int | None]`
yap. bbox `None` ise `(rgb_frame, None)` döndür; aksi halde kırpılmış ROI ile pose'un `track_id`'sini
döndür. (Mevcut EMA smoothing mantığı aynen kalsın; sadece track_id'yi taşı.)

**(d) `process`** — track_id'yi submit anında sakla:
```python
            roi, track_id = self._hand_roi(rgb_frame, poses)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi.copy())
            self._async_image_buffer.append(mp_image)
            if len(self._async_image_buffer) > 30:
                self._async_image_buffer.pop(0)

            if not self._is_processing:
                self._is_processing = True
                self._pending_track_id = track_id   # bu submission'ın sahibi person track
                self._recognizer.recognize_async(mp_image, current_ms)
```

**(e) `_on_result`** — kullanıcıyı pending track'ten çöz:
```python
                            detected_user = self._get_user_for_track(self._pending_track_id)
                            self._dispatcher.submit(
                                self._dispatcher.send_gesture_event,
                                top_gesture, duration, detected_user,
                            )
```

### 3) `main.py`

Gesture recognizer kurulumunda (satır ~67-70) bağlamayı değiştir:
```python
        self._gesture_rec = GestureRecognizer(
            self._cfg,
            self._dispatcher,
            self._tracker_mgr.user_for_person_track,   # eski: get_active_user
        )
```

**(Opsiyonel sertleştirme — ayrı, küçük):** `gesture_rec.is_frontal` yalnızca identified face track
varken güncelleniyor (satır 184-191); face track yokken stale kalıyor ve Thumb_Up/Victory gaze-lock'unu
yanıltabilir. İstersen `for t in active:` döngüsünden ÖNCE `gesture_rec.is_frontal = False` ile
her frame resetle, sonra identified+frontal face varsa `True` yap. Closed_Fist'i etkilemez; istersen
bu adımı atla.

### 4) `tests/test_phase_a_contracts.py`

`test_tracker_manager_propagates_identified_face_to_containing_person` (satır ~643) desenini örnek al
(`TrackerManager.__new__` + alan set). Şunları ekle:

- **`test_user_for_person_track_resolves_named_person`**: `_person_tracks = {7: PersonTrackRecord(..., user="Ada", emitted_user="Ada", ...)}`, `_lock` ata; `manager.user_for_person_track(7) == "Ada"`.
- **`test_user_for_person_track_unknown_when_track_unidentified`**: person track `user="Unknown"` → `user_for_person_track(7) == "Unknown"`.
- **`test_user_for_person_track_missing_track_returns_unknown`**: `user_for_person_track(999) == "Unknown"` (boş/eksik track).
- **`test_user_for_person_track_none_falls_back_to_any_identified`**: bir identified person track varken `user_for_person_track(None) == "Ada"`.
- **`test_raw_hand_bbox_returns_pose_track_id`**: `GestureRecognizer.__new__` ile instance; `_cfg` set;
  tek bir fake pose (`track_id=7`, `bbox`, `right_wrist` dolu) ver; `_raw_hand_bbox(frame, [pose])`
  dönüşünün `(bbox, 7)` olduğunu doğrula.

> `test_gesture_payload_contract_unchanged` GestureRecognizer'ı **kurmuyor** (sadece dispatcher'ı test
> ediyor) → constructor değişikliği onu kırmaz; ama yine de geçtiğini teyit et.

---

## Doğrulama (hepsi geçmeli)

```bash
uv run ruff check tracking/tracker_manager.py detection/gesture_recognizer.py main.py tests/test_phase_a_contracts.py
uv run mypy tracking/tracker_manager.py detection/gesture_recognizer.py main.py
uv run pytest -q
uv run python -c "import main, detection.gesture_recognizer, tracking.tracker_manager; print('ok')"
git diff --check
```

Beklenen: ruff/mypy temiz, mevcut testler kırılmadan + yeni testler **passed**, import ok.

### Manuel / runtime doğrulama (kullanıcı yapacak)
1. Pi'de `init_db.py` sonrası `ogulcan` enroll edilmiş ve Mac galerisinde **stale etiket olmadığından**
   emin ol (OG/meto/teko temizliği — ayrı iş, bkz. log analizi).
2. Mac pipeline + Pi backend çalışırken kameraya bak (person track identified olsun), sonra **yüzü
   kameradan çevirmeden gerekmeden** el kaldırıp jest yap.
3. Mac log: `Sustained gesture: ...` satırının dispatch ettiği kullanıcı artık `ogulcan` olmalı.
4. Pi log: `Gesture Logged ... ogulcan made a '...'` — `Unknown` DEĞİL.
5. Closed_Fist ile emergency mapping'in tetiklendiğini doğrula.

---

## Riskler / dikkat

- **Async race:** `_pending_track_id`'yi MUTLAKA `if not self._is_processing:` bloğunun içinde
  (submit anında) yaz; her `process()` çağrısında değil. Aksi halde callback beklenirken üzerine yazılır.
- **`track_id` var ama Unknown → Unknown.** Fallback'a yalnızca `track_id is None` iken düş. Yanlış
  kişiye jest atfetme riskine karşı.
- **Multi-person:** `_raw_hand_bbox` hâlâ wrist'li **ilk** pose'u seçiyor; aynı anda iki kişi el
  kaldırırsa yalnızca biri işlenir — bu **mevcut sınır**, bu işin kapsamı değil, genişletme.
- **Surgical kal:** Sadece 4 dosya. `get_active_user`'ı silme, başka davranış/endpoint değiştirme,
  enrollment/eşik/Pi'ye dokunma.
