# Analiz Raporu — video_Process Enrollment'ın Agentic Tanıma Üzerindeki Etkisi

> Soru: Kullanıcı fotoğrafının `video_Process` üzerinden enroll edilmesi, `proactive-home-agent`
> tarafındaki agentic sistemin kullanıcıyı tanımasını etkiler mi?
>
> Yöntem: Pi'deki kimlik akışı ve agent'ın kimliği nasıl tükettiği gerçek kodda izlendi (file:line).
> Kapsam: presence, identity events, gesture, greeting, kalıcı hafıza, konuşma/vektör hafızası.

---

## 0. Tek cümlelik cevap

**Evet — ama doğru yönde: `video_Process` enrollment'ı artık tanımanın TEK kaynağı.** Kimi tanıdığını
Mac belirliyor; Pi agent'ı yüz tanımıyor, sadece **isim string'ini** tüketiyor. Temel "tanı + isimle
karşıla" akışı Pi DB'ye ihtiyaç duymadan çalışır. **Kritik bağımlılık tek bir değişmezde toplanıyor:
Mac galeri etiketi (`label`) == Pi `User.username`.** Bu parite bozulursa jest-güvenlik (SOS, özel
jestler) ve kalıcı `last_seen` hafızası **sessizce** çalışmaz.

---

## 1. Tanıma artık nerede yapılıyor (akış)

```
[Mac video_Process]                                  [Pi proactive-home-agent]
 enroll (faces.db, ArcFace)                            (yüz TANIMAZ — sadece isim alır)
        │ canlı kare → YOLO-face → ArcFace                       ▲
        │ gallery match → label ("Berkay" | "Unknown")           │ isim string'i
        ▼                                                         │
 EventDispatcher ──POST──►  /vision/update_presence  ───► handle_detection(name)
                  ──POST──►  /vision/identity_event   ───► log_identity_session + trigger_agent
                  ──POST──►  /vision/gesture          ───► handle_gesture(user=name, gesture)
```

- **Tanıma (kim bu?) tamamen Mac'te** (`identification/face_identifier.py`, ArcFace galeri).
  Galeriyi besleyen şey `video_Process` enrollment'ı. Enroll edilmemiş biri → Mac "Unknown" gönderir →
  agent isimle karşılamaz. Yani **enrollment, kullanıcının tanınıp tanınmayacağını doğrudan belirleyen
  bileşendir.**
- Pi'deki `vision_service.recognize` + `/vision/identify` (`vision_router.py:728`) **artık atıl** —
  Mac lokal ArcFace yaptığı için (önceki review'daki F1) bu yol çağrılmıyor. Pi'nin DeepFace galerisi
  ölü (F2: farklı embedding uzayı).
- Pi'ye giden sözleşme: **isim string'i** (`event.user` / `person_name`). Mac'in galeri etiketi neyse
  agent onu görür.

---

## 2. Agentic sistemin kimlik temas noktaları (kanıt + bağımlılık)

| # | Yer (file:line) | Ne yapıyor | Pi `User` kaydı GEREKLİ mi? |
|---|---|---|---|
| A | `presence_service.handle_detection` (`presence_service.py:115`) | İsim != "Unknown" → ENTRY/PRESENT; bilinmeyene grace period | **Hayır** — saf string, in-memory `active_people` |
| B | `vision_router.trigger_agent_proactively` (`vision_router.py:98-191`) | Karşılama prompt'u: "Greet **{person_name}** warmly", away-time context (history_ledger'dan isimle) | **Hayır** — isim prompt'a metin olarak gömülüyor |
| C | Greeting anti-spam (`vision_router.py:113-119`) | İsim-keyed dict ile 900s spam engeli | **Hayır** — string key |
| D | `presence_service._update_db_last_seen` (`presence_service.py:71-86`) | `select(User).where(username==name)` → `user.last_seen` güncelle | **Evet (soft)** — kayıt yoksa **sessiz no-op** |
| E | `handle_gesture` (`vision_router.py:579-655`) | `User` by name → `user_obj.id` → `SecuritySettings` + `GestureMapping` (owner_id) | **Evet (HARD)** — kayıt yoksa SOS jesti + özel jest haritaları **çalışmaz** |
| F | `handle_identity_event` (`vision_router.py:664-678`) | WS2 entered/left → presence + trigger; (default-off) | **Hayır** (D üzerinden dolaylı soft) |
| G | Konuşma hafızası `agent/graph.py:164-171` | LangGraph `MemorySaver`, `thread_id="home_system_thread"` (TEK paylaşımlı thread) | **Hayır** — kullanıcıya göre namespace YOK |
| H | Uzun-dönem hafıza `vector_db.py:19-74` | episodic/semantic ChromaDB koleksiyonları | **Hayır** — kullanıcı-namespace YOK (global) |

**Sonuç:** Agent'ın kimlik kullanımı büyük ölçüde **isim string'i** üzerinden. DB'de `User` satırına
gerçek bağımlılık yalnızca **iki** yerde: **E (jest-güvenlik, HARD)** ve **D (kalıcı last_seen, soft)**.

---

## 3. video_Process enrollment'ının alt-sistem bazında etkisi

| Agentic yetenek | Mac'te enroll edilmiş + Pi User kaydı VAR | Mac'te enroll + Pi kaydı YOK | Mac'te enroll YOK |
|---|---|---|---|
| İsimle tanıma & karşılama (B,C) | ✅ Çalışır | ✅ **Çalışır** (saf isim) | ❌ "Unknown" → isimsiz |
| Presence ENTRY/EXIT, away-time (A,B) | ✅ | ✅ (oturum içi history_ledger) | ⚠️ "A Stranger" akışı |
| Jest → SOS / acil kilit (E) | ✅ | ❌ **Sessizce çalışmaz** (user_obj=None) | ❌ |
| Özel jest haritaları (E) | ✅ | ❌ **Çalışmaz** | ❌ |
| Kalıcı `last_seen` (D) | ✅ | ⚠️ No-op (restart sonrası unutur) | — |
| Konuşma / uzun-dönem hafıza (G,H) | ✅ (zaten kullanıcı-bağımsız) | ✅ | ✅ |

**Okuma:** Enrollment, "tanınma" kapısını açan bileşendir. Pi User kaydı ise yalnızca **jest-güvenlik**
ve **kalıcı hafıza** için ek olarak gerekir.

---

## 4. Kritik değişmez: ETİKET PARİTESİ (label == username)

D ve E exact-match sorgu kullanıyor (`User.username == name`, SQLite'ta varsayılan **case-sensitive**).
Yani Mac galeri etiketi ile Pi `User.username` **birebir** (büyük/küçük harf + boşluk) eşleşmeli.

- **Homeowner (onboarding):** ✅ Güvenli. Frontend etiketi `localStorage.username`'den alıyor
  (`vision-api.ts:74`), bu da login'de Pi'den gelen `res.data.username` (`login/page.tsx:83`). Aynı
  string → parite korunur. Pi User kaydı zaten signup'ta var (role=admin).
- **Guest (dashboard):** ⚠️ Riskli. Önceki review'daki boşluk burada ısırıyor: yeni akışta guest yüzü
  Mac'e gidiyor, Pi `add-guest` **yalnızca ses varsa** çağrılıyor. Ses kaydı olmayan guest:
  - Mac `faces.db`'de var → tanınır + isimle karşılanır ✅
  - Pi `User` tablosunda **yok** → jest-güvenlik (E) çalışmaz, last_seen (D) no-op.
  Ayrıca homeowner guest adını elle yazıyor (`name.trim()`) → Pi'de bir kayıt olsa bile **harf farkı**
  pariteyi bozabilir.
- **Etiket kayması (genel):** Mac `_clean_label` sadece `strip()` yapıyor, case normalize etmiyor
  (`face_repository.py:202`). "berkay" (Mac) vs "Berkay" (Pi) → D/E sessizce başarısız.

---

## 5. Riskler ve öneriler

- **R1 — Guest jest-güvenliği & kalıcı hafıza (önceki review'ın doğal uzantısı).** Ses olmadan eklenen
  guest Pi'de kayıtsız kalıyor → E sessizce çalışmaz. **Öneri:** guest oluşturmayı sesten ayır — yüz
  enrollment başarılıysa Pi'de bir guest `User` kaydı da oluştur (hafif bir "guest metadata" çağrısı),
  böylece parite ve E/D garanti altına alınır. (Karar D3 zaten "Pi metadata sahibi" demişti.)
- **R2 — Etiket parite garantisi.** Enroll sırasında kullanılan etiketi tek otoriteye bağla: onboarding
  zaten `localStorage.username` kullanıyor (iyi). Guest tarafında, Pi guest kaydı oluştururken aynı
  normalize edilmiş ismi kullan; Mac `_clean_label` ile Pi guest username üretimini **aynı kurala** tabi
  tut (örn. ikisi de trim; case'i koru ama iki tarafta da aynı string).
- **R3 — Sessiz başarısızlık görünürlüğü.** D ve E, kayıt bulunamadığında log dahi üretmiyor (D'de
  `if user:` bloğu sessiz; E'de `if user_obj:` sessiz). **Öneri:** Pi tarafında "isim X event aldı ama
  User kaydı yok" uyarısı logla — parite kopması production'da fark edilsin.
- **R4 — Atıl yol temizliği.** `/vision/identify` + `vision_service` + Pi `User.face_embedding`
  (GhostFaceNet) artık kullanılmıyor; karışıklığı önlemek için deprecate/işaretle (F2). Yanlışlıkla
  tekrar devreye girerse ArcFace uzayıyla uyuşmaz.

---

## 6. Net değerlendirme

- **Tanıma etkilenir mi? Evet — olumlu ve tasarlandığı gibi.** `video_Process` enrollment'ı kullanıcının
  agent tarafından tanınmasının **kaynağıdır**; doğru etiketle enroll edilen kullanıcı isimle tanınır ve
  karşılanır, üstelik Pi DB'ye ihtiyaç duymadan.
- **Tek gerçek koşul: etiket paritesi.** Homeowner için güvenli. **Guest için R1 düzeltilmeden**
  jest-güvenlik ve kalıcı hafıza sessizce eksik kalır.
- Konuşma/uzun-dönem hafıza ve karşılama mantığı kullanıcı-bağımsız olduğundan enrollment kaynağından
  etkilenmez.
