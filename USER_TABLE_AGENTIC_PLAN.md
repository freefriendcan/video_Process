# İmplementasyon Planı — Enrollment Mac'e Taşınınca Pi User Tablosu & Agentic Bütünlük

> Bağlam: Eskiden yüz embedding'i Pi'de (`User.face_embedding`) saklanıyordu; enrollment çağrısı User
> satırını da garanti ediyordu. Artık yüz `video_Process` (`faces.db`) tarafında. Bu plan, Pi'deki
> agentic sistemin (presence/greeting/gesture/kalıcı hafıza) ve auth'un sorun yaşamaması için **User
> tablosu yaşam döngüsünü** analiz eder ve gereken düzeltmeleri tanımlar.
>
> Kaynak rapor: `AGENTIC_RECOGNITION_IMPACT_PLAN.md` (R1–R4). Bu plan onu uygulanabilir iş kalemlerine
> çevirir + onboarding'den itibaren User oluşumunu detaylandırır.

---

## BÖLÜM A — User tablosu nasıl/ne zaman oluşuyor (detaylı analiz)

### A.1 Homeowner (asıl kullanıcı) — enrollment'tan BAĞIMSIZ ✅

Akış: **Register sayfası → `/auth/register` → User satırı**.

- `frontend/.../(auth)/register/page.tsx:50` → `api.post('/auth/register', ...)`.
- `auth_router.register` (`auth_router.py:60-78`): `username`+`email`+`hashed_password`(+role) ile
  **User satırını signup anında** yaratır. Yüz/ses ile **hiç ilgisi yok**.
- Onboarding bundan **sonra** geliyor ve auth-gated: Step1 `onPrev` → `/login` (`onboarding/page.tsx:72`),
  Step3 `localStorage.token` kullanıyor, tüm onboarding endpoint'leri `get_current_user` (JWT) istiyor →
  **onboarding'e gelen kullanıcının User satırı kesinlikle vardır.**
- Onboarding adımları User **oluşturmaz**: `onboarding/setup` Room/Device yazar, `onboarding/complete`
  yalnız `is_onboarding_complete=True` set eder (`onboarding_router.py:34,119`).
- Login: `/auth/login` → `TokenResponse.username = user.username` (`auth_router.py:101-106`); frontend
  bunu `localStorage.username`'e yazar (`login/page.tsx:83`). Mac enroll etiketi de buradan geliyor
  (`vision-api.ts:74`) → **etiket = username paritesi homeowner'da otomatik korunur.**

**Sonuç:** Homeowner için User tablosu oluşumu enrollment'tan tamamen ayrık. Migration **homeowner'ı
etkilemez** — bu, kullanıcının "önceden sorun yoktu" gözleminin sebebidir ve sonrası için de geçerli.

### A.2 Guest (misafir) — eskiden enrollment'a BAĞLIYDI ⚠️ (kırılan yer burası)

Eski akış: dashboard "Enroll New Profile" → her `add-guest` çağrısı (yüz görseliyle bile) **önce User
satırını** yaratıyordu (`user_router.py:69-91`: dummy `email`, sahte `hashed_password`, `role="guest"`,
`owner_id=current_user.id`), sonra ses/yüz embedding işliyordu.

Yeni akış (review edilen frontend): guest yüzü **Mac `/enroll`'a** gidiyor; Pi `add-guest` **yalnızca ses
varsa** çağrılıyor (`UserManager.handleSave`). Yani **sessiz** eklenen guest:
- Mac `faces.db`'de var → tanınır, isimle karşılanır,
- Pi `users` tablosunda **YOK** → aşağıdaki bağımlılıklar bozulur.

### A.3 User satırına gerçekten muhtaç olan tüketiciler

| Tüketici | Bağımlılık | User yoksa |
|---|---|---|
| `presence_service._update_db_last_seen` (`presence_service.py:71`) | `select(User).where(username==name)` | **Sessiz no-op** — kalıcı `last_seen` yazılmaz (D) |
| `handle_gesture` (`vision_router.py:585-655`) | `User`→`id`→`SecuritySettings`/`GestureMapping` | **Sessizce atlanır** — cihaz/SOS aksiyonu yok (E) |
| `/auth/biometric-login` (`auth_router.py:108-135`) | `vision_service.recognize` (Pi DeepFace galeri) | **Her zaman "Unknown"** — yüzle giriş çöker (yeni bulgu) |

> Not (dürüst sınır): Guest User satırı oluşturulsa bile `handle_gesture` guest'in **kendi** `owner_id`'si
> ile `SecuritySettings` arıyor; guest tipik olarak SecuritySettings yapılandırmaz → guest-SOS yine
> ateşlenmez (bu muhtemelen **istenen** davranış). Yani guest satırı asıl olarak **D (last_seen) + kimlik
> bütünlüğünü** (orphan olmama, `user_obj=None` sessiz-fail'in kalkması) onarır. "Guest jestleri ev
> sahibinin ayarlarına mı bağlanmalı?" bir **ürün kararıdır**, bu planın kapsamı dışında.

### A.4 User modeli zorunlulukları (guest satırı yaratırken uyulacak)

`User` (`models.py:96-108`): `username` **unique**, `email` **unique + NOT NULL**, `hashed_password`
**NOT NULL**. Dolayısıyla guest satırı için de dummy unique email + placeholder password gerekir —
`add-guest` zaten bu kalıbı kullanıyor (`user_router.py:72-81`), şablon olarak alınmalı.

### A.5 Net teşhis

- **Homeowner: sorun yok** (signup'tan geliyor, parite otomatik).
- **Guest: User satırı artık oluşmuyor** → last_seen + temiz kimlik bütünlüğü kayıp (R1).
- **Etiket paritesi**: homeowner güvenli; guest'te elle yazılan ad iki tarafa farklı normalize edilirse
  kopar (R2).
- **biometric-login**: Pi galerisi boşaldığı için çöker (yeni — auth kapsamı).

---

## BÖLÜM B — İmplementasyon Planı

Katmanlama: **Pi (FastAPI) + frontend** (çift repo). CORS gibi tek-repo değil. Aşağıdaki WS-1/WS-2
agentic bütünlüğün çekirdeği; WS-3 ucuz görünürlük; WS-4 auth kararı; WS-5 temizlik.

### WS-1 — Guest User satırını sesten ayır (R1) — ÇEKIRDEK

**Pi (`proactive-home-agent`, `new-event`):**
- Yeni hafif endpoint: `POST /users/guest` — yalnız `name` (Form/JSON) alır, `get_current_user` ile
  korunur. Davranış: verilen ada sahip guest User **yoksa** oluştur (dummy unique email + placeholder
  password + `role="guest"` + `owner_id=current_user.id`), **varsa** no-op. **İdempotent**, döner:
  `{status, username, created: bool}`. Medya işlemez (yüz Mac'te, ses ayrı `add-guest`'te).
  - Gerekçe: `add-guest`'i medya mantığıyla yüklememek; niyet açık. (Alternatif: `add-guest`'in
    "en az bir dosya" zorunluluğunu (`user_router.py:66`) gevşetip name-only kabul etmek — daha küçük diff
    ama endpoint'i bulanıklaştırır. **Öneri: yeni endpoint.**)
- `delete_user` (`user_router.py:118`) zaten var; guest silme tarafında parite için frontend'in hem Mac
  `DELETE /enrolled/{label}` hem Pi `DELETE /users/{name}` çağırmasını değerlendirin (ayrı küçük karar).

**Frontend (`UserManager.handleSave`):**
- Mac `enrollFaceBatch(cleanName, faces)` **başarılı olduktan sonra**, sesten bağımsız olarak Pi
  `POST /users/guest` çağır (aynı `cleanName` ile). Ses varsa mevcut `add-guest` (audio-only) akışı kalır.
- Hata sırası: Mac enroll başarısızsa Pi guest oluşturma. Mac başarılı + Pi guest başarısız → kullanıcıya
  uyarı (yüz kaydedildi, metadata oluşturulamadı) ve retry imkânı.

→ **Doğrulama:** sessiz (ses kaydı olmayan) guest ekle → Pi `users` tablosunda guest satırı oluşmalı;
presence event'i geldiğinde `last_seen` yazılmalı.

### WS-2 — Etiket paritesi garantisi (R2)

- Frontend'de **tek normalize fonksiyonu** (örn. `vision-api.ts` içinde `normalizeIdentityLabel(name)`:
  `name.trim()` — case'i KORU). Hem Mac `/enroll` `label`'ı hem Pi `/users/guest` `name`'i **bu aynı
  string** olsun. Sunucu-tarafı normalize farkına güvenme.
- Onboarding: zaten `localStorage.username` (= Pi username) kullanıyor → değişiklik gerekmez, sadece teyit.
- Mac `_clean_label` (`face_repository.py:202`, sadece strip) ile Pi `User.username` üretimi **aynı kurala**
  tabi; case-sensitive eşleşme (SQLite default) korunmalı. İstenirse ileride iki tarafta da
  `COLLATE NOCASE`/lower-normalize, ama o zaman **her iki tarafta birlikte** yapılmalı.

→ **Doğrulama:** "Berkay" guest'i ekle → Mac galeri etiketi == Pi `User.username` birebir; `handle_gesture`
ve `_update_db_last_seen` sorguları eşleşmeli.

### WS-3 — Sessiz başarısızlık görünürlüğü (R3)

**Pi:** `_update_db_last_seen` ve `handle_gesture`, isim için User satırı bulunamadığında **WARNING**
loglasın (örn. `"identity event for unknown user '{name}' — no User row; parity broken?"`). Yalnızca log,
davranış değişmez; üretimde parite kopmasını görünür kılar.

→ **Doğrulama:** Pi'de olmayan bir isimle event simüle et → uyarı logu düşmeli.

### WS-4 — biometric-login kararı (yeni bulgu, AUTH kapsamı)

Migration sonrası Pi galerisi boş → `/auth/biometric-login` daima "Unknown" döner (`login/page.tsx:77`
kullanıyor). **Karar gerek:**
- (A) Özelliği **kaldır/gizle** (en küçük iş) — frontend'de yüzle giriş butonunu devre dışı bırak.
- (B) Mac'e **repoint**: yeni `POST /identify` (Mac) — yüklenen görsel → `FaceDetector`+`embed_face`+galeri
  eşleşmesi → `{name, score}`. Enrollment yolunun kuzeni (store yok). Sonra Pi `biometric-login` bu
  endpoint'i çağırır ya da frontend doğrudan Mac'e gider, dönen isimle Pi token'ı alır.
- **Öneri:** Ürün yüzle-giriş istiyorsa (B); istemiyorsa (A). Çekirdek agentic akışı **bloklamaz**, ayrı
  ele alınabilir.

### WS-5 — Atıl Pi yüz yollarını deprecate et (R4)

- `vision_service.register_face`, `/vision/identify`, `User.face_embedding` **yazımları** artık kullanım
  dışı. İşaretle/guard'la veya kaldır. **Voice** (`speaker_service`) kalır.
- **Engel:** `vision_service.recognize`'ı tamamen kaldırmak WS-4'e bağlı (biometric-login onu kullanıyor).
  Önce WS-4 kararı, sonra tam temizlik.
- `User.face_embedding` kolonunu hemen düşürme (DB migration riski); önce yazımları durdur, kolonu ölü
  bırak, ileride şema temizliği ayrı iş.

---

## Sıralama & kapsam

1. **WS-1 + WS-2** (çift repo) — agentic bütünlüğün çekirdeği; guest paritesini garanti eder. **Önce bunlar.**
2. **WS-3** (Pi) — ucuz, additive; WS-1 ile birlikte gidebilir.
3. **WS-4** (auth) — bağımsız karar; çekirdeği bloklamaz.
4. **WS-5** (temizlik) — WS-4'ten sonra tam kapanış.

## Doğrulama matrisi (kabul)

| Senaryo | Beklenen |
|---|---|
| Homeowner onboarding → enroll | Mac galeri + Pi User (signup'tan) parite ✅; greeting/gesture/last_seen çalışır |
| Guest sessiz ekleme (WS-1 sonrası) | Pi'de guest User satırı oluşur; last_seen yazılır; isim paritesi tam |
| Olmayan isimle event (WS-3) | Pi WARNING loglar |
| biometric-login (WS-4=A) | Buton gizli/uyarı; (WS-4=B) Mac identify ile çalışır |
| Atıl `/vision/identify` (WS-5) | Çağrılmıyor; deprecate işaretli |

## Kapsam dışı (açıkça)
- Guest jestlerinin ev sahibi ayarlarına bağlanıp bağlanmayacağı (ürün kararı).
- ChromaDB / konuşma hafızasının kullanıcı-namespace'lenmesi (zaten kullanıcı-bağımsız, etkilenmiyor).
- `User.face_embedding` kolonunun fiziksel şema silinmesi (ayrı DB migration işi).
