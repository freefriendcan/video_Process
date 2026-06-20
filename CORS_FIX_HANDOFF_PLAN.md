# Handoff — video_process Vision API CORS Fix

> Hedef: tarayıcıdaki `proactive-home-agent` dashboard'unun, Mac vision API'sine (`:8800`) **cross-origin**
> istek atabilmesi. Şu an `api/app.py`'de CORS middleware YOK → tüm tarayıcı çağrıları (özellikle
> `DELETE` preflight) başarısız. Bu **yalnızca video_process** tarafında ufak bir ekleme. Frontend'e
> dokunma.
>
> Rol: Codex implement eder. Aşağıdaki adımlar ve doğrulama bağlayıcı.

---

## Kapsam (sadece bunlar)

1. `config.py` — env-driven CORS origin listesi ekle.
2. `api/app.py` — `CORSMiddleware` ekle (default'lu yeni param ile, mevcut çağrıları kırmadan).
3. `main.py` — `create_app`'e origin listesini geçir.
4. `tests/test_enrollment_api.py` — bir CORS testi ekle.

**Kapsam dışı:** auth eklemek, frontend, guest-record kararı (ayrı iş), başka endpoint.

---

## Tasarım kararları (kilit)

- **Origin politikası:** env-configurable, **default `*`**. Gerekçe: Mac API zaten auth'suz (D2-A, LAN/
  Tailscale güven sınırı) ve tarayıcı Mac'e **credential göndermiyor** (cookie/Authorization yok — JWT
  yalnızca Pi'ye gidiyor). Dolayısıyla `*` güvenlik kaybı yaratmaz ve dashboard hangi origin'den
  sunulursa sunulsun (dev `localhost:3000`, prod Pi IP, vb.) çalışır. Daraltmak isteyen `VISION_API_CORS_ORIGINS`
  ile virgüllü liste verir.
- **`allow_credentials=False`.** Credential kullanılmıyor; ayrıca CORS spec'i `allow_origins=["*"]` ile
  `allow_credentials=True`'yu yasaklar. İkisi tutarlı kalsın.
- **`allow_methods`:** en az `GET, POST, DELETE, OPTIONS` (kolayca `["*"]`). `DELETE /enrolled/{label}`
  non-simple metod → preflight tetikler; middleware OPTIONS'a otomatik yanıt verir.
- **`allow_headers`:** `["*"]` (multipart POST için `Content-Type` zaten safelisted; `*` ileride sorun
  çıkarmaz).

---

## Adım adım

### 1) `config.py`
`PipelineConfig`'e alan ekle (mevcut "Enrollment REST API" bloğunun yanına, `vision_api_port`'un altına):

```python
    vision_api_cors_origins: tuple[str, ...] = ("*",)
```

`__post_init__` içinde env parse et (diğer `vision_api_*` satırlarının yanına). Virgülle ayrılmış string'i
tuple'a çevir; boş değerleri ele; hiç verilmezse default kalsın:

```python
        cors_env = os.environ.get("VISION_API_CORS_ORIGINS")
        if cors_env is not None and cors_env.strip():
            self.vision_api_cors_origins = tuple(
                origin.strip() for origin in cors_env.split(",") if origin.strip()
            )
```

> Not: Tek başına `*` ya da virgüllü gerçek origin listesi (`http://localhost:3000,http://<pi-ip>:3000`)
> desteklenmeli.

### 2) `api/app.py`
`CORSMiddleware`'i ekle. `create_app`'e **default'lu** `cors_origins` parametresi ekle ki mevcut iki test
çağrısı (`create_app(enrollment_service=..., tracking_service=...)`) değişmeden çalışsın:

```python
from collections.abc import Sequence
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def create_app(
    enrollment_service: EnrollmentService,
    tracking_service: TrackingService,
    cors_origins: Sequence[str] = ("*",),
) -> FastAPI:
    app = FastAPI(title="video_process vision API", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(build_enrollment_router(enrollment_service))
    app.include_router(build_tracking_router(tracking_service))
    return app
```

### 3) `main.py`
`create_app` çağrısına origin listesini geçir (mevcut `VisionAPIServer(cfg, create_app(...))` bloğunda):

```python
            create_app(
                enrollment_service=self._enrollment_service,
                tracking_service=self._tracking_service,
                cors_origins=cfg.vision_api_cors_origins,
            ),
```

### 4) `tests/test_enrollment_api.py`
Bir CORS testi ekle. Mevcut `FakeEnrollmentService` / `FakeTrackingService` ile app kur; iki şeyi doğrula:

- **Preflight (`DELETE`):** `OPTIONS /enrolled/Ada` + header `Origin: http://localhost:3000`,
  `Access-Control-Request-Method: DELETE` → status 200 ve yanıt header'ında
  `access-control-allow-origin` mevcut.
- **Basit istek ACAO:** `GET /enrolled` + `Origin: http://localhost:3000` → yanıt header'ında
  `access-control-allow-origin` mevcut.

Default `*` origin ile çalıştığını kontrol et (create_app'e cors_origins geçmeden de ACAO dönmeli).

---

## Doğrulama (hepsi geçmeli)

```bash
uv run ruff check api config.py main.py tests/test_enrollment_api.py
uv run mypy api config.py main.py tests/test_enrollment_api.py
uv run pytest -q                       # mevcut 64 + yeni CORS testi
uv run python -c "import api.app, main; print('ok')"
git diff --check
```

Beklenen: ruff/mypy temiz, pytest **65+ passed** (mevcut 64 kırılmadan + yeni test), import ok.

---

## Riskler / dikkat

- **Mevcut testleri kırma:** `cors_origins` MUTLAKA default'lu olmalı (`("*",)`); aksi halde `create_app`'in
  iki test çağrısı patlar.
- **`*` + credentials çelişkisi:** `allow_credentials` False kalsın.
- **CORS ≠ auth:** Bu tarayıcı-tarafı bir kolaylık; Mac API hâlâ auth'suz (D2-A). Eğer ileride Mac
  API'sini güven sınırı dışına açacaksan ayrı bir auth/secret kararı gerekir (bu işin kapsamı değil).
- **Surgical kal:** Sadece 4 dosya; başka endpoint/davranış değiştirme.
