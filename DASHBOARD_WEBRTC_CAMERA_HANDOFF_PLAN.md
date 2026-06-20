# Handoff — Dashboard'a Kamera Görüntüsünü WebRTC ile Akıtma

> **Hedef:** `proactive-home-agent` dashboard'unda canlı kamera görüntüsü **WebRTC** ile düşük
> gecikmeli aksın. Kaynak: Mac'te çalışan **go2rtc** (Tapo C225 proxy'si).
>
> **Rol:** Codex implement eder. Çoğu iş frontend + Mac env; video_Process Python kodu değişmez.

---

## Mevcut durum (tespit)

`frontend/src/components/CameraFeed.tsx` zaten bir iframe ile yayın çekmeye çalışıyor ama **3 hatası var**:

```tsx
src="http://100.105.136.5:1984/stream.html?src=living_room_cam&mode=mse"
```
1. **Yanlış host:** `100.105.136.5` = **Pi**'nin IP'si. go2rtc **Mac**'te çalışıyor → doğru host
   `100.90.235.67` (Mac Tailscale IP).
2. **Yanlış stream adı:** `living_room_cam` diye bir stream **yok**. go2rtc.yaml'de `living_room_sd`
   (CV pipeline kullanıyor) ve `living_room_hd` (dashboard için ayrılmış) var.
3. **Mod MSE, WebRTC değil:** `mode=mse`. İstenen WebRTC → `mode=webrtc`.

Ayrıca `isOnline` sadece iframe `onLoad`'una bağlı (hata sayfasında da tetiklenir) → "LIVE" rozeti
yanıltıcı. URL hardcoded (env yok).

## Mimari

```
Tapo C225 ──RTSP──> go2rtc (Mac, Docker)  ──┬── living_room_sd (stream2) → CV pipeline (mevcut)
                     :1984 API / :8555 RTC  └── living_room_hd (stream1) → Dashboard (WebRTC)  ← YENİ
                                                          │
                              Browser (dashboard) ──WebRTC── Mac:1984/8555 (Tailscale)
```
- go2rtc **WebRTC'yi zaten sunuyor** (`webrtc: listen :8555`, `api: :1984 origin "*"`). Yeni Python
  kodu **gerekmiyor**.
- Bu **ham kamera** görüntüsüdür; CV overlay'leri (yüz kutusu, fall bölgesi) bu akışta yok — onlar
  ayrı vision WS'te (`:5003`) JSON olarak akıyor. Overlay'i video üstüne çizmek **ayrı/kapsam dışı**.

---

## Bölüm 1 — Mac tarafı (video_Process): go2rtc WebRTC erişilebilirliği

**Python kodu değişmez.** Sadece go2rtc'nin WebRTC ICE candidate'ını ağ üzerinden ulaşılabilir
yapmak gerekiyor (go2rtc Docker'da; NAT yüzünden candidate'ı manuel vermezsen tarayıcı medyayı alamaz).

1. Mac `.env`'e ekle (go2rtc.yaml `candidates: ${GO2RTC_WEBRTC_CANDIDATE}` bunu okuyor):
   ```
   GO2RTC_WEBRTC_CANDIDATE=100.90.235.67:8555
   ```
   > Dashboard'a hep aynı LAN'dan da bakılacaksa ikinci bir candidate (LAN IP) eklenebilir; Tailscale
   > üzerinden erişim için yukarıdaki yeterli. 8555 hem TCP hem UDP map'li (docker-compose) → kısıtlı
   > ağlarda TCP candidate fallback çalışır.

2. go2rtc'nin ayakta ve `living_room_hd`'nin yayınlanabildiğini doğrula (Mac'te):
   ```bash
   curl -s http://localhost:1984/api/streams | jq 'keys'        # living_room_sd, living_room_hd görünmeli
   # Tarayıcı testi: http://100.90.235.67:1984/stream.html?src=living_room_hd&mode=webrtc
   ```
   > Tapo stream1 (HD) + stream2 (SD) ayrı substream; ikisi eşzamanlı sorun değil. go2rtc HD'yi
   > yalnız bir client bağlanınca çeker (on-demand).

3. **Güvenlik notu:** go2rtc API auth'suz + `origin:"*"`. Mac:1984'e ulaşan herkes kamerayı görür.
   Tailscale güven sınırında kabul edilebilir (D2-A ile tutarlı); tailnet dışına açılacaksa ayrı
   auth kararı gerekir (kapsam dışı).

---

## Bölüm 2 — Frontend (proactive-home-agent): CameraFeed WebRTC'ye geçir

### Env
`frontend/.env.local` (veya build env) ekle:
```
NEXT_PUBLIC_GO2RTC_URL=http://100.90.235.67:1984
NEXT_PUBLIC_GO2RTC_STREAM=living_room_hd
```
Mevcut `NEXT_PUBLIC_*` deseniyle tutarlı (build-time; değişince yeniden build/dev gerekir).

### Seçenek A — Hızlı/sağlam (ÖNERİLEN): iframe'i doğru URL + WebRTC moduna çevir
`CameraFeed.tsx` içindeki `<iframe src=...>`'i değiştir:
```tsx
const GO2RTC = process.env.NEXT_PUBLIC_GO2RTC_URL || "http://100.90.235.67:1984";
const STREAM = process.env.NEXT_PUBLIC_GO2RTC_STREAM || "living_room_hd";
...
<iframe
  src={`${GO2RTC}/stream.html?src=${STREAM}&mode=webrtc`}
  ... (mevcut className/style/allow aynı)
/>
```
- **Artı:** Tek satırlık öz; go2rtc'nin kendi oynatıcısı WebRTC el sıkışmasını + (gerekirse) fallback'i
  hallediyor. Tüm overlay UI'si aynen kalır.
- **Eksi:** iframe → gerçek bağlantı durumu bilinemez; `isOnline`/"LIVE" rozeti `onLoad` ile yaklaşık.

### Seçenek B — Native WebRTC (kontrol + gerçek durum, iframe yok)
iframe yerine native `<video>` + `RTCPeerConnection`; go2rtc WebRTC API'siyle offer/answer:
```tsx
"use client";
import { useEffect, useRef, useState } from "react";

const GO2RTC = process.env.NEXT_PUBLIC_GO2RTC_URL || "http://100.90.235.67:1984";
const STREAM = process.env.NEXT_PUBLIC_GO2RTC_STREAM || "living_room_hd";

// ... component içinde:
const videoRef = useRef<HTMLVideoElement>(null);
useEffect(() => {
  const pc = new RTCPeerConnection({ bundlePolicy: "max-bundle" });
  pc.addTransceiver("video", { direction: "recvonly" });
  pc.addTransceiver("audio", { direction: "recvonly" }); // istemezsen kaldır
  pc.ontrack = (e) => { if (videoRef.current) videoRef.current.srcObject = e.streams[0]; };
  pc.onconnectionstatechange = () => setIsOnline(pc.connectionState === "connected");

  (async () => {
    await pc.setLocalDescription(await pc.createOffer());
    // ICE toplanmasını bekle (basit yol):
    await new Promise<void>((r) => {
      if (pc.iceGatheringState === "complete") return r();
      pc.onicegatheringstatechange = () =>
        pc.iceGatheringState === "complete" && r();
    });
    const resp = await fetch(`${GO2RTC}/api/webrtc?src=${STREAM}`, {
      method: "POST",
      headers: { "Content-Type": "application/sdp" },
      body: pc.localDescription!.sdp,
    });
    await pc.setRemoteDescription({ type: "answer", sdp: await resp.text() });
  })();

  return () => pc.close();
}, []);
// <video ref={videoRef} autoPlay muted playsInline className="w-full h-full object-cover" />
```
- **Artı:** Gerçek `connectionState` → doğru LIVE rozeti; native video → fullscreen/overlay video'nun
  kendisinde; iframe yok.
- **Eksi:** ~60 satır WebRTC; `<video>` **`muted autoPlay playsInline`** olmalı (yoksa tarayıcı
  autoplay'i bloklar).
- **DİKKAT — API şeklini doğrula:** go2rtc sürümleri `/api/webrtc` gövdesinde farklılık gösterebilir
  (raw SDP vs JSON `{type,sdp}`). Implementasyondan önce çalışan go2rtc'de teyit et:
  ```bash
  # raw-SDP kabul ediyor mu? (boş/again dönebilir ama 4xx vermemeli)
  curl -s -X POST "http://100.90.235.67:1984/api/webrtc?src=living_room_hd" \
       -H "Content-Type: application/sdp" --data-binary @offer.sdp -i | head
  ```
  JSON bekliyorsa gövdeyi `{ "type":"offer","sdp":pc.localDescription.sdp }` + `application/json`
  yap ve cevabı `await resp.json()` ile al.

> **Öneri:** Önce **Seçenek A** ile uçtan uca WebRTC akışını doğrula (en düşük risk). Gerçek bağlantı
> durumu/iframe'siz native istenirse **Seçenek B**'ye geç. İkisi de aynı env'i kullanır.

---

## Doğrulama (uçtan uca)

1. Mac: `GO2RTC_WEBRTC_CANDIDATE` set, go2rtc restart; `http://100.90.235.67:1984/stream.html?src=living_room_hd&mode=webrtc` tarayıcıda **WebRTC** olarak oynamalı (go2rtc info panelinde "webrtc").
2. Frontend env set + build/dev; dashboard'da room sayfası (`/room/[roomId]`) → CameraFeed canlı görüntü.
3. Tarayıcı DevTools → Network'te `api/webrtc` (B) veya iframe WebRTC; `chrome://webrtc-internals`'ta
   `connectionState=connected`, video track akışta.
4. Tailscale üzerinden başka cihazdan da test et (candidate doğru mu).

---

## Riskler / dikkat
- **Candidate olmazsa siyah ekran:** Docker NAT yüzünden `GO2RTC_WEBRTC_CANDIDATE` zorunlu; en sık hata bu.
- **Mixed content:** Dashboard `https` ise `http://...go2rtc` bloklanır. Şu an her şey `http` (LAN/
  Tailscale) → sorun yok; ileride HTTPS'e geçilirse go2rtc'ye de TLS/proxy gerekir.
- **Autoplay:** Seçenek B'de `<video muted autoPlay playsInline>` şart.
- **Ham görüntü:** CV overlay yok (ayrı iş). İstenirse vision WS (`:5003`) state'i canvas ile video
  üstüne çizilir — ayrı handoff.
- **Kapsam:** Python/CV tarafı değişmiyor; sadece Mac `.env` + frontend. Tek kamera/tek stream
  (multi-room eşleme ileride).
