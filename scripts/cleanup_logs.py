import os
import shutil
import glob
from pathlib import Path

def cleanup_logs_and_screenshots():
    """
    Cleans up all log files and screenshots from the project's data/logs directory.
    """
    project_root = Path(__file__).parent.parent
    logs_dir = project_root / "data" / "logs"
    screenshots_dir = logs_dir / "screenshots"

    print("🧹 Temizlik işlemi başlatılıyor...\n")

    # 1. Klasördeki Log Dosyalarını (events.log, events.jsonl vb.) Silme
    if logs_dir.exists() and logs_dir.is_dir():
        log_files = []
        log_files.extend(logs_dir.glob("*.log"))
        log_files.extend(logs_dir.glob("*.jsonl"))

        if log_files:
            for file_path in log_files:
                try:
                    file_path.unlink()
                    print(f"✅ Başarıyla Silindi (Log Dosyası): {file_path.name}")
                except Exception as e:
                    print(f"❌ Silinemedi (Log Dosyası): {file_path.name} - Hata: {e}")
        else:
            print("ℹ️ Silinecek log dosyası bulunamadı.")
    else:
        print(f"⚠️ Log klasörü bulunamadı: {logs_dir}")

    # 2. Screenshots Klasörünün İçeriğini Temizleme
    if screenshots_dir.exists() and screenshots_dir.is_dir():
        screenshot_files = list(screenshots_dir.glob("*.[pj][np][g]")) # .png, .jpg vb.
        
        if screenshot_files:
            for file_path in screenshot_files:
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"❌ Silinemedi (Screenshot): {file_path.name} - Hata: {e}")
            print(f"✅ Başarıyla Silindi: {len(screenshot_files)} adet screenshot dosyası.")
        else:
            print("ℹ️ Silinecek screenshot bulunamadı.")
    else:
        print(f"⚠️ Screenshot klasörü bulunamadı: {screenshots_dir}")

    print("\n✨ Temizlik işlemi tamamlandı!")

if __name__ == "__main__":
    cleanup_logs_and_screenshots()
