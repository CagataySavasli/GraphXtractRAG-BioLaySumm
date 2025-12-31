import os
import glob
import json
import sqlite3
import gc
import shutil
import pandas as pd
import numpy as np
from typing import List

# ==========================================
# KONFIGÜRASYON
# ==========================================
SEPERATED_PATH = "./dataset/processed/plos/sep_val"

# ÖNEMLİ: Geniş kapasiteli diskinizdeki yolu buraya tekrar girin
LARGE_STORAGE_PATH = "/home/cagatay/Desktop/CS/Projects/GraphXtractRAG-BioLaySumm"
DB_FILENAME = "dataset/plos.db"

DATABASE_FULL_PATH = os.path.join(LARGE_STORAGE_PATH, DB_FILENAME)
TABLE_NAME = "validation"

# Geçici dosyaları kesinlikle buraya yönlendir
CUSTOM_TEMP_DIR = os.path.join(LARGE_STORAGE_PATH, "sqlite_tmp_garbage")
os.makedirs(CUSTOM_TEMP_DIR, exist_ok=True)

# Hem Python hem SQLite için temp ayarlarını ez
os.environ['TMPDIR'] = CUSTOM_TEMP_DIR
os.environ['SQLITE_TMPDIR'] = CUSTOM_TEMP_DIR


# ==========================================
# YARDIMCI SINIFLAR
# ==========================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# ==========================================
# YARDIMCI FONKSİYONLAR
# ==========================================
def get_sorted_files(directory_path: str) -> List[str]:
    files = glob.glob(os.path.join(directory_path, "*.parquet"))
    try:
        files.sort(key=lambda f: int(os.path.basename(f).split('_')[1]))
    except Exception:
        files.sort()
    return files


def serialize_columns(df: pd.DataFrame) -> pd.DataFrame:
    target_cols = [
        'sections', 'headings', 'abstract', 'summary', 'keywords', 'toc',
        'title_embeddings', 'abstract_embeddings', 'section_embeddings'
    ]
    cols_to_process = [c for c in target_cols if c in df.columns]

    for col in cols_to_process:
        df[col] = df[col].apply(
            lambda x: json.dumps(x, cls=NumpyEncoder) if x is not None else None
        )
    return df


def configure_sqlite(conn):
    """SQLite performans ve dosya boyutu ayarları"""
    cursor = conn.cursor()
    # Sayfa boyutunu artır (Büyük veriler için daha verimli)
    cursor.execute("PRAGMA page_size = 32768;")
    # Geçici dosyaları bellekte değil diskte tut (RAM şişmesin)
    cursor.execute(f"PRAGMA temp_store = 1;")  # 1 = File, 2 = Memory
    # Journal modu: DELETE (Disk dolunca en güvenlisi budur, WAL çok yer kaplayabilir)
    cursor.execute("PRAGMA journal_mode = DELETE;")
    cursor.execute("PRAGMA cache_size = -200000;")  # Yaklaşık 200MB RAM cache limiti
    cursor.close()


def process_single_file(file_path: str, db_path: str, table_name: str):
    filename = os.path.basename(file_path)
    df = None

    try:
        # Parquet oku
        df = pd.read_parquet(file_path)
        df = serialize_columns(df)

        with sqlite3.connect(db_path) as conn:
            configure_sqlite(conn)

            # --- KRİTİK DEĞİŞİKLİK: CHUNKSIZE ---
            # 500 satırı tek seferde değil, 50'şer 50'şer yaz.
            # Bu, "Disk Full" hatasını önleyen en önemli adımdır.
            # Transaction log boyutu asla 50 satırı geçmez.
            df.to_sql(
                name=table_name,
                con=conn,
                if_exists='append',
                index=False,
                chunksize=50,  # <-- SİHİRLİ DOKUNUŞ
                method='multi'  # Daha hızlı insert sağlar
            )

        print("✅ OK")

    except Exception as e:
        print(f"\n❌ HATA: {filename} --> {e}")
        # Hata olsa bile devam etmeyi dene, belki bir sonraki dosya sığar

    finally:
        if df is not None:
            del df
        gc.collect()


# ==========================================
# MAIN
# ==========================================
def main():
    print(f"--- İşlem Başlıyor (Chunked Insert Mode) ---")
    print(f"Veritabanı: {DATABASE_FULL_PATH}")
    print(f"Geçici Dosya Yolu: {CUSTOM_TEMP_DIR}")

    # Disk kontrolü (Sadece bilgilendirme amaçlı, hata fırlatmasın)
    try:
        total, used, free = shutil.disk_usage(LARGE_STORAGE_PATH)
        print(f"Başlangıç Boş Alan: {free // (1024 ** 3)} GB")
    except:
        pass

    files = get_sorted_files(SEPERATED_PATH)

    # Daha önce işlenen dosyaları atlamak isterseniz basit bir filtreleme yapılabilir.
    # Şimdilik hata veren dosyadan sonrasını manuel de silebilirsiniz veya
    # kod kaldığı yerden devam etsin diye hepsini tarar.

    total = len(files)

    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{total}] İşleniyor: {os.path.basename(file_path)} ...", end=" ", flush=True)
        process_single_file(file_path, DATABASE_FULL_PATH, TABLE_NAME)

    # İşlem bitince tmp klasörünü temizlemeyi deneyelim
    try:
        shutil.rmtree(CUSTOM_TEMP_DIR)
        print("\nGeçici dosyalar temizlendi.")
    except:
        pass

    print(f"--- Bitti ---")


if __name__ == "__main__":
    main()