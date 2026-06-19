import sqlite3

DB_PATH = "aeroquery.db"


def get_connection():
    
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT,
            confidence REAL,
            x1 REAL,
            y1 REAL,
            x2 REAL,
            y2 REAL,
            detected_at TEXT
        )
    """)

    conn.commit()
    conn.close()