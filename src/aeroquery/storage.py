import sqlite3
from datetime import datetime

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


def save_detection(class_name, confidence, bbox):
    conn = get_connection()
    cursor = conn.cursor()


    x1, y1, x2, y2 = bbox
    detected_at = datetime.now().isoformat()

    cursor.execute(
        "INSERT INTO detections (class_name, confidence, x1, y1, x2, y2, detected_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (class_name, confidence, x1, y1, x2, y2, detected_at)
    )

    conn.commit()
    conn.close()