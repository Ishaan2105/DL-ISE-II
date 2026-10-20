"""
FaceForge AI - SQLite Persistence Layer
Manages generation records, gallery states, and dashboard statistics.
"""

import sqlite3
import logging
from contextlib import contextmanager
from typing import Dict, Any, Optional
from backend.config import DATABASE_PATH

logger = logging.getLogger(__name__)

_DB_INITIALIZED = False

@contextmanager
def get_db():
    """Context manager for SQLite connections with row_factory as dict."""
    global _DB_INITIALIZED
    conn = sqlite3.connect(DATABASE_PATH, timeout=15.0)
    conn.row_factory = sqlite3.Row
    if not _DB_INITIALIZED:
        _DB_INITIALIZED = True
        try:
            init_db_schema(conn)
        except Exception as e:
            logger.error(f"Error initializing schema: {e}")
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database transaction error: {e}")
        raise
    finally:
        conn.close()

def init_db_schema(conn):
    """Create generations table and indexes on connection."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS generations (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            seed INTEGER NOT NULL,
            batch_index INTEGER DEFAULT 0,
            model_name TEXT NOT NULL,
            resolution INTEGER NOT NULL,
            duration_ms REAL NOT NULL,
            device TEXT NOT NULL,
            saved_to_gallery INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_gen_created_at ON generations(created_at DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_gen_saved ON generations(saved_to_gallery)")

def init_db():
    """Explicitly initialize schema."""
    with sqlite3.connect(DATABASE_PATH, timeout=15.0) as conn:
        init_db_schema(conn)
    logger.info("Database initialized successfully at %s", DATABASE_PATH)

def insert_generation(
    gen_id: str,
    filename: str,
    seed: int,
    batch_index: int,
    model_name: str,
    resolution: int,
    duration_ms: float,
    device: str,
    saved_to_gallery: bool = False
) -> Dict[str, Any]:
    """Insert a new generation record."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO generations 
            (id, filename, seed, batch_index, model_name, resolution, duration_ms, device, saved_to_gallery, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (gen_id, filename, seed, batch_index, model_name, resolution, duration_ms, device, 1 if saved_to_gallery else 0))
    return get_generation_by_id(gen_id)

def get_generation_by_id(gen_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single generation record by its ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM generations WHERE id = ?", (gen_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
    return None

def list_generations(
    limit: int = 24,
    offset: int = 0,
    saved_only: bool = False,
    sort_order: str = "newest",
    search: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """Query generations with pagination, search, and filtering."""
    query = "SELECT * FROM generations WHERE 1=1"
    count_query = "SELECT COUNT(*) as total FROM generations WHERE 1=1"
    params = []
    count_params = []
    
    if saved_only:
        query += " AND saved_to_gallery = 1"
        count_query += " AND saved_to_gallery = 1"
        
    if model and model != "all":
        query += " AND model_name = ?"
        count_query += " AND model_name = ?"
        params.append(model)
        count_params.append(model)
        
    if search:
        search_param = f"%{search.strip()}%"
        query += " AND (id LIKE ? OR filename LIKE ? OR CAST(seed AS TEXT) LIKE ?)"
        count_query += " AND (id LIKE ? OR filename LIKE ? OR CAST(seed AS TEXT) LIKE ?)"
        params.extend([search_param, search_param, search_param])
        count_params.extend([search_param, search_param, search_param])
        
    order_clause = " ORDER BY created_at DESC" if sort_order == "newest" else " ORDER BY created_at ASC"
    query += order_clause
    
    query += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(count_query, count_params)
        total = cursor.fetchone()["total"]
        
        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        
    return {
        "items": rows,
        "total": total,
        "limit": limit,
        "offset": offset
    }

def set_saved_status(gen_id: str, saved: bool) -> bool:
    """Update saved_to_gallery status for an image."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE generations SET saved_to_gallery = ? WHERE id = ?", (1 if saved else 0, gen_id))
        return cursor.rowcount > 0

def delete_generation_record(gen_id: str) -> Optional[str]:
    """Delete record from database and return the filename for disk cleanup."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filename FROM generations WHERE id = ?", (gen_id,))
        row = cursor.fetchone()
        if not row:
            return None
        filename = row["filename"]
        cursor.execute("DELETE FROM generations WHERE id = ?", (gen_id,))
        return filename

def get_stats() -> Dict[str, Any]:
    """Compute real dashboard statistics from the database."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total_generations FROM generations")
        total_gen = cursor.fetchone()["total_generations"]
        
        cursor.execute("SELECT COUNT(*) as saved_count FROM generations WHERE saved_to_gallery = 1")
        saved_count = cursor.fetchone()["saved_count"]
        
        cursor.execute("SELECT AVG(duration_ms) as avg_duration FROM generations")
        avg_row = cursor.fetchone()
        avg_duration = round(avg_row["avg_duration"], 2) if avg_row and avg_row["avg_duration"] else 0.0
        
        cursor.execute("SELECT * FROM generations ORDER BY created_at DESC LIMIT 6")
        recent = [dict(row) for row in cursor.fetchall()]
        
    return {
        "total_generations": total_gen,
        "saved_count": saved_count,
        "avg_duration_ms": avg_duration,
        "recent_generations": recent
    }
