import os
import json
from typing import Optional, List, Dict, Any
from datetime import datetime


class DBEngine:
    """Database engine supporting SQLite (default) and optional PostgreSQL via SQLAlchemy.

    Usage:
      DBEngine(db_path='results/inspection_history.db')
      DBEngine(db_url='postgresql://user:pass@host:5432/dbname')
    """

    def __init__(self, db_path: str = "results/inspection_history.db", db_url: Optional[str] = None):
        self._use_sqlalchemy = False
        self.db_url = db_url
        if db_url:
            # try to import SQLAlchemy and use it for Postgres or other URLs
            try:
                from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, Float, ForeignKey
                from sqlalchemy import select
                self._sa = {
                    'create_engine': create_engine,
                    'MetaData': MetaData,
                    'Table': Table,
                    'Column': Column,
                    'Integer': Integer,
                    'String': String,
                    'Text': Text,
                    'Float': Float,
                    'ForeignKey': ForeignKey,
                    'select': select,
                }
                self.engine = create_engine(db_url, future=True)
                self.metadata = MetaData()
                inspections = Table(
                    'inspections', self.metadata,
                    Column('id', Integer, primary_key=True, autoincrement=True),
                    Column('ts', String(64)),
                    Column('part_id', String(128)),
                    Column('status', String(32)),
                    Column('metrics', Text),
                )
                defects = Table(
                    'defects', self.metadata,
                    Column('id', Integer, primary_key=True, autoincrement=True),
                    Column('inspection_id', Integer),
                    Column('defect_type', String(64)),
                    Column('confidence', Float),
                    Column('bbox', Text),
                )
                self._tables = {'inspections': inspections, 'defects': defects}
                self.metadata.create_all(self.engine)
                self._use_sqlalchemy = True
                return
            except Exception:
                # fallback to SQLite file mode
                self._use_sqlalchemy = False

        # default sqlite behavior
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) != "" else ".", exist_ok=True)
        import sqlite3

        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self):
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                part_id TEXT,
                status TEXT,
                metrics TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS defects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id INTEGER,
                defect_type TEXT,
                confidence REAL,
                bbox TEXT,
                FOREIGN KEY(inspection_id) REFERENCES inspections(id) ON DELETE CASCADE
            )
            """
        )
        self._conn.commit()

    def log_inspection(self, metrics: Dict[str, Any], defects: List[Dict[str, Any]], part_id: Optional[str] = None) -> int:
        if self._use_sqlalchemy:
            ins_tbl = self._tables['inspections']
            def_tbl = self._tables['defects']
            from sqlalchemy import insert
            with self.engine.begin() as conn:
                res = conn.execute(insert(ins_tbl).values(ts=datetime.utcnow().isoformat(), part_id=part_id, status=metrics.get('status', 'UNKNOWN'), metrics=json.dumps(metrics)))
                insp_id = int(res.inserted_primary_key[0]) if res.inserted_primary_key else None
                for d in defects:
                    bbox = d.get('bbox', None)
                    bbox_json = json.dumps(bbox) if bbox is not None else None
                    conn.execute(insert(def_tbl).values(inspection_id=insp_id, defect_type=d.get('type', 'unknown'), confidence=float(d.get('confidence', 1.0)), bbox=bbox_json))
                return insp_id

        # sqlite path
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO inspections (ts, part_id, status, metrics) VALUES (?, ?, ?, ?)",
            (
                datetime.utcnow().isoformat(),
                part_id,
                metrics.get("status", "UNKNOWN"),
                json.dumps(metrics),
            ),
        )
        insp_id = cur.lastrowid
        for d in defects:
            bbox = d.get("bbox", None)
            bbox_json = json.dumps(bbox) if bbox is not None else None
            cur.execute(
                "INSERT INTO defects (inspection_id, defect_type, confidence, bbox) VALUES (?, ?, ?, ?)",
                (insp_id, d.get("type", "unknown"), float(d.get("confidence", 1.0)), bbox_json),
            )
        self._conn.commit()
        return insp_id

    def query_inspections(self, limit: int = 100) -> List[Dict[str, Any]]:
        if self._use_sqlalchemy:
            ins_tbl = self._tables['inspections']
            def_tbl = self._tables['defects']
            with self.engine.begin() as conn:
                from sqlalchemy import select
                q = select(ins_tbl).order_by(ins_tbl.c.id.desc()).limit(limit)
                rows = conn.execute(q).fetchall()
                out = []
                for r in rows:
                    dq = select(def_tbl.c.defect_type, def_tbl.c.confidence, def_tbl.c.bbox).where(def_tbl.c.inspection_id == r.id)
                    drows = conn.execute(dq).fetchall()
                    defects = []
                    for d in drows:
                        bbox = json.loads(d.bbox) if d.bbox else None
                        defects.append({"type": d.defect_type, "confidence": float(d.confidence), "bbox": bbox})
                    out.append({"id": int(r.id), "ts": r.ts, "part_id": r.part_id, "status": r.status, "metrics": json.loads(r.metrics or "{}"), "defects": defects})
                return out

        cur = self._conn.cursor()
        cur.execute("SELECT * FROM inspections ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        out = []
        for r in rows:
            cur2 = self._conn.cursor()
            cur2.execute("SELECT defect_type, confidence, bbox FROM defects WHERE inspection_id = ?", (r["id"],))
            drows = cur2.fetchall()
            defects = []
            for d in drows:
                bbox = json.loads(d["bbox"]) if d["bbox"] else None
                defects.append({"type": d["defect_type"], "confidence": d["confidence"], "bbox": bbox})

            out.append({
                "id": r["id"],
                "ts": r["ts"],
                "part_id": r["part_id"],
                "status": r["status"],
                "metrics": json.loads(r["metrics"] or "{}"),
                "defects": defects,
            })
        return out

    def close(self):
        try:
            if self._use_sqlalchemy:
                try:
                    self.engine.dispose()
                except Exception:
                    pass
            else:
                self._conn.close()
        except Exception:
            pass
