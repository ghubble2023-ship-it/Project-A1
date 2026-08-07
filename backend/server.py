from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from project_ai_core import (
    ENGINE_NAME,
    analyze_text,
    scan_photo,
    analyze_profile,
    check_phone,
    full_scan,
    AnalyzeTextParams,
    ScanPhotoParams,
    AnalyzeProfileParams,
    CheckPhoneParams,
    FullScanParams,
)


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

app = FastAPI(title="Project AI - Core Scanner API")
api_router = APIRouter(prefix="/api")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Persistence models
# ---------------------------------------------------------------------------
class ScanRecordCreate(BaseModel):
    scan_type: str  # text | photo | profile | phone | full
    input_summary: str  # a short human-readable summary (never the raw content)
    result: Dict[str, Any]
    overall_risk: str  # LOW | MEDIUM | HIGH | ELEVATED | N/A


class ScanRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scan_type: str
    input_summary: str
    result: Dict[str, Any]
    overall_risk: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Health & meta
# ---------------------------------------------------------------------------
@api_router.get("/")
async def root() -> Dict[str, Any]:
    return {"engine": ENGINE_NAME, "status": "online"}


# ---------------------------------------------------------------------------
# Scan endpoints
# ---------------------------------------------------------------------------
@api_router.post("/scan/text")
async def api_scan_text(params: AnalyzeTextParams) -> Dict[str, Any]:
    return analyze_text(params)


@api_router.post("/scan/photo")
async def api_scan_photo(params: ScanPhotoParams) -> Dict[str, Any]:
    return scan_photo(params)


@api_router.post("/scan/profile")
async def api_scan_profile(params: AnalyzeProfileParams) -> Dict[str, Any]:
    return analyze_profile(params)


@api_router.post("/scan/phone")
async def api_scan_phone(params: CheckPhoneParams) -> Dict[str, Any]:
    return check_phone(params)


@api_router.post("/scan/full")
async def api_scan_full(params: FullScanParams) -> Dict[str, Any]:
    return full_scan(params)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
@api_router.post("/scans", response_model=ScanRecord)
async def save_scan(payload: ScanRecordCreate) -> ScanRecord:
    record = ScanRecord(**payload.model_dump())
    await db.scans.insert_one(record.model_dump())
    return record


@api_router.get("/scans", response_model=List[ScanRecord])
async def list_scans(limit: int = 100) -> List[ScanRecord]:
    docs = await db.scans.find({}, {"_id": 0}).sort("created_at", -1).to_list(limit)
    return [ScanRecord(**d) for d in docs]


@api_router.get("/scans/{scan_id}", response_model=ScanRecord)
async def get_scan(scan_id: str) -> ScanRecord:
    doc = await db.scans.find_one({"id": scan_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Scan not found")
    return ScanRecord(**doc)


@api_router.delete("/scans/{scan_id}")
async def delete_scan(scan_id: str) -> Dict[str, Any]:
    result = await db.scans.delete_one({"id": scan_id})
    return {"deleted": result.deleted_count}


@api_router.delete("/scans")
async def clear_scans() -> Dict[str, Any]:
    result = await db.scans.delete_many({})
    return {"deleted": result.deleted_count}


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@app.on_event("shutdown")
async def shutdown_db_client() -> None:
    client.close()
