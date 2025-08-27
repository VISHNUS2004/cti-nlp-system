from fastapi import FastAPI, UploadFile, File, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from scripts.ingest_all_sources import main as ingest_all_sources_main

import asyncio
import pandas as pd
import tempfile
import os
import json
import logging
from datetime import datetime
from typing import List, Optional

from backend.threat_ner import extract_threat_entities
from backend.classifier import classify_threat
from backend.severity_predictor import predict_severity

# Enhanced analyzer import
try:
    from backend.simple_enhanced_analyzer import (
        analyze_threat_comprehensive, 
        get_model_info,
        initialize_analyzer
    )
    ENHANCED_ANALYZER_AVAILABLE = True
    # Initialize the enhanced analyzer
    initialize_analyzer()
except ImportError as e:
    logger.warning(f"Enhanced analyzer not available: {e}")
    ENHANCED_ANALYZER_AVAILABLE = False

# Database imports
from database.database import get_db_session, get_redis_client, get_database_config
from database.services import ThreatIntelService, EntityService, AnalyticsService
from database.models import DataSource, ThreatCategory, SeverityLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CTI-NLP API",
    description="AI-Powered Cyber Threat Intelligence System",
    version="1.0.0"
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Test database connection
        db_config = get_database_config()
        db = next(db_config.get_db())
        db.execute("SELECT 1")
        db.close()
        
        # Test Redis connection
        redis_client = get_redis_client()
        redis_client.ping()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "database": "connected",
                "redis": "connected",
                "api": "running"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

# -------------------
# Background ingestion
# -------------------
async def ingestion_loop(interval_minutes=10):
    while True:
        try:
            print("[INFO] Running real-time ingestion...")
            ingest_all_sources_main()
        except Exception as e:
            print(f"[ERROR] Ingestion loop failed: {e}")
        await asyncio.sleep(interval_minutes * 60)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(ingestion_loop(interval_minutes=10))

# -------------------
# CORS Middleware
# -------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="dashboard/templates")

# -------------------
# Root endpoint with ingestion status
# -------------------
@app.get("/")
async def root():
    status_file = os.path.join("data", "last_ingestion.json")
    if os.path.exists(status_file):
        with open(status_file, "r", encoding="utf-8") as f:
            status_data = json.load(f)
    else:
        status_data = {
            "last_run": None,
            "summary": {},
            "total_records": 0,
            "errors": {}
        }
    return {
        "status": "CTI-NLP backend running with real-time ingestion",
        "ingestion_status": status_data
    }


@app.get("/feed")
async def get_feed(limit: int = 20):
    file_path = os.path.join("data", "ingested_cti.jsonl")
    if not os.path.exists(file_path):
        return []

    entries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line.strip()))
            except:
                continue

    # Return latest first
    entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return entries[:limit]

# -------------------
# Trigger ingestion manually
# -------------------
@app.post("/ingest_now")
async def ingest_now():
    try:
        # Run ingestion
        ingest_all_sources_main()

        # Read the updated status file
        status_file = os.path.join("data", "last_ingestion.json")
        if os.path.exists(status_file):
            with open(status_file, "r", encoding="utf-8") as f:
                status_data = json.load(f)
        else:
            status_data = {
                "last_run": None,
                "summary": {},
                "total_records": 0,
                "errors": {}
            }

        return {
            "status": "success",
            "message": "Ingestion completed successfully",
            "ingestion_summary": status_data
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e)
        })


# -------------------
# Dashboard
# -------------------
@app.get("/dashboard")
def serve_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# -------------------
# Analyze text endpoint with database storage
# -------------------
@app.post("/analyze")
async def analyze_text(payload: dict, db: Session = Depends(get_db_session)):
    text = payload.get("text", "")
    if not text:
        return JSONResponse(status_code=400, content={"error": "No input text provided."})

    try:
        # Perform ML analysis
        entities = extract_threat_entities(text)
        threat_type = classify_threat(text)
        severity = predict_severity(text)

        # Check if we're in test mode (no database)
        test_mode = os.getenv("TEST_MODE", "false").lower() == "true"
        
        if test_mode:
            # Return results without database storage for testing
            return {
                "id": "test-mode-id",
                "original_text": text,
                "entities": entities,
                "threat_type": str(threat_type),
                "severity": str(severity),
                "timestamp": datetime.utcnow().isoformat(),
                "test_mode": True
            }

        # Store in database (production mode)
        threat_intel_service = ThreatIntelService(db)
        
        # Create threat intelligence record
        threat_intel = threat_intel_service.create_threat_intel(
            source=DataSource.MANUAL,
            original_text=text,
            threat_type=ThreatCategory(threat_type) if threat_type in [e.value for e in ThreatCategory] else None,
            severity=SeverityLevel(severity) if severity in [e.value for e in SeverityLevel] else None,
            confidence_score=0.85  # Default confidence for manual analysis
        )

        # Store entities
        if entities:
            entity_service = EntityService(db)
            entity_list = []
            for entity_type, entity_values in entities.items():
                for value in entity_values:
                    entity_list.append({
                        "entity_type": entity_type,
                        "entity_value": value,
                        "confidence_score": 0.8
                    })
            
            if entity_list:
                entity_service.create_entities(str(threat_intel.id), entity_list)

        return {
            "id": str(threat_intel.id),
            "original_text": text,
            "entities": entities,
            "threat_type": str(threat_type),
            "severity": str(severity),
            "timestamp": threat_intel.created_at.isoformat()
        }

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# -------------------
# Upload CSV
# -------------------
@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        if "text" not in df.columns:
            return JSONResponse(status_code=400, content={"error": "CSV must contain a 'text' column"})

        results = []
        for _, row in df.iterrows():
            text = row["text"]
            if not isinstance(text, str):
                continue

            entities = extract_threat_entities(text)
            threat_type = classify_threat(text)
            severity = predict_severity(text)

            results.append({
                "original_text": text,
                "entities": entities,
                "threat_type": str(threat_type),
                "severity": str(severity),
            })

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            file_path = f.name

        return FileResponse(path=file_path, filename="cti_batch_results.json", media_type='application/json')

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# -------------------
# Database API endpoints
# -------------------
@app.get("/threats")
async def get_threats(
    limit: int = 100,
    source: Optional[str] = None,
    threat_type: Optional[str] = None,
    severity: Optional[str] = None,
    db: Session = Depends(get_db_session)
):
    """Get threat intelligence records with filtering"""
    try:
        threat_intel_service = ThreatIntelService(db)
        
        if source or threat_type or severity:
            # Convert string parameters to enums
            source_enum = DataSource(source) if source else None
            threat_type_enum = ThreatCategory(threat_type) if threat_type else None
            severity_enum = SeverityLevel(severity) if severity else None
            
            threats = threat_intel_service.search_threats(
                query="",
                source=source_enum,
                threat_type=threat_type_enum,
                severity=severity_enum
            )
        else:
            threats = threat_intel_service.get_recent_threats(limit=limit)
        
        # Convert to dict format
        result = []
        for threat in threats:
            result.append({
                "id": str(threat.id),
                "source": threat.source.value,
                "original_text": threat.original_text,
                "threat_type": threat.threat_type.value if threat.threat_type else None,
                "severity": threat.severity.value if threat.severity else None,
                "created_at": threat.created_at.isoformat(),
                "confidence_score": float(threat.confidence_score) if threat.confidence_score else None
            })
        
        return {"threats": result, "count": len(result)}
        
    except Exception as e:
        logger.error(f"Failed to fetch threats: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/threats/{threat_id}")
async def get_threat_details(threat_id: str, db: Session = Depends(get_db_session)):
    """Get detailed information about a specific threat"""
    try:
        threat_intel_service = ThreatIntelService(db)
        entity_service = EntityService(db)
        
        threat = threat_intel_service.get_threat_intel(threat_id)
        if not threat:
            raise HTTPException(status_code=404, detail="Threat not found")
        
        entities = entity_service.get_entities_by_threat(threat_id)
        
        return {
            "id": str(threat.id),
            "source": threat.source.value,
            "original_text": threat.original_text,
            "threat_type": threat.threat_type.value if threat.threat_type else None,
            "severity": threat.severity.value if threat.severity else None,
            "created_at": threat.created_at.isoformat(),
            "confidence_score": float(threat.confidence_score) if threat.confidence_score else None,
            "entities": [
                {
                    "type": entity.entity_type,
                    "value": entity.entity_value,
                    "confidence": float(entity.confidence_score) if entity.confidence_score else None
                }
                for entity in entities
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch threat details: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/analytics")
async def get_analytics(days: int = 30, db: Session = Depends(get_db_session)):
    """Get threat analytics and statistics"""
    try:
        analytics_service = AnalyticsService(db)
        stats = analytics_service.get_threat_statistics(days=days)
        
        # Convert enum keys to strings
        stats["threats_by_category"] = {
            k.value if hasattr(k, 'value') else str(k): v 
            for k, v in stats["threats_by_category"].items()
        }
        stats["threats_by_severity"] = {
            k.value if hasattr(k, 'value') else str(k): v 
            for k, v in stats["threats_by_severity"].items()
        }
        stats["threats_by_source"] = {
            k.value if hasattr(k, 'value') else str(k): v 
            for k, v in stats["threats_by_source"].items()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to fetch analytics: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.delete("/threats/{threat_id}")
async def delete_threat(threat_id: str, db: Session = Depends(get_db_session)):
    """Delete a threat intelligence record"""
    try:
        threat_intel_service = ThreatIntelService(db)
        
        if threat_intel_service.delete_threat_intel(threat_id):
            return {"message": "Threat deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Threat not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete threat: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
