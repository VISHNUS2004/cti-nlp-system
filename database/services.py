"""
Database operations and CRUD functionality
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from database.models import (
    ThreatIntel, Entity, IOC, Prediction, IngestionLog, Feedback,
    ThreatCategory, SeverityLevel, DataSource
)

class ThreatIntelService:
    """Service class for threat intelligence operations"""
    
    def __init__(self, db: Session):
        self.db = db

    def create_threat_intel(self, 
                          source: DataSource,
                          original_text: str,
                          **kwargs) -> ThreatIntel:
        """Create a new threat intelligence record"""
        threat_intel = ThreatIntel(
            source=source,
            original_text=original_text,
            **kwargs
        )
        self.db.add(threat_intel)
        self.db.commit()
        self.db.refresh(threat_intel)
        return threat_intel

    def get_threat_intel(self, threat_id: str) -> Optional[ThreatIntel]:
        """Get threat intelligence by ID"""
        return self.db.query(ThreatIntel).filter(ThreatIntel.id == threat_id).first()

    def get_recent_threats(self, days: int = 7, limit: int = 100) -> List[ThreatIntel]:
        """Get recent threat intelligence records"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return (
            self.db.query(ThreatIntel)
            .filter(ThreatIntel.created_at >= cutoff_date)
            .order_by(desc(ThreatIntel.created_at))
            .limit(limit)
            .all()
        )

    def get_threats_by_type(self, threat_type: ThreatCategory) -> List[ThreatIntel]:
        """Get threats by category"""
        return (
            self.db.query(ThreatIntel)
            .filter(ThreatIntel.threat_type == threat_type)
            .order_by(desc(ThreatIntel.created_at))
            .all()
        )

    def update_threat_intel(self, threat_id: str, **kwargs) -> Optional[ThreatIntel]:
        """Update threat intelligence record"""
        threat_intel = self.get_threat_intel(threat_id)
        if threat_intel:
            for key, value in kwargs.items():
                if hasattr(threat_intel, key):
                    setattr(threat_intel, key, value)
            self.db.commit()
            self.db.refresh(threat_intel)
        return threat_intel

    def delete_threat_intel(self, threat_id: str) -> bool:
        """Delete threat intelligence record"""
        threat_intel = self.get_threat_intel(threat_id)
        if threat_intel:
            self.db.delete(threat_intel)
            self.db.commit()
            return True
        return False

    def search_threats(self, 
                      query: str,
                      source: Optional[DataSource] = None,
                      threat_type: Optional[ThreatCategory] = None,
                      severity: Optional[SeverityLevel] = None) -> List[ThreatIntel]:
        """Search threat intelligence records"""
        q = self.db.query(ThreatIntel)
        
        if query:
            q = q.filter(ThreatIntel.original_text.ilike(f"%{query}%"))
        
        if source:
            q = q.filter(ThreatIntel.source == source)
            
        if threat_type:
            q = q.filter(ThreatIntel.threat_type == threat_type)
            
        if severity:
            q = q.filter(ThreatIntel.severity == severity)
            
        return q.order_by(desc(ThreatIntel.created_at)).all()

class EntityService:
    """Service class for entity operations"""
    
    def __init__(self, db: Session):
        self.db = db

    def create_entities(self, threat_intel_id: str, entities: List[Dict[str, Any]]) -> List[Entity]:
        """Create entities for a threat intelligence record"""
        entity_objects = []
        for entity_data in entities:
            entity = Entity(
                threat_intel_id=threat_intel_id,
                **entity_data
            )
            entity_objects.append(entity)
            self.db.add(entity)
        
        self.db.commit()
        for entity in entity_objects:
            self.db.refresh(entity)
        return entity_objects

    def get_entities_by_threat(self, threat_intel_id: str) -> List[Entity]:
        """Get entities for a specific threat intelligence record"""
        return (
            self.db.query(Entity)
            .filter(Entity.threat_intel_id == threat_intel_id)
            .all()
        )

class IOCService:
    """Service class for IOC operations"""
    
    def __init__(self, db: Session):
        self.db = db

    def create_ioc(self, threat_intel_id: str, ioc_type: str, ioc_value: str) -> IOC:
        """Create an IOC record"""
        ioc = IOC(
            threat_intel_id=threat_intel_id,
            ioc_type=ioc_type,
            ioc_value=ioc_value
        )
        self.db.add(ioc)
        self.db.commit()
        self.db.refresh(ioc)
        return ioc

    def get_active_iocs(self) -> List[IOC]:
        """Get all active IOCs"""
        return self.db.query(IOC).filter(IOC.is_active == True).all()

    def deactivate_ioc(self, ioc_id: str) -> bool:
        """Deactivate an IOC"""
        ioc = self.db.query(IOC).filter(IOC.id == ioc_id).first()
        if ioc:
            ioc.is_active = False
            self.db.commit()
            return True
        return False

class AnalyticsService:
    """Service class for analytics and reporting"""
    
    def __init__(self, db: Session):
        self.db = db

    def get_threat_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get threat statistics for the specified period"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Total threats
        total_threats = (
            self.db.query(func.count(ThreatIntel.id))
            .filter(ThreatIntel.created_at >= cutoff_date)
            .scalar()
        )
        
        # Threats by category
        threats_by_category = (
            self.db.query(ThreatIntel.threat_type, func.count(ThreatIntel.id))
            .filter(ThreatIntel.created_at >= cutoff_date)
            .group_by(ThreatIntel.threat_type)
            .all()
        )
        
        # Threats by severity
        threats_by_severity = (
            self.db.query(ThreatIntel.severity, func.count(ThreatIntel.id))
            .filter(ThreatIntel.created_at >= cutoff_date)
            .group_by(ThreatIntel.severity)
            .all()
        )
        
        # Threats by source
        threats_by_source = (
            self.db.query(ThreatIntel.source, func.count(ThreatIntel.id))
            .filter(ThreatIntel.created_at >= cutoff_date)
            .group_by(ThreatIntel.source)
            .all()
        )
        
        return {
            "total_threats": total_threats,
            "threats_by_category": dict(threats_by_category),
            "threats_by_severity": dict(threats_by_severity),
            "threats_by_source": dict(threats_by_source),
            "period_days": days
        }
