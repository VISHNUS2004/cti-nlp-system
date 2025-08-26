"""
Database models for the CTI-NLP system using SQLAlchemy
"""
import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import List, Optional

from sqlalchemy import (
    Column, String, Text, DateTime, Boolean, Integer, 
    Numeric, ForeignKey, Index, ARRAY, func
)
from sqlalchemy.dialects.postgresql import UUID, ENUM
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.sql import func

Base = declarative_base()

# Enum definitions
class ThreatCategory(PyEnum):
    PHISHING = "Phishing"
    MALWARE = "Malware"
    RANSOMWARE = "Ransomware"
    APT = "APT"
    DDOS = "DDoS"
    DATA_BREACH = "Data Breach"
    VULNERABILITY = "Vulnerability"
    INSIDER_THREAT = "Insider Threat"
    OTHER = "Other"

class SeverityLevel(PyEnum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class DataSource(PyEnum):
    TWITTER = "Twitter"
    DARKWEB = "DarkWeb"
    MITRE = "MITRE"
    MANUAL = "Manual"
    FEED = "Feed"

# SQLAlchemy Enums
threat_category_enum = ENUM(ThreatCategory, name="threat_category", create_type=False)
severity_level_enum = ENUM(SeverityLevel, name="severity_level", create_type=False)
data_source_enum = ENUM(DataSource, name="data_source", create_type=False)

class ThreatIntel(Base):
    __tablename__ = "threat_intel"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source = Column(data_source_enum, nullable=False)
    original_text = Column(Text, nullable=False)
    processed_text = Column(Text)
    threat_type = Column(threat_category_enum)
    severity = Column(severity_level_enum)
    confidence_score = Column(Numeric(3, 2))
    
    # Metadata
    source_url = Column(Text)
    source_id = Column(String(255))
    author = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Additional fields
    language = Column(String(10), default="en")
    region = Column(String(100))
    tags = Column(ARRAY(String))
    
    # Relationships
    entities = relationship("Entity", back_populates="threat_intel", cascade="all, delete-orphan")
    iocs = relationship("IOC", back_populates="threat_intel", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="threat_intel", cascade="all, delete-orphan")
    feedback = relationship("Feedback", back_populates="threat_intel", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_threat_intel_source', 'source'),
        Index('idx_threat_intel_created_at', 'created_at'),
        Index('idx_threat_intel_threat_type', 'threat_type'),
        Index('idx_threat_intel_severity', 'severity'),
    )

class Entity(Base):
    __tablename__ = "entities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    threat_intel_id = Column(UUID(as_uuid=True), ForeignKey("threat_intel.id", ondelete="CASCADE"))
    entity_type = Column(String(50), nullable=False)
    entity_value = Column(String(500), nullable=False)
    confidence_score = Column(Numeric(3, 2))
    start_char = Column(Integer)
    end_char = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    threat_intel = relationship("ThreatIntel", back_populates="entities")

    __table_args__ = (
        Index('idx_entities_threat_intel_id', 'threat_intel_id'),
        Index('idx_entities_type', 'entity_type'),
        Index('idx_entities_value', 'entity_value'),
    )

class IOC(Base):
    __tablename__ = "iocs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    threat_intel_id = Column(UUID(as_uuid=True), ForeignKey("threat_intel.id", ondelete="CASCADE"))
    ioc_type = Column(String(50), nullable=False)
    ioc_value = Column(String(1000), nullable=False)
    first_seen = Column(DateTime(timezone=True), server_default=func.now())
    last_seen = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationships
    threat_intel = relationship("ThreatIntel", back_populates="iocs")

    __table_args__ = (
        Index('idx_iocs_threat_intel_id', 'threat_intel_id'),
        Index('idx_iocs_type', 'ioc_type'),
        Index('idx_iocs_value', 'ioc_value'),
        Index('idx_iocs_active', 'is_active'),
    )

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    threat_intel_id = Column(UUID(as_uuid=True), ForeignKey("threat_intel.id", ondelete="CASCADE"))
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50))
    prediction_type = Column(String(50), nullable=False)
    predicted_value = Column(String(100), nullable=False)
    confidence_score = Column(Numeric(3, 2))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    threat_intel = relationship("ThreatIntel", back_populates="predictions")

    __table_args__ = (
        Index('idx_predictions_threat_intel_id', 'threat_intel_id'),
        Index('idx_predictions_model', 'model_name'),
    )

class IngestionLog(Base):
    __tablename__ = "ingestion_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source = Column(data_source_enum, nullable=False)
    status = Column(String(50), nullable=False)
    records_processed = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    error_message = Column(Text)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    execution_time_ms = Column(Integer)

    __table_args__ = (
        Index('idx_ingestion_logs_source', 'source'),
        Index('idx_ingestion_logs_status', 'status'),
        Index('idx_ingestion_logs_started_at', 'started_at'),
    )

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    threat_intel_id = Column(UUID(as_uuid=True), ForeignKey("threat_intel.id", ondelete="CASCADE"))
    user_id = Column(String(255))
    feedback_type = Column(String(50))
    original_prediction = Column(String(100))
    corrected_value = Column(String(100))
    comments = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    threat_intel = relationship("ThreatIntel", back_populates="feedback")
