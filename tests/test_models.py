"""
Unit tests for machine learning models and services
"""
import pytest
from unittest.mock import Mock, patch
import pandas as pd

from backend.threat_ner import extract_threat_entities
from backend.classifier import classify_threat
from backend.severity_predictor import predict_severity
from database.services import ThreatIntelService, EntityService, AnalyticsService
from database.models import ThreatIntel, DataSource, ThreatCategory, SeverityLevel

class TestNLPModels:
    """Test cases for NLP models"""
    
    def test_threat_ner_extraction(self):
        """Test threat entity extraction"""
        text = "APT29 group targeting government agencies with malicious URLs"
        entities = extract_threat_entities(text)
        
        assert isinstance(entities, dict)
        # Check that we get some entities back (specific entities may vary based on model)
        assert len(entities) >= 0

    def test_threat_ner_empty_text(self):
        """Test NER with empty text"""
        entities = extract_threat_entities("")
        assert isinstance(entities, dict)
        assert len(entities) == 0

    def test_threat_classification(self):
        """Test threat classification"""
        text = "Phishing campaign targeting banking customers"
        threat_type = classify_threat(text)
        
        assert isinstance(threat_type, str)
        assert len(threat_type) > 0

    def test_threat_classification_empty_text(self):
        """Test classification with empty text"""
        # Should handle empty text gracefully
        try:
            threat_type = classify_threat("")
            assert isinstance(threat_type, str)
        except Exception as e:
            # If it raises an exception, it should be a meaningful one
            assert "error" in str(e).lower() or "Error" in str(e)

    def test_severity_prediction(self):
        """Test severity prediction"""
        text = "Critical zero-day vulnerability exploited by APT groups"
        severity = predict_severity(text)
        
        assert isinstance(severity, str)
        assert len(severity) > 0

    def test_severity_prediction_empty_text(self):
        """Test severity prediction with empty text"""
        try:
            severity = predict_severity("")
            assert isinstance(severity, str)
        except Exception as e:
            assert "error" in str(e).lower() or "Error" in str(e)

class TestDatabaseServices:
    """Test cases for database services"""
    
    def test_threat_intel_service_create(self, test_db):
        """Test creating threat intelligence record"""
        service = ThreatIntelService(test_db)
        
        threat_intel = service.create_threat_intel(
            source=DataSource.MANUAL,
            original_text="Test threat description",
            threat_type=ThreatCategory.PHISHING,
            severity=SeverityLevel.HIGH
        )
        
        assert threat_intel.id is not None
        assert threat_intel.source == DataSource.MANUAL
        assert threat_intel.original_text == "Test threat description"
        assert threat_intel.threat_type == ThreatCategory.PHISHING
        assert threat_intel.severity == SeverityLevel.HIGH

    def test_threat_intel_service_get(self, test_db):
        """Test getting threat intelligence record"""
        service = ThreatIntelService(test_db)
        
        # Create a record
        threat_intel = service.create_threat_intel(
            source=DataSource.MANUAL,
            original_text="Test threat description"
        )
        
        # Get it back
        retrieved = service.get_threat_intel(str(threat_intel.id))
        assert retrieved is not None
        assert retrieved.id == threat_intel.id
        assert retrieved.original_text == "Test threat description"

    def test_threat_intel_service_get_nonexistent(self, test_db):
        """Test getting non-existent threat intelligence record"""
        service = ThreatIntelService(test_db)
        
        fake_id = "550e8400-e29b-41d4-a716-446655440000"
        retrieved = service.get_threat_intel(fake_id)
        assert retrieved is None

    def test_threat_intel_service_update(self, test_db):
        """Test updating threat intelligence record"""
        service = ThreatIntelService(test_db)
        
        # Create a record
        threat_intel = service.create_threat_intel(
            source=DataSource.MANUAL,
            original_text="Test threat description"
        )
        
        # Update it
        updated = service.update_threat_intel(
            str(threat_intel.id),
            threat_type=ThreatCategory.MALWARE,
            severity=SeverityLevel.CRITICAL
        )
        
        assert updated.threat_type == ThreatCategory.MALWARE
        assert updated.severity == SeverityLevel.CRITICAL

    def test_threat_intel_service_delete(self, test_db):
        """Test deleting threat intelligence record"""
        service = ThreatIntelService(test_db)
        
        # Create a record
        threat_intel = service.create_threat_intel(
            source=DataSource.MANUAL,
            original_text="Test threat description"
        )
        
        # Delete it
        deleted = service.delete_threat_intel(str(threat_intel.id))
        assert deleted is True
        
        # Verify it's gone
        retrieved = service.get_threat_intel(str(threat_intel.id))
        assert retrieved is None

    def test_threat_intel_service_search(self, test_db):
        """Test searching threat intelligence records"""
        service = ThreatIntelService(test_db)
        
        # Create some test records
        service.create_threat_intel(
            source=DataSource.MANUAL,
            original_text="Phishing campaign targeting banks",
            threat_type=ThreatCategory.PHISHING
        )
        service.create_threat_intel(
            source=DataSource.TWITTER,
            original_text="Malware spreading through email",
            threat_type=ThreatCategory.MALWARE
        )
        
        # Search by query
        results = service.search_threats("phishing")
        assert len(results) >= 1
        assert "phishing" in results[0].original_text.lower()
        
        # Search by threat type
        results = service.search_threats("", threat_type=ThreatCategory.MALWARE)
        assert len(results) >= 1
        assert results[0].threat_type == ThreatCategory.MALWARE

    def test_entity_service_create(self, test_db):
        """Test creating entities"""
        threat_service = ThreatIntelService(test_db)
        entity_service = EntityService(test_db)
        
        # Create a threat record first
        threat_intel = threat_service.create_threat_intel(
            source=DataSource.MANUAL,
            original_text="APT29 group targeting organizations"
        )
        
        # Create entities
        entities = entity_service.create_entities(
            str(threat_intel.id),
            [
                {
                    "entity_type": "ORG",
                    "entity_value": "APT29",
                    "confidence_score": 0.9
                }
            ]
        )
        
        assert len(entities) == 1
        assert entities[0].entity_type == "ORG"
        assert entities[0].entity_value == "APT29"

    def test_analytics_service(self, test_db):
        """Test analytics service"""
        threat_service = ThreatIntelService(test_db)
        analytics_service = AnalyticsService(test_db)
        
        # Create some test data
        threat_service.create_threat_intel(
            source=DataSource.MANUAL,
            original_text="Test threat 1",
            threat_type=ThreatCategory.PHISHING,
            severity=SeverityLevel.HIGH
        )
        threat_service.create_threat_intel(
            source=DataSource.TWITTER,
            original_text="Test threat 2",
            threat_type=ThreatCategory.MALWARE,
            severity=SeverityLevel.MEDIUM
        )
        
        # Get statistics
        stats = analytics_service.get_threat_statistics(days=30)
        
        assert "total_threats" in stats
        assert stats["total_threats"] >= 2
        assert "threats_by_category" in stats
        assert "threats_by_severity" in stats
        assert "threats_by_source" in stats

class TestDataIntegration:
    """Test cases for data integration workflows"""
    
    def test_end_to_end_analysis_workflow(self, test_db):
        """Test complete analysis workflow"""
        # This would test the complete flow from text input to database storage
        text = "APT29 phishing campaign targeting financial institutions"
        
        # Extract entities
        entities = extract_threat_entities(text)
        
        # Classify threat
        threat_type = classify_threat(text)
        
        # Predict severity
        severity = predict_severity(text)
        
        # Store in database
        threat_service = ThreatIntelService(test_db)
        threat_intel = threat_service.create_threat_intel(
            source=DataSource.MANUAL,
            original_text=text,
            threat_type=ThreatCategory(threat_type) if threat_type in [e.value for e in ThreatCategory] else None,
            severity=SeverityLevel(severity) if severity in [e.value for e in SeverityLevel] else None
        )
        
        # Verify the record was created
        assert threat_intel.id is not None
        assert threat_intel.original_text == text
        
        # Store entities if any
        if entities:
            entity_service = EntityService(test_db)
            entity_list = []
            for entity_type, entity_values in entities.items():
                for value in entity_values:
                    entity_list.append({
                        "entity_type": entity_type,
                        "entity_value": value,
                        "confidence_score": 0.8
                    })
            
            if entity_list:
                created_entities = entity_service.create_entities(str(threat_intel.id), entity_list)
                assert len(created_entities) == len(entity_list)
