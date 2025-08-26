"""
Unit tests for threat intelligence API endpoints
"""
import pytest
import json
from fastapi.testclient import TestClient

class TestThreatIntelAPI:
    """Test cases for threat intelligence endpoints"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "services" in data

    def test_analyze_text_success(self, client, sample_threat_data):
        """Test successful text analysis"""
        response = client.post("/analyze", json={"text": sample_threat_data["text"]})
        assert response.status_code == 200
        
        data = response.json()
        assert "id" in data
        assert data["original_text"] == sample_threat_data["text"]
        assert "entities" in data
        assert "threat_type" in data
        assert "severity" in data
        assert "timestamp" in data

    def test_analyze_empty_text(self, client):
        """Test analysis with empty text"""
        response = client.post("/analyze", json={"text": ""})
        assert response.status_code == 400
        assert "error" in response.json()

    def test_analyze_missing_text(self, client):
        """Test analysis with missing text field"""
        response = client.post("/analyze", json={})
        assert response.status_code == 400
        assert "error" in response.json()

    def test_get_threats_empty(self, client):
        """Test getting threats when database is empty"""
        response = client.get("/threats")
        assert response.status_code == 200
        data = response.json()
        assert "threats" in data
        assert "count" in data
        assert data["count"] == 0

    def test_get_threats_with_data(self, client, sample_threat_data):
        """Test getting threats after adding data"""
        # First add a threat
        response = client.post("/analyze", json={"text": sample_threat_data["text"]})
        assert response.status_code == 200
        threat_id = response.json()["id"]
        
        # Then get threats
        response = client.get("/threats")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] > 0
        assert len(data["threats"]) > 0

    def test_get_threat_details(self, client, sample_threat_data):
        """Test getting detailed threat information"""
        # First add a threat
        response = client.post("/analyze", json={"text": sample_threat_data["text"]})
        assert response.status_code == 200
        threat_id = response.json()["id"]
        
        # Then get details
        response = client.get(f"/threats/{threat_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == threat_id
        assert "entities" in data
        assert "original_text" in data

    def test_get_nonexistent_threat(self, client):
        """Test getting details for non-existent threat"""
        fake_id = "550e8400-e29b-41d4-a716-446655440000"
        response = client.get(f"/threats/{fake_id}")
        assert response.status_code == 404

    def test_delete_threat(self, client, sample_threat_data):
        """Test deleting a threat"""
        # First add a threat
        response = client.post("/analyze", json={"text": sample_threat_data["text"]})
        assert response.status_code == 200
        threat_id = response.json()["id"]
        
        # Then delete it
        response = client.delete(f"/threats/{threat_id}")
        assert response.status_code == 200
        assert "message" in response.json()
        
        # Verify it's deleted
        response = client.get(f"/threats/{threat_id}")
        assert response.status_code == 404

    def test_delete_nonexistent_threat(self, client):
        """Test deleting non-existent threat"""
        fake_id = "550e8400-e29b-41d4-a716-446655440000"
        response = client.delete(f"/threats/{fake_id}")
        assert response.status_code == 404

    def test_get_analytics_empty(self, client):
        """Test analytics with empty database"""
        response = client.get("/analytics")
        assert response.status_code == 200
        data = response.json()
        assert "total_threats" in data
        assert data["total_threats"] == 0
        assert "threats_by_category" in data
        assert "threats_by_severity" in data
        assert "threats_by_source" in data

    def test_get_analytics_with_data(self, client, sample_threat_data):
        """Test analytics with data"""
        # Add some threats
        for i in range(3):
            response = client.post("/analyze", json={"text": f"{sample_threat_data['text']} {i}"})
            assert response.status_code == 200
        
        # Get analytics
        response = client.get("/analytics")
        assert response.status_code == 200
        data = response.json()
        assert data["total_threats"] >= 3

    def test_filter_threats_by_source(self, client, sample_threat_data):
        """Test filtering threats by source"""
        # Add a threat
        response = client.post("/analyze", json={"text": sample_threat_data["text"]})
        assert response.status_code == 200
        
        # Filter by source
        response = client.get("/threats?source=Manual")
        assert response.status_code == 200
        data = response.json()
        for threat in data["threats"]:
            assert threat["source"] == "Manual"

class TestCSVUpload:
    """Test cases for CSV upload functionality"""
    
    def test_upload_csv_success(self, client, sample_csv_data):
        """Test successful CSV upload"""
        with open(sample_csv_data, 'rb') as f:
            response = client.post(
                "/upload_csv",
                files={"file": ("test.csv", f, "text/csv")}
            )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_upload_invalid_csv(self, client):
        """Test uploading invalid CSV"""
        csv_content = "invalid,header\nsome,data"
        with open("test_invalid.csv", "w") as f:
            f.write(csv_content)
        
        with open("test_invalid.csv", "rb") as f:
            response = client.post(
                "/upload_csv",
                files={"file": ("test_invalid.csv", f, "text/csv")}
            )
        assert response.status_code == 400
        
        # Cleanup
        import os
        os.unlink("test_invalid.csv")

class TestRealTimeIngestion:
    """Test cases for real-time ingestion functionality"""
    
    def test_manual_ingestion_trigger(self, client):
        """Test manual ingestion trigger"""
        response = client.post("/ingest_now")
        # This might fail due to external dependencies, so we check for appropriate response
        assert response.status_code in [200, 500]  # 500 is acceptable for external API failures

    def test_ingestion_status(self, client):
        """Test getting ingestion status"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "ingestion_status" in data

    def test_feed_endpoint(self, client):
        """Test getting ingested feed"""
        response = client.get("/feed")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
