"""
Test configuration and fixtures
"""
import pytest
import tempfile
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from backend.main import app
from database.models import Base
from database.database import get_db_session

# Test database URL (using SQLite for testing)
TEST_DATABASE_URL = "sqlite:///./test_cti.db"

@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine"""
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    # Cleanup
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def test_db(test_engine):
    """Create test database session"""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture(scope="function")
def client(test_db):
    """Create test client with database dependency override"""
    def override_get_db():
        try:
            yield test_db
        finally:
            pass
    
    app.dependency_overrides[get_db_session] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture
def sample_threat_data():
    """Sample threat intelligence data for testing"""
    return {
        "text": "Phishing campaign targeting banking customers with malicious URLs",
        "expected_threat_type": "Phishing",
        "expected_severity": "High"
    }

@pytest.fixture
def sample_csv_data():
    """Sample CSV data for batch testing"""
    csv_content = """text
"APT29 group using spear-phishing emails to target government agencies"
"Ransomware variant encrypting files with .locked extension"
"DDoS attack targeting financial institutions"
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        f.flush()
        yield f.name
    os.unlink(f.name)
