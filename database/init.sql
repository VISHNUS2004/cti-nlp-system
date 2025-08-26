-- Database initialization script for CTI-NLP System
-- This script creates the necessary tables and indexes

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create enum types
CREATE TYPE threat_category AS ENUM (
    'Phishing', 'Malware', 'Ransomware', 'APT', 'DDoS', 
    'Data Breach', 'Vulnerability', 'Insider Threat', 'Other'
);

CREATE TYPE severity_level AS ENUM ('Low', 'Medium', 'High', 'Critical');

CREATE TYPE data_source AS ENUM ('Twitter', 'DarkWeb', 'MITRE', 'Manual', 'Feed');

-- Threat Intelligence Records
CREATE TABLE threat_intel (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source data_source NOT NULL,
    original_text TEXT NOT NULL,
    processed_text TEXT,
    threat_type threat_category,
    severity severity_level,
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    
    -- Metadata
    source_url TEXT,
    source_id VARCHAR(255),
    author VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Additional fields
    language VARCHAR(10) DEFAULT 'en',
    region VARCHAR(100),
    tags TEXT[],
    
    UNIQUE(source, source_id)
);

-- Named Entities extracted from threat intel
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    threat_intel_id UUID REFERENCES threat_intel(id) ON DELETE CASCADE,
    entity_type VARCHAR(50) NOT NULL, -- PERSON, ORG, LOC, etc.
    entity_value VARCHAR(500) NOT NULL,
    confidence_score DECIMAL(3,2),
    start_char INTEGER,
    end_char INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- IOCs (Indicators of Compromise)
CREATE TABLE iocs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    threat_intel_id UUID REFERENCES threat_intel(id) ON DELETE CASCADE,
    ioc_type VARCHAR(50) NOT NULL, -- ip, domain, hash, email, etc.
    ioc_value VARCHAR(1000) NOT NULL,
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    
    UNIQUE(ioc_type, ioc_value)
);

-- Model predictions and their confidence scores
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    threat_intel_id UUID REFERENCES threat_intel(id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    prediction_type VARCHAR(50) NOT NULL, -- threat_type, severity, etc.
    predicted_value VARCHAR(100) NOT NULL,
    confidence_score DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Data ingestion log
CREATE TABLE ingestion_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source data_source NOT NULL,
    status VARCHAR(50) NOT NULL, -- success, failed, partial
    records_processed INTEGER DEFAULT 0,
    records_failed INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    execution_time_ms INTEGER
);

-- User feedback for model improvement
CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    threat_intel_id UUID REFERENCES threat_intel(id) ON DELETE CASCADE,
    user_id VARCHAR(255),
    feedback_type VARCHAR(50), -- classification, severity, false_positive
    original_prediction VARCHAR(100),
    corrected_value VARCHAR(100),
    comments TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_threat_intel_source ON threat_intel(source);
CREATE INDEX idx_threat_intel_created_at ON threat_intel(created_at);
CREATE INDEX idx_threat_intel_threat_type ON threat_intel(threat_type);
CREATE INDEX idx_threat_intel_severity ON threat_intel(severity);
CREATE INDEX idx_threat_intel_tags ON threat_intel USING GIN(tags);

CREATE INDEX idx_entities_threat_intel_id ON entities(threat_intel_id);
CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_entities_value ON entities(entity_value);

CREATE INDEX idx_iocs_threat_intel_id ON iocs(threat_intel_id);
CREATE INDEX idx_iocs_type ON iocs(ioc_type);
CREATE INDEX idx_iocs_value ON iocs(ioc_value);
CREATE INDEX idx_iocs_active ON iocs(is_active);

CREATE INDEX idx_predictions_threat_intel_id ON predictions(threat_intel_id);
CREATE INDEX idx_predictions_model ON predictions(model_name);

CREATE INDEX idx_ingestion_logs_source ON ingestion_logs(source);
CREATE INDEX idx_ingestion_logs_status ON ingestion_logs(status);
CREATE INDEX idx_ingestion_logs_started_at ON ingestion_logs(started_at);

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_threat_intel_updated_at 
    BEFORE UPDATE ON threat_intel 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Insert some sample data for testing
INSERT INTO threat_intel (source, original_text, threat_type, severity) VALUES
('Manual', 'Phishing campaign targeting banking customers with fake login pages', 'Phishing', 'High'),
('Manual', 'New ransomware variant encrypting files with .locked extension', 'Ransomware', 'Critical'),
('Manual', 'APT group using zero-day vulnerability in web browsers', 'APT', 'Critical');

-- Create views for common queries
CREATE VIEW threat_summary AS
SELECT 
    source,
    threat_type,
    severity,
    COUNT(*) as count,
    AVG(confidence_score) as avg_confidence
FROM threat_intel 
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY source, threat_type, severity;

CREATE VIEW recent_threats AS
SELECT 
    id,
    source,
    original_text,
    threat_type,
    severity,
    confidence_score,
    created_at
FROM threat_intel 
WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY created_at DESC;
