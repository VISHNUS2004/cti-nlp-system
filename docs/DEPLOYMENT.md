# CTI-NLP System Deployment Guide

This guide provides comprehensive instructions for deploying the CTI-NLP system in various environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development Setup](#local-development-setup)
- [Docker Deployment](#docker-deployment)
- [Production Deployment](#production-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **CPU**: 2+ cores (4+ recommended for production)
- **Memory**: 4GB RAM minimum (8GB+ recommended)
- **Storage**: 10GB available space (20GB+ for production)
- **OS**: Linux (Ubuntu 20.04+), macOS, Windows 10+

### Software Dependencies

- Docker and Docker Compose (20.10+)
- Python 3.9+ (for local development)
- PostgreSQL 12+ (if not using Docker)
- Redis 6+ (if not using Docker)
- Git

## Local Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sanjanb/cti-nlp-system.git
cd cti-nlp-system
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

Required environment variables:
```bash
# Database
DATABASE_URL=postgresql://cti_user:cti_password123@localhost:5432/cti_nlp

# Redis
REDIS_URL=redis://localhost:6379/0

# External APIs
TWITTER_BEARER_TOKEN=your_twitter_token_here

# Application
ENVIRONMENT=development
SECRET_KEY=your_secret_key_here
```

### 3. Python Environment Setup

```bash
# Create virtual environment
python -m venv myenv

# Activate virtual environment
# Windows
myenv\Scripts\activate
# Linux/Mac
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 4. Database Setup

```bash
# Start PostgreSQL and Redis (if using Docker)
docker run -d --name postgres -p 5432:5432 \
  -e POSTGRES_DB=cti_nlp \
  -e POSTGRES_USER=cti_user \
  -e POSTGRES_PASSWORD=cti_password123 \
  postgres:15-alpine

docker run -d --name redis -p 6379:6379 redis:7-alpine

# Initialize database
python -c "from database.database import get_database_config; get_database_config().create_tables()"
```

### 5. Run the Application

```bash
# Start the FastAPI server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for the API documentation.

## Docker Deployment

### 1. Quick Start with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 2. Production Docker Compose

For production, use the production compose file:

```bash
# Start production services
docker-compose -f docker-compose.yml up -d

# Check service health
docker-compose ps
```

### 3. Individual Service Management

```bash
# Start only database services
docker-compose up -d postgres redis

# Start application
docker-compose up -d app

# Scale application instances
docker-compose up -d --scale app=3
```

## Production Deployment

### 1. Server Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
sudo pip3 install docker-compose

# Create application user
sudo useradd -m -s /bin/bash cti-nlp
sudo usermod -aG docker cti-nlp
```

### 2. Application Deployment

```bash
# Switch to application user
sudo su - cti-nlp

# Clone repository
git clone https://github.com/sanjanb/cti-nlp-system.git
cd cti-nlp-system

# Set production environment
cp .env.example .env
# Edit .env with production values

# Start services
docker-compose up -d

# Verify deployment
docker-compose ps
curl http://localhost:8000/health
```

### 3. SSL/HTTPS Setup

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Update nginx configuration for SSL
# Edit nginx/nginx.conf to include SSL settings
```

### 4. Firewall Configuration

```bash
# Configure UFW
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable
```

## Cloud Deployment

### AWS Deployment

#### 1. EC2 Setup

```bash
# Launch EC2 instance (t3.medium or larger)
# Configure security groups (ports 22, 80, 443)
# Connect to instance

# Install dependencies
sudo yum update -y
sudo yum install -y docker git
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo pip3 install docker-compose
```

#### 2. RDS Database Setup

```bash
# Create RDS PostgreSQL instance
# Update DATABASE_URL in .env file
DATABASE_URL=postgresql://username:password@rds-endpoint:5432/dbname
```

#### 3. ElastiCache Redis

```bash
# Create ElastiCache Redis cluster
# Update REDIS_URL in .env file
REDIS_URL=redis://elasticache-endpoint:6379/0
```

### Google Cloud Platform

#### 1. Cloud Run Deployment

```bash
# Build and push Docker image
gcloud builds submit --tag gcr.io/PROJECT_ID/cti-nlp

# Deploy to Cloud Run
gcloud run deploy cti-nlp \
  --image gcr.io/PROJECT_ID/cti-nlp \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### 2. Cloud SQL Setup

```bash
# Create Cloud SQL PostgreSQL instance
gcloud sql instances create cti-postgres \
  --database-version=POSTGRES_13 \
  --tier=db-f1-micro \
  --region=us-central1
```

### Azure Deployment

#### 1. Container Instances

```bash
# Create resource group
az group create --name cti-nlp-rg --location eastus

# Deploy container
az container create \
  --resource-group cti-nlp-rg \
  --name cti-nlp-app \
  --image your-registry/cti-nlp:latest \
  --ports 8000 \
  --environment-variables DATABASE_URL=... REDIS_URL=...
```

## Monitoring and Maintenance

### 1. Health Monitoring

```bash
# Application health check
curl http://localhost:8000/health

# Service status
docker-compose ps

# View logs
docker-compose logs -f app
docker-compose logs -f postgres
docker-compose logs -f redis
```

### 2. Log Management

```bash
# Configure log rotation
sudo nano /etc/logrotate.d/cti-nlp

# Content:
/var/log/cti-nlp/*.log {
    daily
    missingok
    rotate 30
    compress
    notifempty
    create 0644 cti-nlp cti-nlp
}
```

### 3. Backup Strategy

```bash
# Database backup script
#!/bin/bash
BACKUP_DIR="/backups/cti-nlp"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
docker-compose exec postgres pg_dump -U cti_user cti_nlp > $BACKUP_DIR/backup_$DATE.sql

# Compress backup
gzip $BACKUP_DIR/backup_$DATE.sql

# Remove old backups (keep 7 days)
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete
```

### 4. Update Procedure

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart services
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Verify deployment
curl http://localhost:8000/health
```

## Performance Optimization

### 1. Database Optimization

```sql
-- Create additional indexes for better performance
CREATE INDEX CONCURRENTLY idx_threat_intel_text_search 
ON threat_intel USING gin(to_tsvector('english', original_text));

-- Analyze tables
ANALYZE threat_intel;
ANALYZE entities;
ANALYZE iocs;
```

### 2. Redis Caching

```python
# Cache frequently accessed data
# Implement in backend/main.py
@app.get("/threats")
async def get_threats(db: Session = Depends(get_db_session)):
    cache_key = "recent_threats"
    cached_data = redis_client.get(cache_key)
    
    if cached_data:
        return json.loads(cached_data)
    
    # Fetch from database
    threats = threat_service.get_recent_threats()
    
    # Cache for 5 minutes
    redis_client.setex(cache_key, 300, json.dumps(threats))
    
    return threats
```

### 3. Load Balancing

```nginx
# nginx/nginx.conf
upstream app_backend {
    server app1:8000;
    server app2:8000;
    server app3:8000;
}

server {
    location / {
        proxy_pass http://app_backend;
    }
}
```

## Security Hardening

### 1. Environment Variables

```bash
# Use secrets management
export DATABASE_PASSWORD=$(cat /run/secrets/db_password)
export API_KEY=$(cat /run/secrets/api_key)
```

### 2. Network Security

```bash
# Create internal network
docker network create --internal cti-internal

# Update docker-compose.yml to use internal network
```

### 3. Regular Updates

```bash
# Update base images regularly
docker-compose pull
docker-compose up -d

# Update Python dependencies
pip install --upgrade -r requirements.txt
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Issues

```bash
# Check database status
docker-compose exec postgres pg_isready

# Check connection from app
docker-compose exec app python -c "
from database.database import get_database_config
db = get_database_config()
print('Database connection successful')
"
```

#### 2. Memory Issues

```bash
# Check memory usage
docker stats

# Increase memory limits in docker-compose.yml
services:
  app:
    deploy:
      resources:
        limits:
          memory: 2G
```

#### 3. Model Loading Issues

```bash
# Check if models exist
ls -la models/

# Retrain models if needed
python scripts/train_threat_classifier.py
python scripts/train_severity_model.py
```

### Log Analysis

```bash
# Application logs
docker-compose logs app | grep ERROR

# Database logs
docker-compose logs postgres | grep ERROR

# System logs
journalctl -u docker -f
```

### Performance Issues

```bash
# Check database performance
docker-compose exec postgres psql -U cti_user -d cti_nlp -c "
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;
"

# Check Redis performance
docker-compose exec redis redis-cli info stats
```

## Support and Maintenance

### Regular Maintenance Tasks

1. **Daily**: Check application health and logs
2. **Weekly**: Database backup and cleanup
3. **Monthly**: Security updates and dependency updates
4. **Quarterly**: Performance review and optimization

### Contact Information

- **Development Team**: [GitHub Issues](https://github.com/sanjanb/cti-nlp-system/issues)
- **Documentation**: [Project Wiki](https://github.com/sanjanb/cti-nlp-system/wiki)
- **Security Issues**: security@your-domain.com

---

*Last updated: August 2025*
