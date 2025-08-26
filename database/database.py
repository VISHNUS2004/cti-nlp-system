"""
Database configuration and connection management
"""
import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import redis
from functools import lru_cache

from database.models import Base

class DatabaseConfig:
    def __init__(self):
        self.database_url = os.getenv(
            "DATABASE_URL", 
            "postgresql://cti_user:cti_password123@localhost:5432/cti_nlp"
        )
        self.redis_url = os.getenv(
            "REDIS_URL",
            "redis://localhost:6379/0"
        )
        
        # Database engine with connection pooling
        self.engine = create_engine(
            self.database_url,
            pool_size=10,
            max_overflow=20,
            pool_recycle=3600,
            echo=os.getenv("SQL_DEBUG", "false").lower() == "true"
        )
        
        # Session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Redis connection
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)

    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)

    def get_db(self) -> Generator[Session, None, None]:
        """Get database session"""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def get_redis(self) -> redis.Redis:
        """Get Redis client"""
        return self.redis_client

# Global database instance
@lru_cache()
def get_database_config() -> DatabaseConfig:
    return DatabaseConfig()

# Dependency for FastAPI
def get_db_session() -> Generator[Session, None, None]:
    db_config = get_database_config()
    yield from db_config.get_db()

def get_redis_client() -> redis.Redis:
    db_config = get_database_config()
    return db_config.get_redis()
