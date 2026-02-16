"""
Database Schema for Benchmark Results

Uses SQLAlchemy ORM for storing benchmark data.
"""

from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from pathlib import Path

Base = declarative_base()


class BenchmarkRun(Base):
    """Table for benchmark runs"""
    __tablename__ = "benchmark_runs"
    
    id = Column(Integer, primary_key=True)
    run_id = Column(String, unique=True, nullable=False)
    model_name = Column(String, nullable=False)
    workload_type = Column(String, nullable=False)
    batch_size = Column(Integer)
    num_samples = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to results
    results = relationship("BenchmarkResult", back_populates="run")
    
    def __repr__(self):
        return f"<BenchmarkRun(run_id='{self.run_id}', model='{self.model_name}')>"


class BenchmarkResult(Base):
    """Table for individual benchmark results"""
    __tablename__ = "benchmark_results"
    
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("benchmark_runs.id"))
    
    # Latency metrics
    latency_ms = Column(Float)
    ttft_ms = Column(Float)  # Time to first token
    
    # Token counts
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    
    # Resource metrics
    gpu_memory_mb = Column(Float)
    gpu_utilization_percent = Column(Float)
    
    # Cost
    cost_usd = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to run
    run = relationship("BenchmarkRun", back_populates="results")
    
    def __repr__(self):
        return f"<BenchmarkResult(latency={self.latency_ms}ms)>"


def init_database(db_path: str = "results/benchmark.db"):
    """
    Initialize the database.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        SQLAlchemy engine
    """
    # Ensure directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create engine
    engine = create_engine(f"sqlite:///{db_path}")
    
    # Create tables
    Base.metadata.create_all(engine)
    
    return engine


def get_session(engine):
    """Get a database session"""
    Session = sessionmaker(bind=engine)
    return Session()