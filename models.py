from datetime import datetime, timedelta
import os
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Text, BigInteger
from sqlalchemy.orm import relationship

db = SQLAlchemy()

class ApiKey(db.Model):
    __tablename__ = 'api_keys'
    
    id = Column(Integer, primary_key=True)
    key = Column(String(64), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    valid_until = Column(DateTime, nullable=False)
    daily_limit = Column(Integer, default=100)
    reset_at = Column(DateTime, default=lambda: datetime.now() + timedelta(days=1))
    count = Column(Integer, default=0)
    created_by = Column(Integer, ForeignKey('api_keys.id'), nullable=True)

    # Self-referential relationship
    created_keys = relationship("ApiKey", backref="creator", remote_side=[id])
    
    def is_expired(self):
        return datetime.now() > self.valid_until
    
    def remaining_requests(self):
        if datetime.now() > self.reset_at:
            return self.daily_limit
        return self.daily_limit - self.count

class ApiLog(db.Model):
    __tablename__ = 'api_logs'
    
    id = Column(Integer, primary_key=True)
    api_key_id = Column(Integer, ForeignKey('api_keys.id'), nullable=False)
    endpoint = Column(String(255), nullable=False)
    query = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    timestamp = Column(DateTime, default=datetime.now)
    response_status = Column(Integer, default=200)
    
    # Relationship
    api_key = relationship("ApiKey", backref="logs")

def init_db(app):
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
        
        # Check if admin API key exists
        admin_key = ApiKey.query.filter_by(key="JAYDIP").first()
        if not admin_key:
            # Create admin key
            admin_key = ApiKey(
                key="JAYDIP",
                name="Admin Key",
                is_admin=True,
                created_at=datetime.now(),
                valid_until=datetime.now() + timedelta(days=365),
                daily_limit=10000,
                reset_at=datetime.now() + timedelta(days=1),
                count=0
            )
            db.session.add(admin_key)
            
            # Create demo key
            public_key = ApiKey(
                key="1a873582a7c83342f961cc0a177b2b26",
                name="Public Demo Key",
                is_admin=False,
                created_at=datetime.now(),
                valid_until=datetime.now() + timedelta(days=365),
                daily_limit=100,
                reset_at=datetime.now() + timedelta(days=1),
                count=0,
                created_by=1  # Will be linked to admin after admin is created
            )
            db.session.add(public_key)
            
            # Create API request key
            api_request_key = ApiKey(
                key="jaydip",
                name="API Request Key",
                is_admin=False,
                created_at=datetime.now(),
                valid_until=datetime.now() + timedelta(days=365),
                daily_limit=5000,
                reset_at=datetime.now() + timedelta(days=1),
                count=0,
                created_by=1  # Will be linked to admin after admin is created
            )
            db.session.add(api_request_key)
            
            db.session.commit()