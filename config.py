import os

# API Constants
API_VERSION = "1.0.0"
DEFAULT_ADMIN_KEY = "JAYDIP"
DEFAULT_API_KEY = "jaydip"
DEFAULT_DEMO_KEY = "1a873582a7c83342f961cc0a177b2b26"

# Flask Configuration
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
SECRET_KEY = os.environ.get("SECRET_KEY", "youtube_api_secure_key_change_in_production")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 5000))

# Database Configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///youtube_api.db")
if DATABASE_URL.startswith("postgres://"):
    # Heroku postgres URL fix
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Cache and Rate Limiting
CACHE_TIMEOUT = int(os.environ.get("CACHE_TIMEOUT", 60 * 60))  # 1 hour in seconds
DEFAULT_RATE_LIMIT = os.environ.get("DEFAULT_RATE_LIMIT", "100 per minute")

# Stream and Download Settings
STREAM_CHUNK_SIZE = int(os.environ.get("STREAM_CHUNK_SIZE", 1024 * 1024))  # 1MB
DOWNLOAD_DIR = os.environ.get("DOWNLOAD_DIR", "downloads")
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", 30))  # seconds

# Create downloads directory if it doesn't exist
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

# Anti-Bot Protection
MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", 10))

# Optional Proxy Settings (comma-separated list)
PROXY_LIST = os.environ.get("PROXY_LIST", "").split(",") if os.environ.get("PROXY_LIST") else []