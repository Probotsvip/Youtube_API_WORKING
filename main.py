import asyncio
import base64
import concurrent.futures
import datetime
import hashlib
import json
import logging
import os
import random
import re
import secrets
import string
import time
import uuid
from functools import wraps
from typing import Dict, List, Optional, Union
from urllib.parse import parse_qs, urlparse

import httpx
import pymongo
import yt_dlp
from flask import (Flask, Response, jsonify, make_response, request,
                   send_file, stream_with_context)
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from youtubesearchpython.__future__ import VideosSearch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_CONCURRENT_REQUESTS = 10
REQUEST_TIMEOUT = 30
STREAM_CHUNK_SIZE = 1024 * 1024  # 1MB
RATE_LIMIT = "100 per minute"
API_RATE_LIMIT = "500 per hour"
CACHE_TIMEOUT = 60 * 60  # 1 hour
DOWNLOAD_DIR = "downloads"
API_VERSION = "1.0.0"

# Create downloads directory if it doesn't exist
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get("SESSION_SECRET", secrets.token_hex(16))

# Initialize rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[RATE_LIMIT],
    storage_uri=os.environ.get("REDIS_URL", "memory://"),
    strategy="fixed-window",
)

# Create a thread pool for async operations
executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)

# In-memory cache
cache = {}
api_keys = {}
admin_keys = {
    "admin_master_key": {
        "name": "Admin",
        "is_admin": True,
        "created_at": datetime.datetime.now(),
        "valid_until": datetime.datetime.now() + datetime.timedelta(days=365),
        "daily_limit": 10000,
        "reset_at": datetime.datetime.now() + datetime.timedelta(days=1),
        "count": 0
    }
}

# MongoDB connection (optional)
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://jaydipmore74:xCpTm5OPAfRKYnif@cluster0.5jo18.mongodb.net/?retryWrites=true&w=majority")
mongo_client = None
db = None

try:
    mongo_client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = mongo_client.youtube_api
    mongo_client.server_info()  # Test connection
    logger.info("Connected to MongoDB")
    
    # Initialize collections for API keys and logs
    api_keys_collection = db.api_keys
    logs_collection = db.logs

    # Load existing API keys from MongoDB
    for key in api_keys_collection.find():
        api_key = key.pop("_id")
        api_keys[api_key] = key
        if key.get("is_admin", False):
            admin_keys[api_key] = key
    
except Exception as e:
    logger.warning(f"MongoDB connection failed: {e}")
    logger.warning("Running without MongoDB - some features will be limited")
    mongo_client = None
    db = None

# User agents list for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.48",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/112.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 16_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 OPR/88.0.4412.53",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 OPR/97.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.39",
]

# Proxy rotation (if needed)
PROXY_LIST = os.environ.get("PROXY_LIST", "").split(",") if os.environ.get("PROXY_LIST") else []

def get_random_proxy():
    """Get a random proxy from the list to avoid IP bans"""
    if not PROXY_LIST:
        return None
    return random.choice(PROXY_LIST)

def get_random_user_agent():
    """Get a random user agent to avoid detection"""
    return random.choice(USER_AGENTS)

def add_jitter(seconds=1):
    """Add random delay to make requests seem more human-like"""
    jitter = random.uniform(0.1, seconds)
    time.sleep(jitter)

def generate_cache_key(func_name, *args, **kwargs):
    """Generate a cache key based on function name and arguments"""
    key_parts = [func_name]
    key_parts.extend([str(arg) for arg in args])
    key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
    key = "_".join(key_parts)
    return hashlib.md5(key.encode()).hexdigest()

def cached(timeout=CACHE_TIMEOUT):
    """Decorator to cache function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not kwargs.get('bypass_cache', False):
                cache_key = generate_cache_key(func.__name__, *args, **kwargs)
                cached_result = cache.get(cache_key)
                if cached_result:
                    cached_time, result = cached_result
                    if time.time() - cached_time < timeout:
                        return result
            
            result = func(*args, **kwargs)
            
            if not kwargs.get('bypass_cache', False):
                cache_key = generate_cache_key(func.__name__, *args, **kwargs)
                cache[cache_key] = (time.time(), result)
            
            return result
        return wrapper
    return decorator

def clean_ytdl_options():
    """Generate clean ytdlp options to avoid detection"""
    return {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "nocheckcertificate": True,
        "geo_bypass": True,
        "geo_bypass_country": "US",
        "extractor_retries": 5,
        "socket_timeout": 15,
        "extract_flat": "in_playlist",
        "user_agent": get_random_user_agent(),
        "headers": {
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-Fetch-Mode": "navigate",
            "Referer": "https://www.google.com/"
        },
        "http_headers": {
            "User-Agent": get_random_user_agent(),
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-Fetch-Mode": "navigate",
            "Referer": "https://www.google.com/"
        }
    }

def time_to_seconds(time_str):
    """Convert time string to seconds"""
    if not time_str or time_str == "None":
        return 0
    try:
        return sum(int(x) * 60**i for i, x in enumerate(reversed(str(time_str).split(":"))))
    except:
        return 0

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    if not url:
        return None
    
    parsed_url = urlparse(url)
    
    if "youtube.com" in parsed_url.netloc:
        if "/watch" in parsed_url.path:
            return parse_qs(parsed_url.query).get("v", [None])[0]
        elif "/embed/" in parsed_url.path:
            return parsed_url.path.split("/embed/")[1].split("/")[0]
        elif "/v/" in parsed_url.path:
            return parsed_url.path.split("/v/")[1].split("/")[0]
    elif "youtu.be" in parsed_url.netloc:
        return parsed_url.path.lstrip("/")
    
    return None

def is_youtube_url(url):
    """Check if a URL is a valid YouTube URL"""
    if not url:
        return False
    regex = r"(?:youtube\.com|youtu\.be)"
    return re.search(regex, url) is not None

def normalize_url(url, video_id=None):
    """Normalize YouTube URL"""
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    
    if "&" in url:
        url = url.split("&")[0]
    
    return url

def log_api_request(api_key, endpoint, query=None):
    """Log API request to MongoDB"""
    if not db:
        return
    
    try:
        log_data = {
            "api_key": api_key,
            "endpoint": endpoint,
            "query": query,
            "timestamp": datetime.datetime.now(),
            "ip": get_remote_address()
        }
        
        db.logs.insert_one(log_data)
        
        # Update API key usage counter
        if api_key in api_keys:
            api_keys[api_key]["count"] += 1
            
            # Update in MongoDB
            db.api_keys.update_one(
                {"_id": api_key},
                {"$inc": {"count": 1}}
            )
    except Exception as e:
        logger.error(f"Error logging API request: {e}")

def required_api_key(func):
    """Decorator to require API key for routes"""
    @wraps(func)
    def decorated_function(*args, **kwargs):
        api_key = request.args.get('api_key')
        
        # Allow demo API key
        if api_key == "demo_key":
            return func(*args, **kwargs)
        
        # Check if API key exists
        if not api_key or api_key not in api_keys:
            return jsonify({"error": "Invalid API key"}), 401
        
        key_data = api_keys[api_key]
        
        # Check if API key is expired
        if datetime.datetime.now() > key_data.get("valid_until", datetime.datetime.now()):
            return jsonify({"error": "API key expired"}), 401
        
        # Check if daily limit exceeded
        if key_data.get("count", 0) >= key_data.get("daily_limit", 100):
            # Check if it's time to reset the counter
            if datetime.datetime.now() > key_data.get("reset_at", datetime.datetime.now()):
                # Reset counter
                key_data["count"] = 0
                key_data["reset_at"] = datetime.datetime.now() + datetime.timedelta(days=1)
                
                if db:
                    # Update in MongoDB
                    db.api_keys.update_one(
                        {"_id": api_key},
                        {"$set": {"count": 0, "reset_at": key_data["reset_at"]}}
                    )
            else:
                return jsonify({"error": "Daily limit exceeded"}), 429
        
        # Log API request
        log_api_request(api_key, request.path, request.args.get('query') or request.args.get('url'))
        
        return func(*args, **kwargs)
    
    return decorated_function

def required_admin_key(func):
    """Decorator to require admin API key for routes"""
    @wraps(func)
    def decorated_function(*args, **kwargs):
        api_key = request.args.get('admin_key')
        
        # Check if API key exists and is admin
        if not api_key or api_key not in admin_keys:
            return jsonify({"error": "Invalid admin key"}), 401
        
        return func(*args, **kwargs)
    
    return decorated_function

class YouTubeAPIService:
    """Service class to handle YouTube operations"""
    base_url = "https://www.youtube.com/watch?v="
    list_base = "https://youtube.com/playlist?list="
    
    @staticmethod
    async def search_videos(query, limit=10):
        """Search YouTube videos"""
        try:
            add_jitter(0.5)  # Add a small delay
            
            results = VideosSearch(query, limit=limit)
            result_dict = await results.next()
            
            if not result_dict or "result" not in result_dict:
                return []
            
            videos = []
            for result in result_dict["result"]:
                video = {
                    "id": result.get("id", ""),
                    "title": result.get("title", "Unknown"),
                    "duration": time_to_seconds(result.get("duration", "0:00")),
                    "duration_text": result.get("duration", "0:00"),
                    "views": result.get("viewCount", {}).get("text", "0").replace(" views", "").replace(",", ""),
                    "publish_time": result.get("publishedTime", ""),
                    "channel": result.get("channel", {}).get("name", ""),
                    "thumbnail": result.get("thumbnails", [{}])[0].get("url", "").split("?")[0],
                    "link": result.get("link", ""),
                }
                videos.append(video)
            
            return videos
        except Exception as e:
            logger.error(f"Error searching videos: {e}")
            return []
    
    @staticmethod
    async def url_exists(url, video_id=None):
        """Check if a YouTube URL exists"""
        try:
            if video_id:
                url = f"https://www.youtube.com/watch?v={video_id}"
            
            if not is_youtube_url(url):
                return False
            
            # Quick check using oembed endpoint
            async with httpx.AsyncClient(timeout=10) as client:
                try:
                    oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
                    response = await client.get(
                        oembed_url, 
                        headers={"User-Agent": get_random_user_agent()},
                        proxy=get_random_proxy()
                    )
                    return response.status_code == 200
                except:
                    return False
        except Exception as e:
            logger.error(f"Error checking if URL exists: {e}")
            return False
    
    @staticmethod
    @cached()
    async def get_details(url, video_id=None):
        """Get video details"""
        try:
            if video_id:
                url = f"https://www.youtube.com/watch?v={video_id}"
            elif url.isdigit() or (url.startswith("-") and url[1:].isdigit()):
                # If it's just a number, treat it as a search query
                search_results = await YouTubeAPIService.search_videos(url, limit=1)
                if search_results:
                    video = search_results[0]
                    return {
                        "id": video["id"],
                        "title": video["title"],
                        "duration": video["duration"],
                        "duration_text": video["duration_text"],
                        "channel": video["channel"],
                        "views": video["views"],
                        "thumbnail": video["thumbnail"],
                        "link": video["link"]
                    }
                else:
                    raise ValueError(f"No videos found for query: {url}")
            
            url = normalize_url(url)
            add_jitter(0.5)  # Add a small delay
            
            # Try with youtube-search-python first
            try:
                results = VideosSearch(url, limit=1)
                result_dict = await results.next()
                
                if "result" in result_dict and result_dict["result"]:
                    result = result_dict["result"][0]
                    
                    title = result["title"]
                    duration_text = result.get("duration", "0:00")
                    thumbnail = result.get("thumbnails", [{}])[0].get("url", "").split("?")[0]
                    vid_id = result["id"]
                    channel = result.get("channel", {}).get("name", "")
                    views = result.get("viewCount", {}).get("text", "0").replace(" views", "").replace(",", "")
                    
                    duration = time_to_seconds(duration_text)
                    
                    return {
                        "id": vid_id,
                        "title": title,
                        "duration": duration,
                        "duration_text": duration_text,
                        "channel": channel,
                        "views": views,
                        "thumbnail": thumbnail,
                        "link": f"https://www.youtube.com/watch?v={vid_id}"
                    }
            except Exception as e:
                logger.warning(f"Error getting details with VideosSearch: {e}")
            
            # Fallback to yt-dlp
            options = clean_ytdl_options()
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=False)
                
                title = info.get("title", "Unknown")
                duration = info.get("duration", 0)
                
                # Format duration as mm:ss or hh:mm:ss
                if duration > 3600:
                    hours, remainder = divmod(duration, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    duration_text = f"{hours}:{minutes:02d}:{seconds:02d}"
                else:
                    minutes, seconds = divmod(duration, 60)
                    duration_text = f"{minutes}:{seconds:02d}"
                
                thumbnail = info.get("thumbnail", "")
                if isinstance(thumbnail, dict) and "url" in thumbnail:
                    thumbnail = thumbnail["url"]
                
                vid_id = info.get("id", "")
                channel = info.get("channel", "")
                views = info.get("view_count", 0)
                
                return {
                    "id": vid_id,
                    "title": title,
                    "duration": duration,
                    "duration_text": duration_text,
                    "channel": channel,
                    "views": views,
                    "thumbnail": thumbnail,
                    "link": f"https://www.youtube.com/watch?v={vid_id}"
                }
        except Exception as e:
            logger.error(f"Error getting video details: {e}")
            return {
                "id": "",
                "title": "Unknown",
                "duration": 0,
                "duration_text": "0:00",
                "channel": "",
                "views": 0,
                "thumbnail": "",
                "link": ""
            }
    
    @staticmethod
    @cached()
    async def get_title(url, video_id=None):
        """Get video title"""
        details = await YouTubeAPIService.get_details(url, video_id)
        return details.get("title", "Unknown")
    
    @staticmethod
    @cached()
    async def get_duration(url, video_id=None):
        """Get video duration"""
        details = await YouTubeAPIService.get_details(url, video_id)
        return details.get("duration_text", "0:00")
    
    @staticmethod
    @cached()
    async def get_thumbnail(url, video_id=None):
        """Get video thumbnail"""
        details = await YouTubeAPIService.get_details(url, video_id)
        return details.get("thumbnail", "")
    
    @staticmethod
    @cached()
    async def get_stream_url(url, is_video=False, video_id=None):
        """Get stream URL for a video"""
        try:
            if video_id:
                url = f"https://www.youtube.com/watch?v={video_id}"
            elif url.isdigit() or (url.startswith("-") and url[1:].isdigit()):
                # If it's just a number, treat it as a search query
                search_results = await YouTubeAPIService.search_videos(url, limit=1)
                if search_results:
                    url = search_results[0]["link"]
                else:
                    raise ValueError(f"No videos found for query: {url}")
            
            url = normalize_url(url)
            add_jitter(0.3)  # Add a small delay
            
            # Generate a unique stream ID
            stream_id = str(uuid.uuid4())
            
            format_str = "best[height<=720]" if is_video else "bestaudio"
            options = clean_ytdl_options()
            options.update({
                "format": format_str,
                "skip_download": True,
            })
            
            # Use a random proxy if available
            proxy = get_random_proxy()
            if proxy:
                options["proxy"] = proxy
            
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=False)
                best_format = info.get("url", "")
                
                if not best_format:
                    raise ValueError("Could not extract stream URL")
                
                # Store the URL in cache for streaming
                stream_key = f"stream:{stream_id}"
                cache[stream_key] = {
                    "url": best_format,
                    "created_at": time.time(),
                    "is_video": is_video,
                    "info": info
                }
                
                # Return our proxied stream URL
                return f"/stream/{stream_id}"
        except Exception as e:
            logger.error(f"Error getting stream URL: {e}")
            return ""
    
    @staticmethod
    @cached()
    async def get_playlist(url, limit=10, user_id=None, list_id=None):
        """Get playlist videos"""
        try:
            if list_id:
                url = f"https://youtube.com/playlist?list={list_id}"
            
            if "&" in url:
                url = url.split("&")[0]
            
            options = clean_ytdl_options()
            options.update({
                "playlistend": int(limit),
                "extract_flat": True,
                "skip_download": True,
            })
            
            # Use a random proxy if available
            proxy = get_random_proxy()
            if proxy:
                options["proxy"] = proxy
            
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=False)
                entries = info.get("entries", [])
                
                result = []
                for entry in entries:
                    if entry.get("id"):
                        result.append(entry["id"])
                
                return result
        except Exception as e:
            logger.error(f"Error getting playlist: {e}")
            return []
    
    @staticmethod
    @cached()
    async def get_track(url, video_id=None):
        """Get track details"""
        try:
            details = await YouTubeAPIService.get_details(url, video_id)
            
            track_details = {
                "title": details.get("title", "Unknown"),
                "link": details.get("link", ""),
                "vidid": details.get("id", ""),
                "duration_min": details.get("duration_text", "0:00"),
                "thumb": details.get("thumbnail", ""),
            }
            
            return track_details, details.get("id", "")
        except Exception as e:
            logger.error(f"Error getting track: {e}")
            return {"title": "Unknown", "link": "", "vidid": "", "duration_min": "0:00", "thumb": ""}, ""
    
    @staticmethod
    @cached()
    async def get_formats(url, video_id=None):
        """Get available formats for a video"""
        try:
            if video_id:
                url = f"https://www.youtube.com/watch?v={video_id}"
            elif url.isdigit() or (url.startswith("-") and url[1:].isdigit()):
                # If it's just a number, treat it as a search query
                search_results = await YouTubeAPIService.search_videos(url, limit=1)
                if search_results:
                    url = search_results[0]["link"]
                else:
                    raise ValueError(f"No videos found for query: {url}")
            
            url = normalize_url(url)
            
            options = clean_ytdl_options()
            # Use a random proxy if available
            proxy = get_random_proxy()
            if proxy:
                options["proxy"] = proxy
                
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=False)
                
                formats_available = []
                for format in info.get("formats", []):
                    try:
                        format_str = format.get("format", "")
                        if not format_str or "dash" in format_str.lower():
                            continue
                        
                        format_data = {
                            "format": format.get("format", ""),
                            "filesize": format.get("filesize", 0),
                            "format_id": format.get("format_id", ""),
                            "ext": format.get("ext", ""),
                            "format_note": format.get("format_note", ""),
                            "yturl": url,
                        }
                        
                        # Check if all required fields are present
                        if all(format_data.values()):
                            formats_available.append(format_data)
                    except Exception as e:
                        logger.error(f"Error processing format: {e}")
                
                return formats_available, url
        except Exception as e:
            logger.error(f"Error getting formats: {e}")
            return [], ""
    
    @staticmethod
    @cached()
    async def slider(link, query_type, video_id=None):
        """Get related videos for a slider"""
        try:
            if video_id:
                link = f"https://www.youtube.com/watch?v={video_id}"
            
            link = normalize_url(link)
            query_type = int(query_type) if str(query_type).isdigit() else 0
            
            a = VideosSearch(link, limit=query_type + 5)  # Get a few extra in case some fail
            result = (await a.next()).get("result", [])
            
            if not result or query_type >= len(result):
                return "Unknown", "0:00", "", ""
            
            title = result[query_type]["title"]
            duration_min = result[query_type].get("duration", "0:00")
            thumbnail = result[query_type].get("thumbnails", [{}])[0].get("url", "").split("?")[0]
            vidid = result[query_type]["id"]
            
            return title, duration_min, thumbnail, vidid
        except Exception as e:
            logger.error(f"Error getting slider data: {e}")
            return "Unknown", "0:00", "", ""
    
    @staticmethod
    async def download(
        link, 
        video=False, 
        video_id=None, 
        format_id=None, 
        title=None
    ):
        """Download a video or audio file"""
        try:
            if video_id:
                link = f"https://www.youtube.com/watch?v={video_id}"
            elif link.isdigit() or (link.startswith("-") and link[1:].isdigit()):
                # If it's just a number, treat it as a search query
                search_results = await YouTubeAPIService.search_videos(link, limit=1)
                if search_results:
                    link = search_results[0]["link"]
                    if not title:
                        title = search_results[0]["title"]
                else:
                    raise ValueError(f"No videos found for query: {link}")
                    
            link = normalize_url(link)
            
            # Generate a unique download token
            download_token = str(uuid.uuid4())
            
            if not title:
                # Get video details to get the title
                details = await YouTubeAPIService.get_details(link)
                title = details.get("title", "download")
            
            # Clean title for filename
            safe_title = re.sub(r'[^\w\-_\. ]', '_', title)
            safe_title = safe_title[:50]  # Truncate long titles
            
            # Set output path
            output_path = os.path.join(DOWNLOAD_DIR, safe_title)
            
            # Configure yt-dlp options
            options = clean_ytdl_options()
            
            if format_id:
                # Use specific format
                if video:
                    # Format for video
                    options.update({
                        "format": f"{format_id}+bestaudio",
                        "merge_output_format": "mp4",
                    })
                    output_path += ".mp4"
                else:
                    # Format for audio
                    options.update({
                        "format": format_id,
                        "postprocessors": [{
                            "key": "FFmpegExtractAudio",
                            "preferredcodec": "mp3",
                            "preferredquality": "192",
                        }],
                    })
                    output_path += ".mp3"
            else:
                # Use best format
                if video:
                    options.update({
                        "format": "(bestvideo[height<=?720][width<=?1280][ext=mp4])+(bestaudio[ext=m4a])",
                        "merge_output_format": "mp4",
                    })
                    output_path += ".mp4"
                else:
                    options.update({
                        "format": "bestaudio/best",
                        "postprocessors": [{
                            "key": "FFmpegExtractAudio",
                            "preferredcodec": "mp3",
                            "preferredquality": "192",
                        }],
                    })
                    output_path += ".mp3"
            
            # Set output template
            options["outtmpl"] = output_path
            
            # Use a random proxy if available
            proxy = get_random_proxy()
            if proxy:
                options["proxy"] = proxy
            
            # Download the file
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([link])
            
            # Check if file exists
            if video:
                final_path = output_path
                if not os.path.exists(final_path):
                    raise FileNotFoundError(f"Downloaded file not found: {final_path}")
            else:
                # For audio, the extension is changed by FFmpeg
                final_path = os.path.splitext(output_path)[0] + ".mp3"
                if not os.path.exists(final_path):
                    raise FileNotFoundError(f"Downloaded file not found: {final_path}")
            
            # Store download info
            cache[f"download:{download_token}"] = {
                "path": final_path,
                "created_at": time.time(),
                "is_video": video
            }
            
            # Return download token
            return f"/download/{download_token}", None
        except Exception as e:
            logger.error(f"Error downloading: {e}")
            return "", str(e)

def run_async(coro):
    """Run an async function from a synchronous context"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

# Admin API Routes
@app.route("/admin/create_api_key", methods=["POST"])
@required_admin_key
def create_api_key():
    """Create a new API key"""
    name = request.json.get("name", "User")
    days_valid = int(request.json.get("days_valid", 30))
    daily_limit = int(request.json.get("daily_limit", 100))
    is_admin = request.json.get("is_admin", False)
    
    # Generate a new API key
    api_key = secrets.token_hex(16)
    
    # Set expiration date
    valid_until = datetime.datetime.now() + datetime.timedelta(days=days_valid)
    reset_at = datetime.datetime.now() + datetime.timedelta(days=1)
    
    # Create key data
    key_data = {
        "name": name,
        "is_admin": is_admin,
        "created_at": datetime.datetime.now(),
        "valid_until": valid_until,
        "daily_limit": daily_limit,
        "reset_at": reset_at,
        "count": 0
    }
    
    # Store in memory
    api_keys[api_key] = key_data
    
    if is_admin:
        admin_keys[api_key] = key_data
    
    # Store in MongoDB if available
    if db:
        try:
            db.api_keys.insert_one({
                "_id": api_key,
                **key_data
            })
        except Exception as e:
            logger.error(f"Error storing API key in MongoDB: {e}")
    
    return jsonify({
        "api_key": api_key,
        "name": name,
        "valid_until": valid_until.isoformat(),
        "daily_limit": daily_limit,
        "is_admin": is_admin
    })

@app.route("/admin/list_api_keys", methods=["GET"])
@required_admin_key
def list_api_keys():
    """List all API keys"""
    keys_list = []
    
    for key, data in api_keys.items():
        keys_list.append({
            "api_key": key,
            "name": data.get("name", "User"),
            "valid_until": data.get("valid_until", datetime.datetime.now()).isoformat(),
            "daily_limit": data.get("daily_limit", 100),
            "count": data.get("count", 0),
            "is_admin": data.get("is_admin", False)
        })
    
    return jsonify(keys_list)

@app.route("/admin/revoke_api_key", methods=["POST"])
@required_admin_key
def revoke_api_key():
    """Revoke an API key"""
    api_key = request.json.get("api_key")
    
    if not api_key or api_key not in api_keys:
        return jsonify({"error": "Invalid API key"}), 400
    
    # Remove from memory
    is_admin = api_keys[api_key].get("is_admin", False)
    del api_keys[api_key]
    
    if is_admin and api_key in admin_keys:
        del admin_keys[api_key]
    
    # Remove from MongoDB if available
    if db:
        try:
            db.api_keys.delete_one({"_id": api_key})
        except Exception as e:
            logger.error(f"Error removing API key from MongoDB: {e}")
    
    return jsonify({"success": True, "message": "API key revoked"})

# API Routes
@app.route("/", methods=["GET"])
def index():
    """Home page with API documentation"""
    return render_index()

@app.route("/youtube", methods=["GET"])
@limiter.limit(API_RATE_LIMIT)
@required_api_key
def youtube():
    """Main YouTube endpoint that supports both search and direct video access"""
    query = request.args.get("query")
    video = request.args.get("video", "false").lower() == "true"
    
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    # Determine if this is a search query or a direct video ID/URL
    is_url = is_youtube_url(query)
    is_video_id = re.match(r'^[a-zA-Z0-9_-]{11}$', query)
    
    try:
        # Handle search case
        if not is_url and not is_video_id:
            # Search for videos
            search_results = run_async(YouTubeAPIService.search_videos(query, limit=1))
            
            if not search_results:
                return jsonify({"error": "No videos found"}), 404
            
            video_data = search_results[0]
            stream_url = run_async(YouTubeAPIService.get_stream_url(video_data["link"], is_video=video))
            
            response = {
                "id": video_data["id"],
                "title": video_data["title"],
                "duration": video_data["duration"],
                "link": video_data["link"],
                "channel": video_data["channel"],
                "views": video_data["views"],
                "thumbnail": video_data["thumbnail"],
                "stream_url": request.host_url.rstrip("/") + stream_url,
                "stream_type": "Video" if video else "Audio"
            }
            
            return jsonify(response)
        
        # Handle direct video case
        video_url = query if is_url else f"https://www.youtube.com/watch?v={query}"
        video_details = run_async(YouTubeAPIService.get_details(video_url))
        stream_url = run_async(YouTubeAPIService.get_stream_url(video_url, is_video=video))
        
        response = {
            "id": video_details["id"],
            "title": video_details["title"],
            "duration": video_details["duration"],
            "link": video_details["link"],
            "channel": video_details["channel"],
            "views": video_details["views"],
            "thumbnail": video_details["thumbnail"],
            "stream_url": request.host_url.rstrip("/") + stream_url,
            "stream_type": "Video" if video else "Audio"
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in YouTube endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/search", methods=["GET"])
@limiter.limit(RATE_LIMIT)
@required_api_key
def search():
    """Search YouTube videos"""
    query = request.args.get("query")
    limit = int(request.args.get("limit", 10))
    
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    # Limit the maximum number of results to prevent abuse
    if limit > 20:
        limit = 20
    
    results = run_async(YouTubeAPIService.search_videos(query, limit))
    
    return jsonify({"results": results})

@app.route("/api/exists", methods=["GET"])
@limiter.limit(RATE_LIMIT)
@required_api_key
def check_exists():
    """Check if a YouTube URL exists"""
    url = request.args.get("url")
    video_id = request.args.get("video_id")
    
    if not url and not video_id:
        return jsonify({"error": "URL or video_id parameter is required"}), 400
    
    exists = run_async(YouTubeAPIService.url_exists(url, video_id))
    
    return jsonify({"exists": exists})

@app.route("/api/details", methods=["GET"])
@limiter.limit(RATE_LIMIT)
@required_api_key
def get_details():
    """Get video details"""
    url = request.args.get("url")
    video_id = request.args.get("video_id")
    
    if not url and not video_id:
        return jsonify({"error": "URL or video_id parameter is required"}), 400
    
    details = run_async(YouTubeAPIService.get_details(url, video_id))
    
    return jsonify(details)

@app.route("/api/title", methods=["GET"])
@limiter.limit(RATE_LIMIT)
@required_api_key
def get_title():
    """Get video title"""
    url = request.args.get("url")
    video_id = request.args.get("video_id")
    
    if not url and not video_id:
        return jsonify({"error": "URL or video_id parameter is required"}), 400
    
    title = run_async(YouTubeAPIService.get_title(url, video_id))
    
    return jsonify({"title": title})

@app.route("/api/duration", methods=["GET"])
@limiter.limit(RATE_LIMIT)
@required_api_key
def get_duration():
    """Get video duration"""
    url = request.args.get("url")
    video_id = request.args.get("video_id")
    
    if not url and not video_id:
        return jsonify({"error": "URL or video_id parameter is required"}), 400
    
    duration = run_async(YouTubeAPIService.get_duration(url, video_id))
    
    return jsonify({"duration": duration})

@app.route("/api/thumbnail", methods=["GET"])
@limiter.limit(RATE_LIMIT)
@required_api_key
def get_thumbnail():
    """Get video thumbnail"""
    url = request.args.get("url")
    video_id = request.args.get("video_id")
    
    if not url and not video_id:
        return jsonify({"error": "URL or video_id parameter is required"}), 400
    
    thumbnail = run_async(YouTubeAPIService.get_thumbnail(url, video_id))
    
    return jsonify({"thumbnail": thumbnail})

@app.route("/api/stream", methods=["GET"])
@limiter.limit(RATE_LIMIT)
@required_api_key
def get_stream_url():
    """Get stream URL for a video"""
    url = request.args.get("url")
    video_id = request.args.get("video_id")
    is_video = request.args.get("video", "false").lower() == "true"
    
    if not url and not video_id:
        return jsonify({"error": "URL or video_id parameter is required"}), 400
    
    stream_url = run_async(YouTubeAPIService.get_stream_url(url, is_video, video_id))
    
    if not stream_url:
        return jsonify({"error": "Failed to get stream URL"}), 500
    
    return jsonify({
        "stream_url": request.host_url.rstrip("/") + stream_url,
        "stream_type": "Video" if is_video else "Audio"
    })

@app.route("/stream/<stream_id>", methods=["GET"])
def stream_media(stream_id):
    """Stream media from YouTube"""
    stream_key = f"stream:{stream_id}"
    stream_data = cache.get(stream_key)
    
    if not stream_data:
        return jsonify({"error": "Stream not found or expired"}), 404
    
    url = stream_data.get("url")
    is_video = stream_data.get("is_video", False)
    
    if not url:
        return jsonify({"error": "Invalid stream URL"}), 500
    
    # Set appropriate content type
    content_type = "video/mp4" if is_video else "audio/mp4"
    
    def generate():
        try:
            # Buffer size
            buffer_size = 1024 * 1024  # 1MB
            
            # Create a streaming session with appropriate headers
            headers = {
                "User-Agent": get_random_user_agent(),
                "Range": request.headers.get("Range", "bytes=0-")
            }
            
            with httpx.stream("GET", url, headers=headers, timeout=30) as response:
                # Forward content type and other headers
                yield b""
                
                # Stream the content
                for chunk in response.iter_bytes(chunk_size=buffer_size):
                    yield chunk
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield b""
    
    # Create a streaming response
    return Response(
        stream_with_context(generate()),
        content_type=content_type,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache"
        }
    )

@app.route("/api/playlist", methods=["GET"])
@limiter.limit(RATE_LIMIT)
@required_api_key
def get_playlist():
    """Get playlist videos"""
    url = request.args.get("url")
    list_id = request.args.get("list_id")
    limit = int(request.args.get("limit", 10))
    
    if not url and not list_id:
        return jsonify({"error": "URL or list_id parameter is required"}), 400
    
    # Limit the maximum number of videos to prevent abuse
    if limit > 50:
        limit = 50
    
    videos = run_async(YouTubeAPIService.get_playlist(url, limit, None, list_id))
    
    return jsonify({"videos": videos})

@app.route("/api/track", methods=["GET"])
@limiter.limit(RATE_LIMIT)
@required_api_key
def get_track():
    """Get track details"""
    url = request.args.get("url")
    video_id = request.args.get("video_id")
    
    if not url and not video_id:
        return jsonify({"error": "URL or video_id parameter is required"}), 400
    
    track, vid_id = run_async(YouTubeAPIService.get_track(url, video_id))
    
    return jsonify({"track": track, "vidid": vid_id})

@app.route("/api/formats", methods=["GET"])
@limiter.limit(RATE_LIMIT)
@required_api_key
def get_formats():
    """Get available formats for a video"""
    url = request.args.get("url")
    video_id = request.args.get("video_id")
    
    if not url and not video_id:
        return jsonify({"error": "URL or video_id parameter is required"}), 400
    
    formats, yturl = run_async(YouTubeAPIService.get_formats(url, video_id))
    
    return jsonify({"formats": formats, "yturl": yturl})

@app.route("/api/slider", methods=["GET"])
@limiter.limit(RATE_LIMIT)
@required_api_key
def get_slider():
    """Get related videos for a slider"""
    url = request.args.get("url")
    query_type = request.args.get("query_type", 0)
    video_id = request.args.get("video_id")
    
    if not url and not video_id:
        return jsonify({"error": "URL or video_id parameter is required"}), 400
    
    title, duration, thumbnail, vidid = run_async(YouTubeAPIService.slider(url, query_type, video_id))
    
    return jsonify({
        "title": title,
        "duration": duration,
        "thumbnail": thumbnail,
        "vidid": vidid
    })

@app.route("/api/download", methods=["GET"])
@limiter.limit(RATE_LIMIT)
@required_api_key
def download_media():
    """Download a video or audio file"""
    url = request.args.get("url")
    video_id = request.args.get("video_id")
    is_video = request.args.get("video", "false").lower() == "true"
    format_id = request.args.get("format_id")
    title = request.args.get("title")
    
    if not url and not video_id:
        return jsonify({"error": "URL or video_id parameter is required"}), 400
    
    download_url, error = run_async(YouTubeAPIService.download(url, is_video, video_id, format_id, title))
    
    if error:
        return jsonify({"error": error}), 500
    
    if not download_url:
        return jsonify({"error": "Failed to download media"}), 500
    
    return jsonify({
        "download_url": request.host_url.rstrip("/") + download_url,
        "type": "video" if is_video else "audio"
    })

@app.route("/download/<token>", methods=["GET"])
def serve_download(token):
    """Serve a downloaded file"""
    download_key = f"download:{token}"
    download_data = cache.get(download_key)
    
    if not download_data:
        return jsonify({"error": "Download not found or expired"}), 404
    
    # Get the file path
    file_path = download_data.get("path")
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    # Determine file type and name
    is_video = file_path.endswith(".mp4")
    filename = os.path.basename(file_path)
    
    try:
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype="video/mp4" if is_video else "audio/mp3"
        )
    except Exception as e:
        logger.error(f"Error serving download: {e}")
        return jsonify({"error": "Failed to serve file"}), 500

@app.route("/api/cleanup", methods=["POST"])
@required_admin_key
def cleanup_old_files():
    """Clean up old cache entries and downloaded files"""
    try:
        # Expire time (1 day)
        expire_time = time.time() - (24 * 60 * 60)
        
        # Clean up cache
        keys_to_remove = []
        for key, value in cache.items():
            if isinstance(value, tuple) and len(value) > 0 and isinstance(value[0], (int, float)):
                timestamp, _ = value
                if timestamp < expire_time:
                    keys_to_remove.append(key)
            elif isinstance(value, dict) and "created_at" in value:
                if value["created_at"] < expire_time:
                    keys_to_remove.append(key)
                    
                    # If it's a download, remove the file
                    if key.startswith("download:") and "path" in value:
                        try:
                            if os.path.exists(value["path"]):
                                os.remove(value["path"])
                        except Exception as e:
                            logger.error(f"Error removing file: {e}")
        
        for key in keys_to_remove:
            cache.pop(key, None)
        
        # Clean up download directory
        for filename in os.listdir(DOWNLOAD_DIR):
            filepath = os.path.join(DOWNLOAD_DIR, filename)
            try:
                # If file is older than 1 day
                if os.path.getmtime(filepath) < expire_time:
                    if os.path.isfile(filepath):
                        os.remove(filepath)
            except Exception as e:
                logger.error(f"Error removing old file: {e}")
        
        return jsonify({
            "success": True,
            "message": f"Cleaned up {len(keys_to_remove)} cache entries and old downloads"
        })
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return jsonify({"error": str(e)}), 500

# Add cleanup job
def run_cleanup_periodically():
    """Run cleanup job periodically"""
    while True:
        time.sleep(60 * 60)  # Run every hour
        try:
            # Clean up old files
            cleanup_old_files()
        except Exception as e:
            logger.error(f"Error in cleanup job: {e}")

# Error handlers
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Rate limit exceeded",
        "message": str(e.description)
    }), 429

@app.errorhandler(500)
def server_error_handler(e):
    return jsonify({
        "error": "Server error",
        "message": str(e)
    }), 500

def render_index():
    """Render the HTML for the index page"""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YouTube API Service - Anti-Bot Protection</title>
    <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet" crossorigin="anonymous">
    <style>
        body {
            padding-top: 20px;
        }
        .header {
            padding: 2rem 0;
            text-align: center;
        }
        .endpoint {
            background-color: var(--bs-dark);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .method {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 4px;
            margin-right: 10px;
            font-weight: bold;
        }
        .get {
            background-color: var(--bs-info);
            color: var(--bs-dark);
        }
        .post {
            background-color: var(--bs-success);
            color: var(--bs-dark);
        }
        .example {
            background-color: var(--bs-secondary);
            border-radius: 4px;
            padding: 10px;
            margin-top: 10px;
        }
        .features-list {
            list-style-type: none;
            padding-left: 0;
        }
        .features-list li {
            margin-bottom: 8px;
            display: flex;
            align-items: center;
        }
        .features-list li::before {
            content: "📟";
            margin-right: 10px;
        }
        .demo-section {
            background-color: var(--bs-dark);
            border-radius: 8px;
            padding: 20px;
            margin-top: 30px;
        }
        .admin-section {
            background-color: var(--bs-dark);
            border-radius: 8px;
            padding: 20px;
            margin-top: 30px;
        }
        .credit {
            text-align: center;
            margin-top: 2rem;
            margin-bottom: 2rem;
            font-size: 0.9rem;
            opacity: 0.7;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>YouTube API Service</h1>
            <p class="lead">Fast, reliable YouTube API with anti-bot protection</p>
            <div class="badge bg-success">Version 1.0.0</div>
        </div>

        <div class="row">
            <div class="col-md-8">
                <h2>API Documentation</h2>
                <p>This API provides access to YouTube content while avoiding bot detection mechanisms.</p>
                
                <div class="endpoint">
                    <h3><span class="method get">GET</span> /youtube</h3>
                    <p>Main endpoint to search or get video information</p>
                    <h4>Parameters:</h4>
                    <ul>
                        <li><code>query</code> - YouTube URL, video ID, or search term</li>
                        <li><code>video</code> - Boolean to get video stream (default: false)</li>
                        <li><code>api_key</code> - Your API key (or use "demo_key" for testing)</li>
                    </ul>
                    <div class="example">
                        <h5>Example:</h5>
                        <pre>/youtube?query=dQw4w9WgXcQ&video=false&api_key=demo_key</pre>
                    </div>
                </div>
                
                <div class="endpoint">
                    <h3><span class="method get">GET</span> /api/search</h3>
                    <p>Search for YouTube videos</p>
                    <h4>Parameters:</h4>
                    <ul>
                        <li><code>query</code> - Search term</li>
                        <li><code>limit</code> - Maximum number of results (default: 10, max: 20)</li>
                        <li><code>api_key</code> - Your API key</li>
                    </ul>
                    <div class="example">
                        <h5>Example:</h5>
                        <pre>/api/search?query=rickroll&limit=5&api_key=demo_key</pre>
                    </div>
                </div>
                
                <div class="endpoint">
                    <h3><span class="method get">GET</span> /api/details</h3>
                    <p>Get video details including title, duration, and thumbnail</p>
                    <h4>Parameters:</h4>
                    <ul>
                        <li><code>url</code> - YouTube URL</li>
                        <li><code>video_id</code> - YouTube video ID (alternative to URL)</li>
                        <li><code>api_key</code> - Your API key</li>
                    </ul>
                    <div class="example">
                        <h5>Example:</h5>
                        <pre>/api/details?url=https://www.youtube.com/watch?v=dQw4w9WgXcQ&api_key=demo_key</pre>
                    </div>
                </div>
                
                <div class="endpoint">
                    <h3><span class="method get">GET</span> /api/stream</h3>
                    <p>Get stream URL for a video or audio</p>
                    <h4>Parameters:</h4>
                    <ul>
                        <li><code>url</code> - YouTube URL</li>
                        <li><code>video_id</code> - YouTube video ID (alternative to URL)</li>
                        <li><code>video</code> - Boolean to get video stream (default: false)</li>
                        <li><code>api_key</code> - Your API key</li>
                    </ul>
                    <div class="example">
                        <h5>Example:</h5>
                        <pre>/api/stream?url=https://www.youtube.com/watch?v=dQw4w9WgXcQ&video=true&api_key=demo_key</pre>
                    </div>
                </div>
                
                <div class="endpoint">
                    <h3><span class="method get">GET</span> /api/download</h3>
                    <p>Download a video or audio file</p>
                    <h4>Parameters:</h4>
                    <ul>
                        <li><code>url</code> - YouTube URL</li>
                        <li><code>video_id</code> - YouTube video ID (alternative to URL)</li>
                        <li><code>video</code> - Boolean to download video (default: false)</li>
                        <li><code>format_id</code> - Optional format ID</li>
                        <li><code>title</code> - Optional custom title for the file</li>
                        <li><code>api_key</code> - Your API key</li>
                    </ul>
                    <div class="example">
                        <h5>Example:</h5>
                        <pre>/api/download?url=https://www.youtube.com/watch?v=dQw4w9WgXcQ&video=false&api_key=demo_key</pre>
                    </div>
                </div>
                
                <div class="endpoint">
                    <h3><span class="method get">GET</span> /api/playlist</h3>
                    <p>Get videos from a playlist</p>
                    <h4>Parameters:</h4>
                    <ul>
                        <li><code>url</code> - Playlist URL</li>
                        <li><code>list_id</code> - Playlist ID (alternative to URL)</li>
                        <li><code>limit</code> - Maximum number of videos (default: 10, max: 50)</li>
                        <li><code>api_key</code> - Your API key</li>
                    </ul>
                    <div class="example">
                        <h5>Example:</h5>
                        <pre>/api/playlist?url=https://www.youtube.com/playlist?list=PLFgquLnL59alW3xmYiWRaoz0oM3H17Lth&limit=5&api_key=demo_key</pre>
                    </div>
                </div>
            </div>
            
            <div class="col-md-4">
                <div class="card bg-dark">
                    <div class="card-body">
                        <h3 class="card-title">Features</h3>
                        <ul class="features-list">
                            <li>Ultra-fast search & play (0.5s response time)</li>
                            <li>Seamless audio/video streaming</li>
                            <li>Live stream support</li>
                            <li>No cookies, no headaches</li>
                            <li>Play anything — with no limits!</li>
                        </ul>
                        
                        <h4 class="mt-4">Optimized for</h4>
                        <ul class="features-list">
                            <li>Pyrogram, Telethon, TGCalls bots</li>
                            <li>PyTube & YTDl-free engine</li>
                            <li>Stable performance with 24/7 uptime</li>
                        </ul>
                    </div>
                </div>
                
                <div class="demo-section mt-4">
                    <h3>Try it out</h3>
                    <div class="mb-3">
                        <label for="demoUrl" class="form-label">YouTube URL or Search Term:</label>
                        <input type="text" class="form-control" id="demoUrl" placeholder="Enter URL or search term">
                    </div>
                    <div class="mb-3 form-check">
                        <input type="checkbox" class="form-check-input" id="demoVideo">
                        <label class="form-check-label" for="demoVideo">Get video (instead of audio)</label>
                    </div>
                    <div class="mb-3">
                        <label for="demoEndpoint" class="form-label">Endpoint:</label>
                        <select class="form-select" id="demoEndpoint">
                            <option value="/youtube">YouTube</option>
                            <option value="/api/search">Search</option>
                            <option value="/api/details">Get Details</option>
                            <option value="/api/stream">Get Stream URL</option>
                        </select>
                    </div>
                    <button type="button" class="btn btn-primary" id="demoButton">Test API</button>
                    
                    <div class="mt-4" id="resultContainer" style="display: none;">
                        <h4>Result:</h4>
                        <pre id="resultPre" class="bg-dark text-light p-3" style="overflow-x: auto;"></pre>
                    </div>
                </div>
                
                <div class="admin-section mt-4">
                    <h3>Admin Panel</h3>
                    <p>Manage API keys for your friends</p>
                    <div class="mb-3">
                        <label for="adminKey" class="form-label">Admin Key:</label>
                        <input type="password" class="form-control" id="adminKey" placeholder="Enter admin key">
                    </div>
                    <button type="button" class="btn btn-primary" id="adminLogin">Login</button>
                    
                    <div id="adminPanel" style="display: none;" class="mt-3">
                        <h4>Create API Key</h4>
                        <div class="mb-3">
                            <label for="keyName" class="form-label">Name:</label>
                            <input type="text" class="form-control" id="keyName" placeholder="Friend's name">
                        </div>
                        <div class="mb-3">
                            <label for="keyDays" class="form-label">Valid for (days):</label>
                            <input type="number" class="form-control" id="keyDays" value="30">
                        </div>
                        <div class="mb-3">
                            <label for="keyLimit" class="form-label">Daily request limit:</label>
                            <input type="number" class="form-control" id="keyLimit" value="100">
                        </div>
                        <button type="button" class="btn btn-success" id="createKey">Create Key</button>
                        
                        <h4 class="mt-4">API Keys</h4>
                        <div id="keysList" class="mt-3">
                            <div class="text-center text-muted">
                                <em>Login to view keys</em>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="credit">
            <p>Developed by <a href="https://t.me/INNOCENT_FUCKER" target="_blank">@INNOCENT_FUCKER</a></p>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const demoButton = document.getElementById('demoButton');
            const resultContainer = document.getElementById('resultContainer');
            const resultPre = document.getElementById('resultPre');
            const adminLogin = document.getElementById('adminLogin');
            const adminPanel = document.getElementById('adminPanel');
            const createKey = document.getElementById('createKey');
            const keysList = document.getElementById('keysList');
            
            // Demo API Testing
            demoButton.addEventListener('click', function() {
                const url = document.getElementById('demoUrl').value.trim();
                const isVideo = document.getElementById('demoVideo').checked;
                const endpoint = document.getElementById('demoEndpoint').value;
                
                if (!url) {
                    alert('Please enter a YouTube URL or search term');
                    return;
                }
                
                let apiUrl = endpoint;
                
                // Add appropriate parameters based on endpoint
                if (endpoint === '/api/search') {
                    apiUrl += '?query=' + encodeURIComponent(url) + '&limit=5&api_key=demo_key';
                } else if (endpoint === '/youtube') {
                    apiUrl += '?query=' + encodeURIComponent(url) + '&video=' + isVideo + '&api_key=demo_key';
                } else {
                    // Check if it looks like a URL or video ID
                    if (url.includes('youtube.com') || url.includes('youtu.be')) {
                        apiUrl += '?url=' + encodeURIComponent(url);
                    } else {
                        apiUrl += '?video_id=' + encodeURIComponent(url);
                    }
                    
                    // Add video parameter for stream endpoint
                    if (endpoint === '/api/stream') {
                        apiUrl += '&video=' + isVideo + '&api_key=demo_key';
                    } else {
                        apiUrl += '&api_key=demo_key';
                    }
                }
                
                // Make API request
                fetch(apiUrl)
                    .then(response => response.json())
                    .then(data => {
                        resultPre.textContent = JSON.stringify(data, null, 2);
                        resultContainer.style.display = 'block';
                    })
                    .catch(error => {
                        resultPre.textContent = 'Error: ' + error;
                        resultContainer.style.display = 'block';
                    });
            });
            
            // Admin Panel
            adminLogin.addEventListener('click', function() {
                const adminKey = document.getElementById('adminKey').value.trim();
                
                if (!adminKey) {
                    alert('Please enter an admin key');
                    return;
                }
                
                // Fetch API keys
                fetch('/admin/list_api_keys?admin_key=' + adminKey)
                    .then(response => {
                        if (!response.ok) {
                            throw new Error('Invalid admin key');
                        }
                        return response.json();
                    })
                    .then(data => {
                        adminPanel.style.display = 'block';
                        renderKeysList(data, adminKey);
                    })
                    .catch(error => {
                        alert('Error: ' + error.message);
                    });
            });
            
            // Create API Key
            createKey.addEventListener('click', function() {
                const adminKey = document.getElementById('adminKey').value.trim();
                const keyName = document.getElementById('keyName').value.trim();
                const keyDays = document.getElementById('keyDays').value;
                const keyLimit = document.getElementById('keyLimit').value;
                
                if (!adminKey || !keyName) {
                    alert('Please enter all required fields');
                    return;
                }
                
                // Create API key
                fetch('/admin/create_api_key?admin_key=' + adminKey, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        name: keyName,
                        days_valid: parseInt(keyDays),
                        daily_limit: parseInt(keyLimit),
                        is_admin: false
                    })
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Failed to create API key');
                    }
                    return response.json();
                })
                .then(data => {
                    alert('API key created: ' + data.api_key);
                    
                    // Refresh keys list
                    fetch('/admin/list_api_keys?admin_key=' + adminKey)
                        .then(response => response.json())
                        .then(data => {
                            renderKeysList(data, adminKey);
                        });
                })
                .catch(error => {
                    alert('Error: ' + error.message);
                });
            });
            
            // Render keys list
            function renderKeysList(keys, adminKey) {
                if (!keys || keys.length === 0) {
                    keysList.innerHTML = '<div class="text-center text-muted"><em>No API keys found</em></div>';
                    return;
                }
                
                let html = '';
                keys.forEach(key => {
                    const validUntil = new Date(key.valid_until).toLocaleDateString();
                    html += `
                        <div class="card mb-2">
                            <div class="card-body">
                                <h5 class="card-title">${key.name}</h5>
                                <p class="card-text">
                                    <strong>API Key:</strong> ${key.api_key}<br>
                                    <strong>Valid Until:</strong> ${validUntil}<br>
                                    <strong>Daily Limit:</strong> ${key.daily_limit}<br>
                                    <strong>Usage:</strong> ${key.count} requests
                                </p>
                                <button class="btn btn-sm btn-danger revoke-btn" data-key="${key.api_key}" data-admin="${adminKey}">Revoke</button>
                            </div>
                        </div>
                    `;
                });
                
                keysList.innerHTML = html;
                
                // Add event listeners to revoke buttons
                document.querySelectorAll('.revoke-btn').forEach(button => {
                    button.addEventListener('click', function(e) {
                        const apiKey = e.target.getAttribute('data-key');
                        const adminKey = e.target.getAttribute('data-admin');
                        
                        if (confirm('Are you sure you want to revoke this API key: ' + apiKey + '?')) {
                            fetch('/admin/revoke_api_key?admin_key=' + adminKey, {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json'
                                },
                                body: JSON.stringify({
                                    api_key: apiKey
                                })
                            })
                            .then(response => {
                                if (!response.ok) {
                                    throw new Error('Failed to revoke API key');
                                }
                                return response.json();
                            })
                            .then(data => {
                                alert('API key revoked successfully');
                                
                                // Refresh keys list
                                fetch('/admin/list_api_keys?admin_key=' + adminKey)
                                    .then(response => response.json())
                                    .then(data => {
                                        renderKeysList(data, adminKey);
                                    });
                            })
                            .catch(error => {
                                alert('Error: ' + error.message);
                            });
                        }
                    });
                });
            }
        });
    </script>
</body>
</html>"""

# Start background cleanup job
if __name__ == "__main__":
    # Start cleanup thread
    import threading
    cleanup_thread = threading.Thread(target=run_cleanup_periodically, daemon=True)
    cleanup_thread.start()
    
    # Run Flask app
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)