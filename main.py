import asyncio
import base64
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
import yt_dlp
from flask import Flask, Response, jsonify, request, send_file, stream_with_context, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_sqlalchemy import SQLAlchemy
from youtubesearchpython.__future__ import VideosSearch

# Import models
from models import db, init_db, ApiKey, ApiLog

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

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize database
init_db(app)

# Initialize rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[RATE_LIMIT],
    storage_uri=os.environ.get("REDIS_URL", "memory://"),
    strategy="fixed-window",
)

# In-memory cache
cache = {}

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
    jitter = random.uniform(0.1, int(seconds))
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
    
    # YouTube URL patterns
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)([^&\n?#]+)',
        r'youtube\.com/watch.*?v=([^&\n?#]+)',
        r'youtube\.com/shorts/([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
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

def log_api_request(api_key_str, endpoint, query=None, status=200):
    """Log API request to database"""
    try:
        # Find the API key in the database
        api_key = ApiKey.query.filter_by(key=api_key_str).first()
        
        if api_key:
            # Update the usage counter
            api_key.count += 1
            
            # Reset counter if it's past reset time
            if datetime.datetime.now() > api_key.reset_at:
                api_key.count = 1
                api_key.reset_at = datetime.datetime.now() + datetime.timedelta(days=1)
            
            # Create log entry
            log = ApiLog(
                api_key_id=api_key.id,
                endpoint=endpoint,
                query=query,
                ip_address=get_remote_address(),
                timestamp=datetime.datetime.now(),
                response_status=status
            )
            
            db.session.add(log)
            db.session.commit()
    except Exception as e:
        logger.error(f"Error logging API request: {e}")
        db.session.rollback()

def required_api_key(func):
    """Decorator to require API key for routes"""
    @wraps(func)
    def decorated_function(*args, **kwargs):
        api_key_str = request.args.get('api_key')
        
        # Get the API key from database
        api_key = ApiKey.query.filter_by(key=api_key_str).first()
        
        # Check if API key exists
        if not api_key:
            return jsonify({"error": "Invalid API key"}), 401
        
        # Check if API key is expired
        if api_key.is_expired:
            return jsonify({"error": "API key expired"}), 401
        
        # Check if daily limit exceeded
        if api_key.count >= api_key.daily_limit:
            # Check if it's time to reset the counter
            if datetime.datetime.now() > api_key.reset_at:
                # Reset counter
                api_key.count = 0
                api_key.reset_at = datetime.datetime.now() + datetime.timedelta(days=1)
                db.session.commit()
            else:
                return jsonify({"error": "Daily limit exceeded"}), 429
        
        try:
            # Execute the function
            response = func(*args, **kwargs)
            
            # Log the successful request
            log_api_request(api_key_str, request.path, request.args.get('query'), 
                            response[1] if isinstance(response, tuple) else 200)
            
            return response
        except Exception as e:
            # Log the failed request
            log_api_request(api_key_str, request.path, request.args.get('query'), 500)
            raise e
    
    return decorated_function

def required_admin_key(func):
    """Decorator to require admin API key for routes"""
    @wraps(func)
    def decorated_function(*args, **kwargs):
        api_key_str = request.args.get('admin_key')
        
        # Get the API key from database
        api_key = ApiKey.query.filter_by(key=api_key_str, is_admin=True).first()
        
        # Check if API key exists and is admin
        if not api_key:
            return jsonify({"error": "Invalid admin key"}), 401
        
        return func(*args, **kwargs)
    
    return decorated_function

class YouTubeAPIService:
    """Service class to handle YouTube operations"""
    base_url = "https://www.youtube.com/watch?v="
    list_base = "https://youtube.com/playlist?list="
    
    @staticmethod
    async def search_videos(query, limit=1):
        """Search YouTube videos"""
        try:
            add_jitter(1)  # Add a small delay
            
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
                        headers={"User-Agent": get_random_user_agent()}
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
            elif url.isdigit() or (url.startswith("-") and url[1:].isdigit()) or not is_youtube_url(url):
                # If it's just a number or not a YouTube URL, treat it as a search query
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
            add_jitter(0.2)  # Add a small delay
            
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
            elif url.isdigit() or (url.startswith("-") and url[1:].isdigit()) or not is_youtube_url(url):
                # If it's just a number or not a YouTube URL, treat it as a search query
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
            elif url.isdigit() or (url.startswith("-") and url[1:].isdigit()) or not is_youtube_url(url):
                # If it's just a number or not a YouTube URL, treat it as a search query
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
            elif link.isdigit() or (link.startswith("-") and link[1:].isdigit()) or not is_youtube_url(link):
                # If it's just a number or not a YouTube URL, treat it as a search query
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

# Main Routes
@app.route("/", methods=["GET"])
def index():
    """Home page with API documentation"""
    return render_template("index.html")

@app.route("/admin", methods=["GET"])
@required_admin_key
def admin_panel():
    """Admin panel for managing API keys"""
    return render_template("admin.html")

@app.route("/youtube", methods=["GET"])
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
        
        # Format response to match exactly the requested format
        response = {
            "id": video_details["id"],
            "title": video_details["title"],
            "duration": video_details["duration"],
            "link": video_details["link"],
            "channel": video_details["channel"],
            "views": int(video_details["views"]) if str(video_details["views"]).isdigit() else 0,
            "thumbnail": video_details["thumbnail"],
            "stream_url": request.host_url.rstrip("/") + stream_url,
            "stream_type": "Video" if video else "Audio"
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in YouTube endpoint: {e}")
        return jsonify({"error": str(e)}), 500

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

# Admin API Routes
@app.route("/admin/metrics", methods=["GET"])
@required_admin_key
def get_metrics():
    """Get API usage metrics"""
    try:
        # Total requests
        total_requests = ApiLog.query.count()
        
        # Today's requests
        today_start = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_requests = ApiLog.query.filter(ApiLog.timestamp >= today_start).count()
        
        # Active keys
        active_keys = ApiKey.query.filter(ApiKey.valid_until >= datetime.datetime.now()).count()
        
        # Error rate
        error_logs = ApiLog.query.filter(ApiLog.response_status >= 400).count()
        error_rate = round((error_logs / total_requests) * 100, 2) if total_requests > 0 else 0
        
        # Daily requests for the past 7 days
        daily_requests = {}
        for i in range(7):
            day = datetime.datetime.now() - datetime.timedelta(days=i)
            day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day.replace(hour=23, minute=59, second=59, microsecond=999999)
            count = ApiLog.query.filter(ApiLog.timestamp.between(day_start, day_end)).count()
            daily_requests[day.strftime("%a")] = count
        
        # Key distribution
        key_distribution = {}
        for key in ApiKey.query.all():
            count = ApiLog.query.filter(ApiLog.api_key_id == key.id).count()
            if count > 0:
                key_distribution[key.name] = count
        
        return jsonify({
            "total_requests": total_requests,
            "today_requests": today_requests,
            "active_keys": active_keys,
            "error_rate": error_rate,
            "daily_requests": daily_requests,
            "key_distribution": key_distribution
        })
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/admin/list_api_keys", methods=["GET"])
@required_admin_key
def list_api_keys():
    """List all API keys"""
    try:
        keys = []
        for key in ApiKey.query.all():
            keys.append({
                "id": key.id,
                "key": key.key,
                "name": key.name,
                "is_admin": key.is_admin,
                "created_at": key.created_at.isoformat(),
                "valid_until": key.valid_until.isoformat(),
                "daily_limit": key.daily_limit,
                "count": key.count,
                "created_by": key.created_by
            })
        
        return jsonify(keys)
    except Exception as e:
        logger.error(f"Error listing API keys: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/admin/create_api_key", methods=["POST"])
@required_admin_key
def create_api_key():
    """Create a new API key"""
    try:
        admin_key_str = request.args.get("admin_key")
        admin_key = ApiKey.query.filter_by(key=admin_key_str).first()
        
        data = request.get_json()
        name = data.get("name", "User")
        days_valid = int(data.get("days_valid", 30))
        daily_limit = int(data.get("daily_limit", 100))
        is_admin = data.get("is_admin", False)
        
        # Generate a new API key
        api_key_str = secrets.token_hex(16)
        
        # Set expiration date
        valid_until = datetime.datetime.now() + datetime.timedelta(days=days_valid)
        reset_at = datetime.datetime.now() + datetime.timedelta(days=1)
        
        # Create the API key
        new_key = ApiKey(
            key=api_key_str,
            name=name,
            is_admin=is_admin,
            created_at=datetime.datetime.now(),
            valid_until=valid_until,
            daily_limit=daily_limit,
            reset_at=reset_at,
            count=0,
            created_by=admin_key.id if admin_key else None
        )
        
        db.session.add(new_key)
        db.session.commit()
        
        return jsonify({
            "id": new_key.id,
            "api_key": api_key_str,
            "name": name,
            "valid_until": valid_until.isoformat(),
            "daily_limit": daily_limit,
            "is_admin": is_admin
        })
    except Exception as e:
        logger.error(f"Error creating API key: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route("/admin/revoke_api_key", methods=["POST"])
@required_admin_key
def revoke_api_key():
    """Revoke an API key"""
    try:
        data = request.get_json()
        key_id = data.get("id")
        
        if not key_id:
            return jsonify({"error": "Key ID is required"}), 400
        
        # Find the key
        api_key = ApiKey.query.get(key_id)
        if not api_key:
            return jsonify({"error": "API key not found"}), 404
        
        # Delete the key
        db.session.delete(api_key)
        db.session.commit()
        
        return jsonify({"success": True, "message": "API key revoked"})
    except Exception as e:
        logger.error(f"Error revoking API key: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route("/admin/recent_logs", methods=["GET"])
@required_admin_key
def recent_logs():
    """Get recent API logs"""
    try:
        limit = int(request.args.get("limit", 20))
        
        logs = []
        for log in ApiLog.query.order_by(ApiLog.timestamp.desc()).limit(limit).all():
            api_key = ApiKey.query.get(log.api_key_id)
            logs.append({
                "id": log.id,
                "api_key": api_key.key if api_key else "",
                "endpoint": log.endpoint,
                "query": log.query,
                "ip_address": log.ip_address,
                "timestamp": log.timestamp.isoformat(),
                "status": log.response_status
            })
        
        return jsonify(logs)
    except Exception as e:
        logger.error(f"Error getting recent logs: {e}")
        return jsonify({"error": str(e)}), 500

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

# Add cleanup job
def run_cleanup_periodically():
    """Run cleanup job periodically"""
    while True:
        time.sleep(60 * 60)  # Run every hour
        try:
            # Clean up old files
            with app.app_context():
                cleanup_old_files()
        except Exception as e:
            logger.error(f"Error in cleanup job: {e}")

# Start background cleanup job
if __name__ == "__main__":
    # Start cleanup thread
    import threading
    cleanup_thread = threading.Thread(target=run_cleanup_periodically, daemon=True)
    cleanup_thread.start()
    
    # Run Flask app
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)