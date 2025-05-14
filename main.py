import asyncio
import base64
import concurrent.futures
import hashlib
import json
import logging
import os
import random
import re
import string
import time
from functools import wraps
from typing import Dict, List, Optional, Union
from urllib.parse import parse_qs, urlparse

import httpx
import yt_dlp
from flask import Flask, Response, jsonify, render_template, request, send_file, stream_with_context
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from youtubesearchpython.__future__ import VideosSearch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_CONCURRENT_REQUESTS = 5
REQUEST_TIMEOUT = 30
STREAM_CHUNK_SIZE = 1024 * 1024  # 1MB
RATE_LIMIT = "60 per minute"
STRICT_RATE_LIMIT = "30 per minute"
CACHE_TIMEOUT = 60 * 60  # 1 hour

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get("SESSION_SECRET", "youtube_api_secret_key")

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

# User agents list for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36 OPR/78.0.4093.112",
]

# Proxy configuration (if needed)
PROXY_URL = os.environ.get("PROXY_URL", None)


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


class YouTubeAPIService:
    """Service class to handle YouTube operations"""
    base_url = "https://www.youtube.com/watch?v="
    list_base = "https://youtube.com/playlist?list="
    
    @staticmethod
    async def url_exists(url, video_id=None):
        """Check if a YouTube URL exists"""
        if video_id:
            url = f"https://www.youtube.com/watch?v={video_id}"
        
        if not is_youtube_url(url):
            return False
        
        # Quick check using oembed endpoint
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
                response = await client.get(oembed_url, headers={"User-Agent": get_random_user_agent()})
                return response.status_code == 200
            except:
                return False
    
    @staticmethod
    @cached()
    async def get_details(url, video_id=None):
        """Get video details"""
        try:
            if video_id:
                url = f"https://www.youtube.com/watch?v={video_id}"
            
            url = normalize_url(url)
            
            # Try with youtube-search-python first
            try:
                results = VideosSearch(url, limit=1)
                result_dict = await results.next()
                
                if "result" in result_dict and result_dict["result"]:
                    result = result_dict["result"][0]
                    
                    title = result["title"]
                    duration_min = result["duration"]
                    thumbnail = result["thumbnails"][0]["url"].split("?")[0]
                    vid_id = result["id"]
                    
                    duration_sec = time_to_seconds(duration_min)
                    
                    return {
                        "title": title,
                        "duration_min": duration_min,
                        "duration_sec": duration_sec,
                        "thumbnail": thumbnail,
                        "vidid": vid_id
                    }
            except Exception as e:
                logger.warning(f"Error getting details with VideosSearch: {e}")
            
            # Fallback to yt-dlp
            options = clean_ytdl_options()
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=False)
                
                title = info.get("title", "Unknown")
                duration_sec = info.get("duration", 0)
                
                # Format duration as mm:ss or hh:mm:ss
                if duration_sec > 3600:
                    hours, remainder = divmod(duration_sec, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    duration_min = f"{hours}:{minutes:02d}:{seconds:02d}"
                else:
                    minutes, seconds = divmod(duration_sec, 60)
                    duration_min = f"{minutes}:{seconds:02d}"
                
                thumbnail = info.get("thumbnail", "")
                if isinstance(thumbnail, dict) and "url" in thumbnail:
                    thumbnail = thumbnail["url"]
                
                vid_id = info.get("id", "")
                
                return {
                    "title": title,
                    "duration_min": duration_min,
                    "duration_sec": duration_sec,
                    "thumbnail": thumbnail,
                    "vidid": vid_id
                }
        except Exception as e:
            logger.error(f"Error getting video details: {e}")
            return {
                "title": "Unknown",
                "duration_min": "0:00",
                "duration_sec": 0,
                "thumbnail": "",
                "vidid": ""
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
        return details.get("duration_min", "0:00")
    
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
            
            url = normalize_url(url)
            
            # Generate a unique stream ID
            stream_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=32))
            
            format_str = "best" if is_video else "bestaudio"
            options = clean_ytdl_options()
            options.update({
                "format": format_str,
                "skip_download": True,
            })
            
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=False)
                best_format = info.get("url", "")
                
                if not best_format:
                    return ""
                
                # Store the URL in cache for streaming
                stream_key = f"stream:{stream_id}"
                cache[stream_key] = {
                    "url": best_format,
                    "created_at": time.time(),
                    "is_video": is_video
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
            if video_id:
                url = f"https://www.youtube.com/watch?v={video_id}"
            
            url = normalize_url(url)
            
            results = VideosSearch(url, limit=1)
            result_dict = await results.next()
            
            if "result" in result_dict and result_dict["result"]:
                result = result_dict["result"][0]
                
                title = result["title"]
                duration_min = result["duration"]
                vid_id = result["id"]
                yturl = result["link"]
                thumbnail = result["thumbnails"][0]["url"].split("?")[0]
                
                track_details = {
                    "title": title,
                    "link": yturl,
                    "vidid": vid_id,
                    "duration_min": duration_min,
                    "thumb": thumbnail,
                }
                
                return track_details, vid_id
            else:
                return {"title": "Unknown", "link": "", "vidid": "", "duration_min": "0:00", "thumb": ""}, ""
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
            
            url = normalize_url(url)
            
            options = clean_ytdl_options()
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
                    except:
                        continue
                
                return formats_available, url
        except Exception as e:
            logger.error(f"Error getting formats: {e}")
            return [], url
    
    @staticmethod
    async def slider(link, query_type, video_id=None):
        """Get related videos for a slider"""
        try:
            if video_id:
                link = f"https://www.youtube.com/watch?v={video_id}"
            
            if "&" in link:
                link = link.split("&")[0]
            
            results = VideosSearch(link, limit=10)
            result_dict = await results.next()
            
            if "result" in result_dict and result_dict["result"]:
                result = result_dict["result"][query_type]
                
                title = result["title"]
                duration_min = result["duration"]
                thumbnail = result["thumbnails"][0]["url"].split("?")[0]
                vid_id = result["id"]
                
                return title, duration_min, thumbnail, vid_id
            else:
                return "Unknown", "0:00", "", ""
        except Exception as e:
            logger.error(f"Error in slider: {e}")
            return "Unknown", "0:00", "", ""
    
    @staticmethod
    async def download(link, video=False, video_id=None, format_id=None, title=None):
        """Download a video or audio file"""
        try:
            if video_id:
                link = f"https://www.youtube.com/watch?v={video_id}"
            
            if "&" in link:
                link = link.split("&")[0]
            
            # Generate a unique file path
            filename = title or ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
            
            # Define options based on whether it's video or audio
            options = clean_ytdl_options()
            
            if video:
                if format_id:
                    options["format"] = f"{format_id}+140"
                else:
                    options["format"] = "(bestvideo[height<=?720][width<=?1280][ext=mp4])+(bestaudio[ext=m4a])"
                options["merge_output_format"] = "mp4"
                output_path = f"downloads/{filename}.mp4"
            else:
                if format_id:
                    options["format"] = format_id
                else:
                    options["format"] = "bestaudio/best"
                options["postprocessors"] = [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }]
                output_path = f"downloads/{filename}.mp3"
            
            options["outtmpl"] = output_path
            
            # Create downloads directory if it doesn't exist
            os.makedirs("downloads", exist_ok=True)
            
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([link])
            
            # Generate a download token
            download_token = ''.join(random.choices(string.ascii_lowercase + string.digits, k=16))
            cache[f"download:{download_token}"] = {
                "path": output_path,
                "created_at": time.time()
            }
            
            return f"/download/{download_token}"
        except Exception as e:
            logger.error(f"Error in download: {e}")
            return None


# Set up async event loop for the Flask app
def run_async(coro):
    """Run an async function from a synchronous context"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# API Routes
@app.route("/", methods=["GET"])
def index():
    """Home page with API documentation"""
    return render_index()


@app.route("/api/exists", methods=["GET"])
@limiter.limit(RATE_LIMIT)
def check_exists():
    """Check if a YouTube URL exists"""
    url = request.args.get("url")
    video_id = request.args.get("video_id")
    
    if not url and not video_id:
        return jsonify({"error": "URL or video_id required"}), 400
    
    result = run_async(YouTubeAPIService.url_exists(url, video_id))
    return jsonify({"exists": result})


@app.route("/api/details", methods=["GET"])
@limiter.limit(RATE_LIMIT)
def get_details():
    """Get video details"""
    url = request.args.get("url")
    video_id = request.args.get("video_id")
    
    if not url and not video_id:
        return jsonify({"error": "URL or video_id required"}), 400
    
    details = run_async(YouTubeAPIService.get_details(url, video_id))
    return jsonify(details)


@app.route("/api/title", methods=["GET"])
@limiter.limit(RATE_LIMIT)
def get_title():
    """Get video title"""
    url = request.args.get("url")
    video_id = request.args.get("video_id")
    
    if not url and not video_id:
        return jsonify({"error": "URL or video_id required"}), 400
    
    title = run_async(YouTubeAPIService.get_title(url, video_id))
    return jsonify({"title": title})


@app.route("/api/duration", methods=["GET"])
@limiter.limit(RATE_LIMIT)
def get_duration():
    """Get video duration"""
    url = request.args.get("url")
    video_id = request.args.get("video_id")
    
    if not url and not video_id:
        return jsonify({"error": "URL or video_id required"}), 400
    
    duration = run_async(YouTubeAPIService.get_duration(url, video_id))
    return jsonify({"duration": duration})


@app.route("/api/thumbnail", methods=["GET"])
@limiter.limit(RATE_LIMIT)
def get_thumbnail():
    """Get video thumbnail"""
    url = request.args.get("url")
    video_id = request.args.get("video_id")
    
    if not url and not video_id:
        return jsonify({"error": "URL or video_id required"}), 400
    
    thumbnail = run_async(YouTubeAPIService.get_thumbnail(url, video_id))
    return jsonify({"thumbnail": thumbnail})


@app.route("/api/stream", methods=["GET"])
@limiter.limit(STRICT_RATE_LIMIT)
def get_stream_url():
    """Get stream URL for a video"""
    url = request.args.get("url")
    video_id = request.args.get("video_id")
    video = request.args.get("video", "false").lower() == "true"
    
    if not url and not video_id:
        return jsonify({"error": "URL or video_id required"}), 400
    
    # Add API key validation (optional)
    api_key = request.args.get("api_key")
    if not api_key:
        return jsonify({"error": "API key required"}), 401
    
    stream_url = run_async(YouTubeAPIService.get_stream_url(url, video, video_id))
    
    # If using the format from the example
    result = {}
    if video_id or url:
        # Get additional details for the response
        details = run_async(YouTubeAPIService.get_details(url, video_id))
        result = {
            "id": details.get("vidid", ""),
            "title": details.get("title", "Unknown"),
            "duration": details.get("duration_sec", 0),
            "link": url or f"https://www.youtube.com/watch?v={video_id}",
            "channel": "",  # Would need additional API call
            "views": 0,  # Would need additional API call
            "thumbnail": details.get("thumbnail", ""),
            "stream_url": request.url_root.rstrip('/') + stream_url if stream_url else "",
            "stream_type": "Video" if video else "Audio"
        }
    
    return jsonify(result)


@app.route("/stream/<stream_id>", methods=["GET"])
def stream_media(stream_id):
    """Stream media from YouTube"""
    stream_key = f"stream:{stream_id}"
    stream_data = cache.get(stream_key)
    
    if not stream_data:
        return jsonify({"error": "Stream not found or expired"}), 404
    
    # Get the actual YouTube URL
    youtube_url = stream_data.get("url")
    
    # Function to stream content in chunks
    def generate():
        with httpx.stream("GET", youtube_url, timeout=None) as response:
            for chunk in response.iter_bytes(chunk_size=STREAM_CHUNK_SIZE):
                yield chunk
    
    # Set appropriate headers
    headers = {
        "Content-Type": "video/mp4" if stream_data.get("is_video") else "audio/mpeg",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    
    return Response(
        stream_with_context(generate()),
        headers=headers
    )


@app.route("/api/playlist", methods=["GET"])
@limiter.limit(STRICT_RATE_LIMIT)
def get_playlist():
    """Get playlist videos"""
    url = request.args.get("url")
    list_id = request.args.get("list_id")
    limit = request.args.get("limit", 10)
    user_id = request.args.get("user_id")
    
    if not url and not list_id:
        return jsonify({"error": "URL or list_id required"}), 400
    
    try:
        limit = int(limit)
    except:
        limit = 10
    
    playlist = run_async(YouTubeAPIService.get_playlist(url, limit, user_id, list_id))
    return jsonify({"playlist": playlist})


@app.route("/api/track", methods=["GET"])
@limiter.limit(RATE_LIMIT)
def get_track():
    """Get track details"""
    url = request.args.get("url")
    video_id = request.args.get("video_id")
    
    if not url and not video_id:
        return jsonify({"error": "URL or video_id required"}), 400
    
    track_details, vid_id = run_async(YouTubeAPIService.get_track(url, video_id))
    return jsonify({"track": track_details, "vid_id": vid_id})


@app.route("/api/formats", methods=["GET"])
@limiter.limit(RATE_LIMIT)
def get_formats():
    """Get available formats for a video"""
    url = request.args.get("url")
    video_id = request.args.get("video_id")
    
    if not url and not video_id:
        return jsonify({"error": "URL or video_id required"}), 400
    
    formats, video_url = run_async(YouTubeAPIService.get_formats(url, video_id))
    return jsonify({"formats": formats, "url": video_url})


@app.route("/api/slider", methods=["GET"])
@limiter.limit(RATE_LIMIT)
def get_slider():
    """Get related videos for a slider"""
    url = request.args.get("url")
    video_id = request.args.get("video_id")
    query_type = request.args.get("query_type", "0")
    
    if not url and not video_id:
        return jsonify({"error": "URL or video_id required"}), 400
    
    try:
        query_type = int(query_type)
    except:
        query_type = 0
    
    title, duration, thumbnail, vid_id = run_async(YouTubeAPIService.slider(url, query_type, video_id))
    return jsonify({
        "title": title,
        "duration": duration,
        "thumbnail": thumbnail,
        "vid_id": vid_id
    })


@app.route("/api/download", methods=["GET"])
@limiter.limit(STRICT_RATE_LIMIT)
def download_media():
    """Download a video or audio file"""
    url = request.args.get("url")
    video_id = request.args.get("video_id")
    video = request.args.get("video", "false").lower() == "true"
    format_id = request.args.get("format_id")
    title = request.args.get("title")
    
    if not url and not video_id:
        return jsonify({"error": "URL or video_id required"}), 400
    
    download_url = run_async(YouTubeAPIService.download(url, video, video_id, format_id, title))
    
    if not download_url:
        return jsonify({"error": "Download failed"}), 500
    
    return jsonify({"download_url": request.url_root.rstrip('/') + download_url})


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
            mimetype="video/mp4" if is_video else "audio/mpeg",
            as_attachment=True,
            download_name=filename
        )
    finally:
        # Schedule file cleanup (after 1 hour)
        cleanup_time = time.time() + 3600
        download_data["cleanup_at"] = cleanup_time


# Clean up old cache and downloads periodically
def cleanup_old_files():
    """Clean up old cache entries and downloaded files"""
    current_time = time.time()
    
    # Clean up stream cache entries (older than 1 hour)
    stream_keys = [key for key in cache.keys() if key.startswith("stream:")]
    for key in stream_keys:
        data = cache.get(key)
        if data and (current_time - data.get("created_at", 0)) > 3600:
            cache.pop(key, None)
    
    # Clean up download cache entries and files (older than 1 hour)
    download_keys = [key for key in cache.keys() if key.startswith("download:")]
    for key in download_keys:
        data = cache.get(key)
        if data and (current_time - data.get("cleanup_at", current_time + 3600)) <= current_time:
            file_path = data.get("path")
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
            cache.pop(key, None)


# Schedule cleanup to run every hour
def run_cleanup_periodically():
    while True:
        cleanup_old_files()
        time.sleep(3600)  # Sleep for 1 hour


# Start cleanup thread
import threading
cleanup_thread = threading.Thread(target=run_cleanup_periodically, daemon=True)
cleanup_thread.start()


# Error handlers
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429


@app.errorhandler(500)
def server_error_handler(e):
    return jsonify({"error": "Internal server error. Please try again later."}), 500


# Create downloads directory if it doesn't exist
os.makedirs("downloads", exist_ok=True)


# HTML template for index page
@app.route("/", methods=["GET"])
def render_index():
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YouTube API Service</title>
    <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        pre {
            padding: 1rem;
            border-radius: 0.3rem;
            background-color: var(--bs-dark);
        }
        .endpoint {
            margin-bottom: 2rem;
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: var(--bs-gray-800);
        }
        .method {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-weight: bold;
            margin-right: 0.5rem;
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
            margin-top: 1rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="pb-3 mb-4 border-bottom">
            <h1 class="fw-bold">YouTube API Service</h1>
            <p class="lead">A service to interact with YouTube without triggering bot detection</p>
        </header>

        <div class="p-4 mb-4 bg-light rounded-3">
            <div class="container-fluid py-2">
                <h2>API Documentation</h2>
                <p>This API provides various endpoints to interact with YouTube data. All endpoints return JSON responses.</p>
                
                <div class="alert alert-warning">
                    <strong>Note:</strong> Rate limits are applied to prevent API abuse and avoid YouTube bot detection.
                </div>
                
                <div class="card mb-4">
                    <div class="card-header">
                        <h3>Demo</h3>
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <label for="demo-url" class="form-label">YouTube URL or Video ID</label>
                            <input type="text" class="form-control" id="demo-url" placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ">
                        </div>
                        <div class="mb-3">
                            <label for="demo-endpoint" class="form-label">Select Endpoint</label>
                            <select class="form-select" id="demo-endpoint">
                                <option value="/api/details">Get Video Details</option>
                                <option value="/api/stream">Get Stream URL</option>
                                <option value="/api/formats">Get Available Formats</option>
                            </select>
                        </div>
                        <div class="mb-3 form-check">
                            <input type="checkbox" class="form-check-input" id="demo-video">
                            <label class="form-check-label" for="demo-video">Is Video? (for stream endpoint)</label>
                        </div>
                        <button class="btn btn-primary" onclick="runDemo()">Test API</button>
                        
                        <div class="mt-3" id="demo-result-container" style="display: none;">
                            <h4>Result:</h4>
                            <pre id="demo-result"></pre>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mb-4">
            <div class="col-12">
                <h2>Endpoints</h2>
                
                <div class="endpoint">
                    <h3><span class="method get">GET</span> /api/exists</h3>
                    <p>Check if a YouTube URL exists</p>
                    <h4>Parameters:</h4>
                    <ul>
                        <li><code>url</code> - YouTube URL</li>
                        <li><code>video_id</code> - YouTube video ID (alternative to URL)</li>
                    </ul>
                    <div class="example">
                        <h5>Example:</h5>
                        <pre>/api/exists?url=https://www.youtube.com/watch?v=dQw4w9WgXcQ</pre>
                    </div>
                </div>
                
                <div class="endpoint">
                    <h3><span class="method get">GET</span> /api/details</h3>
                    <p>Get video details including title, duration, and thumbnail</p>
                    <h4>Parameters:</h4>
                    <ul>
                        <li><code>url</code> - YouTube URL</li>
                        <li><code>video_id</code> - YouTube video ID (alternative to URL)</li>
                    </ul>
                    <div class="example">
                        <h5>Example:</h5>
                        <pre>/api/details?url=https://www.youtube.com/watch?v=dQw4w9WgXcQ</pre>
                    </div>
                </div>
                
                <div class="endpoint">
                    <h3><span class="method get">GET</span> /api/title</h3>
                    <p>Get video title</p>
                    <h4>Parameters:</h4>
                    <ul>
                        <li><code>url</code> - YouTube URL</li>
                        <li><code>video_id</code> - YouTube video ID (alternative to URL)</li>
                    </ul>
                </div>
                
                <div class="endpoint">
                    <h3><span class="method get">GET</span> /api/duration</h3>
                    <p>Get video duration</p>
                    <h4>Parameters:</h4>
                    <ul>
                        <li><code>url</code> - YouTube URL</li>
                        <li><code>video_id</code> - YouTube video ID (alternative to URL)</li>
                    </ul>
                </div>
                
                <div class="endpoint">
                    <h3><span class="method get">GET</span> /api/thumbnail</h3>
                    <p>Get video thumbnail URL</p>
                    <h4>Parameters:</h4>
                    <ul>
                        <li><code>url</code> - YouTube URL</li>
                        <li><code>video_id</code> - YouTube video ID (alternative to URL)</li>
                    </ul>
                </div>
                
                <div class="endpoint">
                    <h3><span class="method get">GET</span> /api/stream</h3>
                    <p>Get stream URL for a video or audio</p>
                    <h4>Parameters:</h4>
                    <ul>
                        <li><code>url</code> - YouTube URL</li>
                        <li><code>video_id</code> - YouTube video ID (alternative to URL)</li>
                        <li><code>video</code> - Set to "true" for video, "false" for audio (default: false)</li>
                        <li><code>api_key</code> - Required for authentication</li>
                    </ul>
                    <div class="example">
                        <h5>Example:</h5>
                        <pre>/api/stream?url=https://www.youtube.com/watch?v=dQw4w9WgXcQ&video=true&api_key=your_api_key</pre>
                    </div>
                </div>
                
                <div class="endpoint">
                    <h3><span class="method get">GET</span> /api/playlist</h3>
                    <p>Get videos from a playlist</p>
                    <h4>Parameters:</h4>
                    <ul>
                        <li><code>url</code> - YouTube playlist URL</li>
                        <li><code>list_id</code> - YouTube playlist ID (alternative to URL)</li>
                        <li><code>limit</code> - Maximum number of videos to return (default: 10)</li>
                        <li><code>user_id</code> - Optional user ID for tracking</li>
                    </ul>
                </div>
                
                <div class="endpoint">
                    <h3><span class="method get">GET</span> /api/track</h3>
                    <p>Get track details for audio</p>
                    <h4>Parameters:</h4>
                    <ul>
                        <li><code>url</code> - YouTube URL</li>
                        <li><code>video_id</code> - YouTube video ID (alternative to URL)</li>
                    </ul>
                </div>
                
                <div class="endpoint">
                    <h3><span class="method get">GET</span> /api/formats</h3>
                    <p>Get available formats for a video</p>
                    <h4>Parameters:</h4>
                    <ul>
                        <li><code>url</code> - YouTube URL</li>
                        <li><code>video_id</code> - YouTube video ID (alternative to URL)</li>
                    </ul>
                </div>
                
                <div class="endpoint">
                    <h3><span class="method get">GET</span> /api/slider</h3>
                    <p>Get related videos for a slider</p>
                    <h4>Parameters:</h4>
                    <ul>
                        <li><code>url</code> - YouTube URL</li>
                        <li><code>video_id</code> - YouTube video ID (alternative to URL)</li>
                        <li><code>query_type</code> - Index of the related video (default: 0)</li>
                    </ul>
                </div>
                
                <div class="endpoint">
                    <h3><span class="method get">GET</span> /api/download</h3>
                    <p>Download a video or audio file</p>
                    <h4>Parameters:</h4>
                    <ul>
                        <li><code>url</code> - YouTube URL</li>
                        <li><code>video_id</code> - YouTube video ID (alternative to URL)</li>
                        <li><code>video</code> - Set to "true" for video, "false" for audio (default: false)</li>
                        <li><code>format_id</code> - Optional format ID</li>
                        <li><code>title</code> - Optional title for the file</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <footer class="pt-3 mt-4 text-muted border-top">
            &copy; 2023 YouTube API Service
        </footer>
    </div>

    <script>
        function runDemo() {
            const urlInput = document.getElementById('demo-url');
            const endpointSelect = document.getElementById('demo-endpoint');
            const isVideo = document.getElementById('demo-video').checked;
            const resultContainer = document.getElementById('demo-result-container');
            const resultPre = document.getElementById('demo-result');
            
            let url = urlInput.value.trim();
            if (!url) {
                alert('Please enter a YouTube URL or video ID');
                return;
            }
            
            let endpoint = endpointSelect.value;
            let apiUrl = endpoint;
            
            // Determine if input is a URL or a video ID
            if (url.includes('youtube.com') || url.includes('youtu.be')) {
                apiUrl += `?url=${encodeURIComponent(url)}`;
            } else {
                apiUrl += `?video_id=${encodeURIComponent(url)}`;
            }
            
            // Add video parameter for stream endpoint
            if (endpoint === '/api/stream') {
                apiUrl += `&video=${isVideo}&api_key=demo_key`;
            }
            
            // Make API request
            fetch(apiUrl)
                .then(response => response.json())
                .then(data => {
                    resultPre.textContent = JSON.stringify(data, null, 2);
                    resultContainer.style.display = 'block';
                })
                .catch(error => {
                    resultPre.textContent = `Error: ${error}`;
                    resultContainer.style.display = 'block';
                });
        }
    </script>
</body>
</html>
'''


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
