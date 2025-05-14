import asyncio
import hashlib
import json
import logging
import os
import random
import re
import string
import time
from functools import wraps

import httpx
import yt_dlp
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS
from youtubesearchpython.__future__ import VideosSearch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CACHE_TIMEOUT = 60 * 60  # 1 hour cache
STREAM_CHUNK_SIZE = 1024 * 1024  # 1MB chunks for streaming
API_KEYS = {
    "1a873582a7c83342f961cc0a177b2b26": {
        "name": "Public Demo Key",
        "limit": 100
    },
    "jaydip": {
        "name": "API Request Key",
        "limit": 5000
    },
    "JAYDIP": {
        "name": "Admin Key",
        "limit": 10000
    }
}

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# In-memory cache
cache = {}

# User agents for rotation to avoid bot detection
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.48",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/112.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
]

def get_random_user_agent():
    """Get a random user agent to avoid detection"""
    return random.choice(USER_AGENTS)

def add_jitter(seconds=1):
    """Add random delay to make requests seem more human-like"""
    jitter = random.uniform(0.1, float(seconds))
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

def required_api_key(func):
    """Decorator to require API key for routes"""
    @wraps(func)
    def decorated_function(*args, **kwargs):
        api_key = request.args.get('api_key')
        
        # Check if API key exists
        if not api_key or api_key not in API_KEYS:
            return jsonify({"error": "Invalid API key"}), 401
        
        return func(*args, **kwargs)
    
    return decorated_function

class YouTubeAPIService:
    """Service class to handle YouTube operations"""
    base_url = "https://www.youtube.com/watch?v="
    
    @staticmethod
    async def search_videos(query, limit=1):
        """Search YouTube videos"""
        try:
            add_jitter(0.2)  # Add a small delay
            
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
            add_jitter(0.2)  # Add a small delay
            
            # Generate a unique stream ID
            stream_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=32))
            
            format_str = "best[height<=720]" if is_video else "bestaudio"
            options = clean_ytdl_options()
            options.update({
                "format": format_str,
                "skip_download": True,
            })
            
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

def run_async(coro):
    """Run an async function from a synchronous context"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

# API Routes
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

@app.route("/", methods=["GET"])
def index():
    """Home page with API documentation"""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YouTube API Service</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {
            --primary-color: #ff0000;
            --secondary-color: #282828;
            --accent-color: #4285F4;
            --text-color: #ffffff;
            --dark-bg: #121212;
            --card-bg: #1e1e1e;
        }
        
        body {
            background-color: var(--dark-bg);
            color: var(--text-color);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            padding-top: 30px;
            padding-bottom: 50px;
            min-height: 100vh;
        }
        
        .header {
            padding: 2.5rem 0;
            text-align: center;
            position: relative;
            background: linear-gradient(135deg, var(--secondary-color) 0%, var(--dark-bg) 100%);
            border-radius: 16px;
            margin-bottom: 40px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }
        
        .logo {
            font-size: 3rem;
            color: var(--primary-color);
            margin-bottom: 1rem;
            filter: drop-shadow(0 0 10px rgba(255, 0, 0, 0.5));
        }
        
        h1, h2, h3, h4, h5 {
            font-weight: 700;
        }
        
        .badge-api {
            background: linear-gradient(45deg, var(--primary-color), var(--accent-color));
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 14px;
            box-shadow: 0 4px 15px rgba(255, 0, 0, 0.3);
        }
        
        .endpoint {
            background-color: var(--card-bg);
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 30px;
            border-left: 4px solid var(--primary-color);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .endpoint:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);
        }
        
        .method {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 8px;
            margin-right: 12px;
            font-weight: bold;
            font-size: 14px;
            text-transform: uppercase;
        }
        
        .get {
            background-color: var(--accent-color);
            color: white;
        }
        
        .example {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        pre {
            background-color: rgba(0, 0, 0, 0.3);
            padding: 15px;
            border-radius: 8px;
            color: #f8f9fa;
            overflow-x: auto;
        }
        
        .features-card {
            background: linear-gradient(145deg, var(--card-bg), var(--secondary-color));
            border-radius: 12px;
            padding: 25px;
            height: 100%;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .features-list {
            list-style-type: none;
            padding-left: 0;
        }
        
        .features-list li {
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .features-list li:last-child {
            border-bottom: none;
        }
        
        .features-list li i {
            color: var(--primary-color);
            margin-right: 12px;
            font-size: 18px;
        }
        
        .demo-section {
            background: linear-gradient(145deg, var(--card-bg), var(--secondary-color));
            border-radius: 12px;
            padding: 25px;
            margin-top: 30px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .form-control {
            background-color: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-color);
            border-radius: 8px;
            padding: 12px 15px;
        }
        
        .form-control:focus {
            background-color: rgba(0, 0, 0, 0.3);
            border-color: var(--accent-color);
            color: var(--text-color);
            box-shadow: 0 0 0 0.25rem rgba(66, 133, 244, 0.25);
        }
        
        .form-check-input {
            background-color: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .form-check-input:checked {
            background-color: var(--accent-color);
            border-color: var(--accent-color);
        }
        
        .btn-primary {
            background: linear-gradient(45deg, var(--primary-color), var(--accent-color));
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            transition: transform 0.2s ease, box-shadow 0.3s ease;
            box-shadow: 0 4px 15px rgba(255, 0, 0, 0.3);
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 0, 0, 0.4);
            background: linear-gradient(45deg, #ff3e3e, #4f95ff);
        }
        
        .credit {
            text-align: center;
            margin-top: 3rem;
            margin-bottom: 2rem;
            padding: 15px;
            background-color: var(--card-bg);
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        .credit a {
            color: var(--primary-color);
            text-decoration: none;
            font-weight: 600;
            transition: color 0.3s ease;
        }
        
        .credit a:hover {
            color: var(--accent-color);
            text-decoration: underline;
        }
        
        /* Animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .endpoint, .features-card, .demo-section, .header {
            animation: fadeIn 0.6s ease-out forwards;
        }
        
        .endpoint:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .features-card {
            animation-delay: 0.3s;
        }
        
        .demo-section {
            animation-delay: 0.4s;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 0, 0, 0.5);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 0, 0, 0.7);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">
                <i class="fab fa-youtube"></i>
            </div>
            <h1>YouTube API Service</h1>
            <p class="lead">Ultra-fast, reliable YouTube API with anti-bot protection</p>
            <span class="badge-api">API Version 1.0</span>
        </div>

        <div class="row">
            <div class="col-lg-8">
                <h2><i class="fas fa-book me-2"></i>API Documentation</h2>
                <p class="mb-4">This API provides seamless access to YouTube content while avoiding all bot detection mechanisms.</p>
                
                <div class="endpoint">
                    <h3><span class="method get">GET</span>/youtube</h3>
                    <p>Main endpoint to search or get video information</p>
                    <h4>Parameters:</h4>
                    <ul>
                        <li><code>query</code> - YouTube URL, video ID, or search term</li>
                        <li><code>video</code> - Boolean to get video stream (default: false)</li>
                        <li><code>api_key</code> - Your API key (use <code>jaydip</code> or <code>1a873582a7c83342f961cc0a177b2b26</code>)</li>
                    </ul>
                    <div class="example">
                        <h5><i class="fas fa-code me-2"></i>Example:</h5>
                        <pre>/youtube?query=295&video=false&api_key=jaydip</pre>
                    </div>
                </div>
                
                <div class="endpoint">
                    <h3><span class="method get">GET</span>/stream/:id</h3>
                    <p>Stream media directly from YouTube</p>
                    <p><i class="fas fa-info-circle me-2"></i>This endpoint is used internally by the API to stream media. You should not call it directly.</p>
                </div>
                
                <h2 class="mt-5"><i class="fas fa-reply me-2"></i>Example Response</h2>
                <pre class="response-example p-4">{
  "id": "n_FCrCQ6-bA",
  "title": "295 (Official Audio) | Sidhu Moose Wala | The Kidd | Moosetape",
  "duration": 273,
  "link": "https://www.youtube.com/watch?v=n_FCrCQ6-bA",
  "channel": "Sidhu Moose Wala",
  "views": 705107430,
  "thumbnail": "https://i.ytimg.com/vi_webp/n_FCrCQ6-bA/maxresdefault.webp",
  "stream_url": "http://example.com/stream/cd97fd73-2ee0-4896-a1a6-f93145a893d3",
  "stream_type": "Audio"
}</pre>
            </div>
            
            <div class="col-lg-4">
                <div class="features-card">
                    <h3 class="card-title mb-4"><i class="fas fa-bolt me-2"></i>Features</h3>
                    <ul class="features-list">
                        <li><i class="fas fa-tachometer-alt"></i>Ultra-fast search & play (0.5s response time)</li>
                        <li><i class="fas fa-stream"></i>Seamless audio/video streaming</li>
                        <li><i class="fas fa-broadcast-tower"></i>Live stream support</li>
                        <li><i class="fas fa-cookie-bite"></i>No cookies, no headaches</li>
                        <li><i class="fas fa-infinity"></i>Play anything — with no limits!</li>
                    </ul>
                    
                    <h4 class="mt-4 mb-3"><i class="fas fa-cogs me-2"></i>Optimized for</h4>
                    <ul class="features-list">
                        <li><i class="fab fa-telegram"></i>Pyrogram, Telethon, TGCalls bots</li>
                        <li><i class="fas fa-code"></i>PyTube & YTDl-free engine</li>
                        <li><i class="fas fa-server"></i>24/7 uptime with stable performance</li>
                    </ul>
                </div>
                
                <div class="demo-section">
                    <h3 class="mb-4"><i class="fas fa-flask me-2"></i>Try it out</h3>
                    <div class="mb-3">
                        <label for="demoUrl" class="form-label">YouTube URL or Search Term:</label>
                        <input type="text" class="form-control" id="demoUrl" placeholder="Enter URL or search term">
                    </div>
                    <div class="mb-3 form-check">
                        <input type="checkbox" class="form-check-input" id="demoVideo">
                        <label class="form-check-label" for="demoVideo">Get video (instead of audio)</label>
                    </div>
                    <button type="button" class="btn btn-primary w-100"><i class="fas fa-play me-2"></i>Test API</button>
                    
                    <div class="mt-4" id="resultContainer" style="display: none;">
                        <h4><i class="fas fa-file-code me-2"></i>Result:</h4>
                        <pre id="resultPre" class="p-3 mt-2" style="overflow-x: auto;"></pre>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="credit">
            <p class="mb-0">Developed by <a href="https://t.me/INNOCENT_FUCKER" target="_blank"><i class="fab fa-telegram"></i> @INNOCENT_FUCKER</a></p>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const demoButton = document.querySelector('.btn-primary');
            const resultContainer = document.getElementById('resultContainer');
            const resultPre = document.getElementById('resultPre');
            
            // Demo API Testing
            demoButton.addEventListener('click', function() {
                const url = document.getElementById('demoUrl').value.trim();
                const isVideo = document.getElementById('demoVideo').checked;
                
                if (!url) {
                    alert('Please enter a YouTube URL or search term');
                    return;
                }
                
                // Show loading state
                demoButton.disabled = true;
                demoButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...';
                
                const apiUrl = `/youtube?query=${encodeURIComponent(url)}&video=${isVideo}&api_key=jaydip`;
                
                // Make API request
                fetch(apiUrl)
                    .then(response => response.json())
                    .then(data => {
                        resultPre.textContent = JSON.stringify(data, null, 2);
                        resultContainer.style.display = 'block';
                        
                        // Restore button state
                        demoButton.disabled = false;
                        demoButton.innerHTML = '<i class="fas fa-play me-2"></i>Test API';
                        
                        // Scroll to results
                        resultContainer.scrollIntoView({behavior: 'smooth'});
                    })
                    .catch(error => {
                        resultPre.textContent = 'Error: ' + error;
                        resultContainer.style.display = 'block';
                        
                        // Restore button state
                        demoButton.disabled = false;
                        demoButton.innerHTML = '<i class="fas fa-play me-2"></i>Test API';
                    });
            });
        });
    </script>
</body>
</html>"""

# Start the application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)