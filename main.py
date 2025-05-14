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
    "my_secret_key": {
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
            <div class="badge bg-success">API Version 1.0</div>
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
                        <li><code>api_key</code> - Your API key (or use provided demo key)</li>
                    </ul>
                    <div class="example">
                        <h5>Example:</h5>
                        <pre>/youtube?query=never+gonna+give+you+up&video=false&api_key=1a873582a7c83342f961cc0a177b2b26</pre>
                    </div>
                </div>
                
                <div class="endpoint">
                    <h3><span class="method get">GET</span> /stream/:id</h3>
                    <p>Stream media from YouTube</p>
                    <p>This endpoint is used internally by the API to stream media. You should not call it directly.</p>
                </div>
                
                <h2>Example Response</h2>
                <pre class="bg-dark p-3">{
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
                    <button type="button" class="btn btn-primary" id="demoButton">Test API</button>
                    
                    <div class="mt-4" id="resultContainer" style="display: none;">
                        <h4>Result:</h4>
                        <pre id="resultPre" class="bg-dark text-light p-3" style="overflow-x: auto;"></pre>
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
            
            // Demo API Testing
            demoButton.addEventListener('click', function() {
                const url = document.getElementById('demoUrl').value.trim();
                const isVideo = document.getElementById('demoVideo').checked;
                
                if (!url) {
                    alert('Please enter a YouTube URL or search term');
                    return;
                }
                
                const apiUrl = `/youtube?query=${encodeURIComponent(url)}&video=${isVideo}&api_key=1a873582a7c83342f961cc0a177b2b26`;
                
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
        });
    </script>
</body>
</html>"""

# Start the application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)