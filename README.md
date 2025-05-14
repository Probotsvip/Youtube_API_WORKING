# YouTube API Service

A robust Flask API for YouTube content retrieval with advanced anti-bot protection and streaming capabilities. Designed to provide reliable, efficient, and secure access to YouTube media resources.

## Key Features

- **Ultra-fast search & play**: Average response time of 0.5 seconds
- **Seamless streaming**: Both audio and video formats supported
- **Live stream support**: Access live content just like regular videos
- **Anti-Bot Protection**: Multiple mechanisms to avoid YouTube's bot detection
  - IP rotation
  - User-agent cycling
  - Request jittering
  - Custom headers
- **API Key Management**: Admin panel to create and manage API keys
- **Optimized for Pyrogram & Telethon**: Perfect for Telegram bots

## API Endpoints

### Main Endpoint

```
GET /youtube
```

**Parameters:**
- `query` - YouTube URL, video ID, or search term
- `video` - Boolean to get video stream (default: false)
- `api_key` - Your API key (use the one provided to you)

**Example Response:**
```json
{
  "id": "n_FCrCQ6-bA",
  "title": "295 (Official Audio) | Sidhu Moose Wala | The Kidd | Moosetape",
  "duration": 273,
  "link": "https://www.youtube.com/watch?v=n_FCrCQ6-bA",
  "channel": "Sidhu Moose Wala",
  "views": 705107430,
  "thumbnail": "https://i.ytimg.com/vi_webp/n_FCrCQ6-bA/maxresdefault.webp",
  "stream_url": "https://yourapi.com/stream/cd97fd73-2ee0-4896-a1a6-f93145a893d3",
  "stream_type": "Audio"
}
```

### Stream Endpoint

```
GET /stream/:id
```

This endpoint is used internally by the API to stream media. You should not call it directly.

## Example Usage with Python

```python
import asyncio
import httpx

async def get_stream_url(query, video=False):
    api_url = "http://your-api-url.com/youtube"
    api_key = "your_api_key"
    
    async with httpx.AsyncClient(timeout=60) as client:
        params = {"query": query, "video": video, "api_key": api_key}
        response = await client.get(api_url, params=params)
        if response.status_code != 200:
            return ""
        info = response.json()
        return info.get("stream_url")
```

## Admin Panel

The admin panel is available at `/admin` and requires an admin API key as a query parameter:

```
/admin?admin_key=YOUR_ADMIN_KEY
```

Through the admin panel, you can:
- Monitor API usage statistics
- Create new API keys for friends
- View recent API logs
- Revoke existing API keys

## Deployment

### Heroku

[![Deploy to Heroku](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)

### Docker

```bash
docker build -t youtube-api-service .
docker run -p 5000:5000 youtube-api-service
```

### Manual Deployment

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables (optional)
4. Run the server: `gunicorn main:app`

## Environment Variables

- `DATABASE_URL` - Database connection URL (default: SQLite)
- `SECRET_KEY` - Secret key for session management
- `PORT` - Port to run the server on (default: 5000)
- `PROXY_LIST` - Comma-separated list of proxies to use (optional)

## Credits

Developed by [@INNOCENT_FUCKER](https://t.me/INNOCENT_FUCKER)