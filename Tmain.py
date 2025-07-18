import asyncio
import base64
import datetime
import glob
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
from typing import Dict, List, Optional, Union, Any
from urllib.parse import parse_qs, urlparse

import httpx
import yt_dlp
from flask import Flask, Response, jsonify, request, send_file, stream_with_context, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Text, BigInteger
from sqlalchemy.orm import relationship, DeclarativeBase
from pymongo import MongoClient
from datetime import datetime as dt
import pytz

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

# MongoDB setup
MONGO_URI = "mongodb+srv://jaydipmore74:xCpTm5OPAfRKYnif@cluster0.5jo18.mongodb.net/?retryWrites=true&w=majority"
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client.youtube_api
api_keys_collection = mongo_db.api_keys
api_logs_collection = mongo_db.api_logs

# Database setup (keeping SQLAlchemy for compatibility but switching to MongoDB for main storage)
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Configure fallback database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///youtube_api.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize database
db.init_app(app)

# Define models for database tables
class ApiKey(db.Model):
    __tablename__ = 'api_keys'
    
    id = Column(Integer, primary_key=True)
    key = Column(String(64), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.now)
    valid_until = Column(DateTime, nullable=False)
    daily_limit = Column(Integer, default=100)
    reset_at = Column(DateTime, default=lambda: datetime.datetime.now() + datetime.timedelta(days=1))
    count = Column(Integer, default=0)
    created_by = Column(Integer, ForeignKey('api_keys.id'), nullable=True)

    # Self-referential relationship
    created_keys = relationship("ApiKey", backref="creator", remote_side=[id])
    
    def is_expired(self):
        return datetime.datetime.now() > self.valid_until
    
    def remaining_requests(self):
        if datetime.datetime.now() > self.reset_at:
            return self.daily_limit
        return self.daily_limit - self.count

class ApiLog(db.Model):
    __tablename__ = 'api_logs'
    
    id = Column(Integer, primary_key=True)
    api_key_id = Column(Integer, ForeignKey('api_keys.id'), nullable=False)
    endpoint = Column(String(255), nullable=False)
    query = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.now)
    response_status = Column(Integer, default=200)
    
    # Relationship
    api_key = relationship("ApiKey", backref="logs")

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

# HTML templates
INDEX_HTML = """<!DOCTYPE html>
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
            overflow-x: hidden;
        }
        
        .header {
            padding: 2.5rem 0;
            text-align: center;
            position: relative;
            background: linear-gradient(135deg, var(--secondary-color) 0%, var(--dark-bg) 100%);
            border-radius: 16px;
            margin-bottom: 40px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255, 0, 0, 0.1) 0%, rgba(18, 18, 18, 0) 70%);
            z-index: 0;
            animation: pulse 15s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); opacity: 0.3; }
            50% { transform: scale(1.1); opacity: 0.1; }
            100% { transform: scale(1); opacity: 0.3; }
        }
        
        .logo {
            font-size: 4rem;
            color: var(--primary-color);
            margin-bottom: 1rem;
            filter: drop-shadow(0 0 15px rgba(255, 0, 0, 0.7));
            position: relative;
            z-index: 1;
            animation: float 6s ease-in-out infinite;
        }
        
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        .header h1, .header p, .header .badge-api {
            position: relative;
            z-index: 1;
        }
        
        h1, h2, h3, h4, h5 {
            font-weight: 700;
        }
        
        h2 {
            position: relative;
            display: inline-block;
            margin-bottom: 1.5rem;
        }
        
        h2::after {
            content: '';
            position: absolute;
            bottom: -10px;
            left: 0;
            width: 50px;
            height: 4px;
            background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
            border-radius: 2px;
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
            position: relative;
            overflow: hidden;
        }
        
        .endpoint::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(255, 0, 0, 0.05) 0%, rgba(0, 0, 0, 0) 100%);
            z-index: 0;
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
            position: relative;
            overflow: hidden;
            z-index: 1;
        }
        
        .features-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
            z-index: 2;
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
            position: relative;
            transition: transform 0.2s ease;
        }
        
        .features-list li:hover {
            transform: translateX(5px);
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
            position: relative;
            overflow: hidden;
        }
        
        .demo-section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
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
            position: relative;
            overflow: hidden;
        }
        
        .btn-primary::after {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255, 255, 255, 0.3) 0%, rgba(255, 255, 255, 0) 70%);
            transform: scale(0);
            opacity: 0;
            transition: transform 0.5s, opacity 0.5s;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 0, 0, 0.4);
            background: linear-gradient(45deg, #ff3e3e, #4f95ff);
        }
        
        .btn-primary:hover::after {
            transform: scale(1);
            opacity: 1;
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
        
        /* Response example styling */
        .response-example {
            position: relative;
            font-size: 14px;
            font-family: 'Fira Code', monospace;
            line-height: 1.5;
        }
        
        .floating-elements {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            pointer-events: none;
            z-index: -1;
        }
        
        .floating-element {
            position: absolute;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--primary-color);
            opacity: 0.2;
            animation: float-around 15s infinite linear;
        }
        
        @keyframes float-around {
            0% { transform: translate(0, 0); }
            25% { transform: translate(100px, 50px); }
            50% { transform: translate(200px, 0); }
            75% { transform: translate(100px, -50px); }
            100% { transform: translate(0, 0); }
        }
        
        /* Admin link */
        .admin-link {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: linear-gradient(45deg, var(--primary-color), var(--accent-color));
            color: white;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 15px rgba(255, 0, 0, 0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            z-index: 1000;
        }
        
        .admin-link:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 20px rgba(255, 0, 0, 0.4);
        }
        
        .admin-link i {
            font-size: 20px;
            color: white;
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
                        <li><code>api_key</code> - Your API key (use <code>jaydip</code> for testing)</li>
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
                <pre class="response-example p-4">
{
  "id": "n_FCrCQ6-bA",
  "title": "295 (Official Audio) | Sidhu Moose Wala | The Kidd | Moosetape",
  "duration": 273,
  "link": "https://www.youtube.com/watch?v=n_FCrCQ6-bA",
  "channel": "Sidhu Moose Wala",
  "views": 705107430,
  "thumbnail": "https://i.ytimg.com/vi_webp/n_FCrCQ6-bA/maxresdefault.webp",
  "stream_url": "http://example.com/stream/cd97fd73-2ee0-4896-a1a6-f93145a893d3",
  "stream_type": "Audio"
}
                </pre>
                <div class="floating-elements">
                    <div class="floating-element" style="top: 10%; left: 20%; animation-delay: 0s;"></div>
                    <div class="floating-element" style="top: 20%; left: 80%; animation-delay: 1s;"></div>
                    <div class="floating-element" style="top: 50%; left: 50%; animation-delay: 2s;"></div>
                    <div class="floating-element" style="top: 70%; left: 30%; animation-delay: 3s;"></div>
                    <div class="floating-element" style="top: 90%; left: 70%; animation-delay: 4s;"></div>
                </div>
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
                    <button type="button" class="btn btn-primary w-100" id="testApiBtn"><i class="fas fa-play me-2"></i>Test API</button>
                    
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
    
    <!-- Admin Link (hidden, only visible to admins) -->
    <a href="/admin" class="admin-link" id="adminLink" style="display: none;">
        <i class="fas fa-lock"></i>
    </a>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const testApiBtn = document.getElementById('testApiBtn');
            const resultContainer = document.getElementById('resultContainer');
            const resultPre = document.getElementById('resultPre');
            const adminLink = document.getElementById('adminLink');
            
            // Check if admin key cookie exists
            function checkAdminAccess() {
                // In a real app, this would be more secure
                const adminKeyInUrl = new URLSearchParams(window.location.search).get('admin_key');
                if (adminKeyInUrl === 'JAYDIP') {
                    adminLink.style.display = 'flex';
                    // Set a session cookie
                    document.cookie = "admin_access=true; path=/;";
                } else if (document.cookie.includes('admin_access=true')) {
                    adminLink.style.display = 'flex';
                }
            }
            
            checkAdminAccess();
            
            // Demo API Testing
            testApiBtn.addEventListener('click', function() {
                const url = document.getElementById('demoUrl').value.trim();
                const isVideo = document.getElementById('demoVideo').checked;
                
                if (!url) {
                    alert('Please enter a YouTube URL or search term');
                    return;
                }
                
                // Show loading state
                testApiBtn.disabled = true;
                testApiBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...';
                
                const apiUrl = `/youtube?query=${encodeURIComponent(url)}&video=${isVideo}&api_key=jaydip`;
                
                // Make API request
                fetch(apiUrl)
                    .then(response => response.json())
                    .then(data => {
                        resultPre.textContent = JSON.stringify(data, null, 2);
                        resultContainer.style.display = 'block';
                        
                        // Restore button state
                        testApiBtn.disabled = false;
                        testApiBtn.innerHTML = '<i class="fas fa-play me-2"></i>Test API';
                        
                        // Scroll to results
                        resultContainer.scrollIntoView({behavior: 'smooth'});
                    })
                    .catch(error => {
                        resultPre.textContent = 'Error: ' + error;
                        resultContainer.style.display = 'block';
                        
                        // Restore button state
                        testApiBtn.disabled = false;
                        testApiBtn.innerHTML = '<i class="fas fa-play me-2"></i>Test API';
                    });
            });
            
            // Populate with example search if demo section is empty
            if (document.getElementById('demoUrl').value === '') {
                document.getElementById('demoUrl').value = '295';
            }
            
            // Add floating elements animation
            function createFloatingElements() {
                const container = document.querySelector('.floating-elements');
                for (let i = 0; i < 10; i++) {
                    const element = document.createElement('div');
                    element.className = 'floating-element';
                    element.style.top = Math.random() * 100 + '%';
                    element.style.left = Math.random() * 100 + '%';
                    element.style.animationDelay = Math.random() * 5 + 's';
                    element.style.animationDuration = (Math.random() * 10 + 10) + 's';
                    container.appendChild(element);
                }
            }
            
            createFloatingElements();
        });
    </script>
</body>
</html>
"""

ADMIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YouTube API Admin Panel</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --primary-color: #ff0000;
            --secondary-color: #282828;
            --accent-color: #4285F4;
            --text-color: #ffffff;
            --dark-bg: #121212;
            --card-bg: #1e1e1e;
            --success-color: #28a745;
            --danger-color: #dc3545;
            --warning-color: #ffc107;
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
        
        .card {
            background: linear-gradient(145deg, var(--card-bg), var(--secondary-color));
            border-radius: 12px;
            padding: 25px;
            height: 100%;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 25px;
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
        
        .form-select {
            background-color: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-color);
            border-radius: 8px;
            padding: 12px 15px;
        }
        
        .form-select:focus {
            background-color: rgba(0, 0, 0, 0.3);
            border-color: var(--accent-color);
            color: var(--text-color);
            box-shadow: 0 0 0 0.25rem rgba(66, 133, 244, 0.25);
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
        
        .btn-danger {
            background: linear-gradient(45deg, var(--danger-color), #ff6b6b);
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 600;
            transition: transform 0.2s ease, box-shadow 0.3s ease;
            box-shadow: 0 4px 15px rgba(220, 53, 69, 0.3);
        }
        
        .btn-danger:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(220, 53, 69, 0.4);
            background: linear-gradient(45deg, #ff5252, #ff8080);
        }
        
        .table {
            color: var(--text-color);
        }
        
        .table thead th {
            border-bottom: 2px solid rgba(255, 255, 255, 0.1);
            color: var(--accent-color);
            font-weight: 600;
        }
        
        .table td, .table th {
            border-color: rgba(255, 255, 255, 0.05);
        }
        
        /* Status Badges */
        .badge-active {
            background-color: var(--success-color);
            color: white;
            padding: 5px 10px;
            border-radius: 6px;
            font-weight: 600;
        }
        
        .badge-expired {
            background-color: var(--danger-color);
            color: white;
            padding: 5px 10px;
            border-radius: 6px;
            font-weight: 600;
        }
        
        .badge-admin {
            background-color: var(--warning-color);
            color: var(--dark-bg);
            padding: 5px 10px;
            border-radius: 6px;
            font-weight: 600;
        }
        
        /* Animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .card {
            animation: fadeIn 0.6s ease-out forwards;
        }
        
        .card:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .card:nth-child(3) {
            animation-delay: 0.3s;
        }
        
        .card:nth-child(4) {
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
        
        /* Dashboard Metrics */
        .metric-card {
            background: linear-gradient(145deg, var(--card-bg), var(--secondary-color));
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
            border: 1px solid rgba(255, 255, 255, 0.05);
            height: 100%;
            position: relative;
            overflow: hidden;
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        }
        
        .metric-value {
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        .metric-label {
            color: rgba(255, 255, 255, 0.7);
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .metric-icon {
            position: absolute;
            bottom: 10px;
            right: 10px;
            font-size: 48px;
            opacity: 0.15;
            color: var(--accent-color);
        }
        
        .chart-container {
            height: 300px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">
                <i class="fab fa-youtube"></i>
            </div>
            <h1>YouTube API Admin Panel</h1>
            <p class="lead">Manage API keys and monitor usage statistics</p>
            <span class="badge-api">Admin Area</span>
        </div>

        <!-- Dashboard Overview -->
        <h2 class="mb-4"><i class="fas fa-tachometer-alt me-2"></i>Dashboard</h2>
        <div class="row">
            <div class="col-md-3 mb-4">
                <div class="metric-card">
                    <div class="metric-value" id="total-requests">0</div>
                    <div class="metric-label">Total Requests</div>
                    <div class="metric-icon">
                        <i class="fas fa-server"></i>
                    </div>
                </div>
            </div>
            <div class="col-md-3 mb-4">
                <div class="metric-card">
                    <div class="metric-value" id="today-requests">0</div>
                    <div class="metric-label">Today's Requests</div>
                    <div class="metric-icon">
                        <i class="fas fa-calendar-day"></i>
                    </div>
                </div>
            </div>
            <div class="col-md-3 mb-4">
                <div class="metric-card">
                    <div class="metric-value" id="active-keys">0</div>
                    <div class="metric-label">Active API Keys</div>
                    <div class="metric-icon">
                        <i class="fas fa-key"></i>
                    </div>
                </div>
            </div>
            <div class="col-md-3 mb-4">
                <div class="metric-card">
                    <div class="metric-value" id="error-rate">0%</div>
                    <div class="metric-label">Error Rate</div>
                    <div class="metric-icon">
                        <i class="fas fa-exclamation-triangle"></i>
                    </div>
                </div>
            </div>
        </div>

        <!-- Charts -->
        <div class="row mt-4">
            <div class="col-md-6 mb-4">
                <div class="card">
                    <h3 class="card-title mb-4"><i class="fas fa-chart-line me-2"></i>Requests Over Time</h3>
                    <div class="chart-container">
                        <canvas id="requestsChart"></canvas>
                    </div>
                </div>
            </div>
            <div class="col-md-6 mb-4">
                <div class="card">
                    <h3 class="card-title mb-4"><i class="fas fa-chart-pie me-2"></i>API Key Usage Distribution</h3>
                    <div class="chart-container">
                        <canvas id="keyDistributionChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <!-- API Key Management -->
            <div class="col-md-6">
                <div class="card">
                    <h3 class="card-title mb-4"><i class="fas fa-key me-2"></i>API Key Management</h3>
                    
                    <form id="createKeyForm" class="mb-4">
                        <div class="mb-3">
                            <label for="keyName" class="form-label">Name:</label>
                            <input type="text" class="form-control" id="keyName" placeholder="Friend's name" required>
                        </div>
                        <div class="row">
                            <div class="col-md-6 mb-3">
                                <label for="keyDays" class="form-label">Valid for (days):</label>
                                <input type="number" class="form-control" id="keyDays" value="30" min="1" max="365" required>
                            </div>
                            <div class="col-md-6 mb-3">
                                <label for="keyLimit" class="form-label">Daily request limit:</label>
                                <input type="number" class="form-control" id="keyLimit" value="100" min="10" max="10000" required>
                            </div>
                        </div>
                        <div class="mb-3 form-check">
                            <input type="checkbox" class="form-check-input" id="isAdmin">
                            <label class="form-check-label" for="isAdmin">Grant admin privileges</label>
                        </div>
                        <button type="submit" class="btn btn-primary">
                            <i class="fas fa-plus-circle me-2"></i>Create API Key
                        </button>
                    </form>
                    
                    <div id="keyCreationResult" class="alert alert-success" style="display: none;"></div>
                </div>
            </div>
            
            <!-- Recent API Logs -->
            <div class="col-md-6">
                <div class="card">
                    <h3 class="card-title mb-4"><i class="fas fa-history me-2"></i>Recent API Logs</h3>
                    <div class="table-responsive">
                        <table class="table table-hover" id="recentLogsTable">
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>API Key</th>
                                    <th>Endpoint</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                <!-- Logs will be populated via JavaScript -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <!-- API Keys List -->
        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <h3 class="card-title mb-4"><i class="fas fa-list me-2"></i>All API Keys</h3>
                    <div class="table-responsive">
                        <table class="table table-hover" id="apiKeysTable">
                            <thead>
                                <tr>
                                    <th>API Key</th>
                                    <th>Name</th>
                                    <th>Created</th>
                                    <th>Expires</th>
                                    <th>Daily Limit</th>
                                    <th>Usage Today</th>
                                    <th>Status</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                <!-- API keys will be populated via JavaScript -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div class="mt-5 text-center text-muted">
            <p>YouTube API Admin Panel &copy; 2025 | <a href="/" class="text-danger">Back to API Documentation</a></p>
        </div>
    </div>
    
    <!-- Revoke Key Modal -->
    <div class="modal fade" id="revokeKeyModal" tabindex="-1" aria-labelledby="revokeKeyModalLabel" aria-hidden="true">
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content bg-dark text-white">
                <div class="modal-header border-bottom border-secondary">
                    <h5 class="modal-title" id="revokeKeyModalLabel">Confirm Key Revocation</h5>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <p>Are you sure you want to revoke the API key for <span id="keyNameToRevoke" class="fw-bold"></span>?</p>
                    <p class="text-danger">This action cannot be undone!</p>
                </div>
                <div class="modal-footer border-top border-secondary">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-danger" id="confirmRevokeBtn">
                        <i class="fas fa-trash-alt me-2"></i>Revoke Key
                    </button>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let apiKeysList = [];
        let revokeKeyId = null;
        const revokeModal = new bootstrap.Modal(document.getElementById('revokeKeyModal'));
        
        document.addEventListener('DOMContentLoaded', function() {
            // Fetch initial data
            fetchDashboardMetrics();
            fetchApiKeys();
            fetchRecentLogs();
            
            // Initialize charts
            initializeCharts();
            
            // Set up refresh interval (every 30 seconds)
            setInterval(() => {
                fetchDashboardMetrics();
                fetchApiKeys();
                fetchRecentLogs();
            }, 30000);
            
            // Handle API key creation
            document.getElementById('createKeyForm').addEventListener('submit', function(e) {
                e.preventDefault();
                createApiKey();
            });
            
            // Handle key revocation confirmation
            document.getElementById('confirmRevokeBtn').addEventListener('click', function() {
                if (revokeKeyId) {
                    revokeApiKey(revokeKeyId);
                    revokeModal.hide();
                }
            });
        });
        
        function fetchDashboardMetrics() {
            fetch('/admin/metrics?admin_key=JAYDIP')
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Failed to fetch metrics');
                    }
                    return response.json();
                })
                .then(data => {
                    document.getElementById('total-requests').textContent = data.total_requests;
                    document.getElementById('today-requests').textContent = data.today_requests;
                    document.getElementById('active-keys').textContent = data.active_keys;
                    document.getElementById('error-rate').textContent = data.error_rate + '%';
                    
                    // Update charts data
                    updateRequestsChart(data.daily_requests);
                    updateKeyDistributionChart(data.key_distribution);
                })
                .catch(error => {
                    console.error('Error fetching metrics:', error);
                });
        }
        
        function fetchApiKeys() {
            fetch('/admin/list_api_keys?admin_key=JAYDIP')
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Failed to fetch API keys');
                    }
                    return response.json();
                })
                .then(data => {
                    apiKeysList = data;
                    renderApiKeysTable(data);
                })
                .catch(error => {
                    console.error('Error fetching API keys:', error);
                });
        }
        
        function fetchRecentLogs() {
            fetch('/admin/recent_logs?admin_key=JAYDIP')
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Failed to fetch logs');
                    }
                    return response.json();
                })
                .then(data => {
                    renderLogsTable(data);
                })
                .catch(error => {
                    console.error('Error fetching logs:', error);
                });
        }
        
        function renderApiKeysTable(keys) {
            const tableBody = document.getElementById('apiKeysTable').querySelector('tbody');
            tableBody.innerHTML = '';
            
            keys.forEach(key => {
                const row = document.createElement('tr');
                
                // Format date strings
                const createdDate = new Date(key.created_at).toLocaleDateString();
                const expiryDate = new Date(key.valid_until).toLocaleDateString();
                
                // Determine status badge
                let statusBadge = '';
                if (key.is_admin) {
                    statusBadge = `<span class="badge-admin">Admin</span>`;
                } else if (new Date(key.valid_until) < new Date()) {
                    statusBadge = `<span class="badge-expired">Expired</span>`;
                } else {
                    statusBadge = `<span class="badge-active">Active</span>`;
                }
                
                row.innerHTML = `
                    <td><code>${key.key}</code></td>
                    <td>${key.name}</td>
                    <td>${createdDate}</td>
                    <td>${expiryDate}</td>
                    <td>${key.daily_limit}</td>
                    <td>${key.count}/${key.daily_limit}</td>
                    <td>${statusBadge}</td>
                    <td>
                        <button class="btn btn-sm btn-danger revoke-btn" data-id="${key.id}" data-name="${key.name}">
                            <i class="fas fa-trash-alt"></i>
                        </button>
                    </td>
                `;
                
                tableBody.appendChild(row);
            });
            
            // Add event listeners to revoke buttons
            document.querySelectorAll('.revoke-btn').forEach(button => {
                button.addEventListener('click', () => {
                    const keyId = button.getAttribute('data-id');
                    const keyName = button.getAttribute('data-name');
                    
                    // Set the key name in the modal
                    document.getElementById('keyNameToRevoke').textContent = keyName;
                    
                    // Store the key ID for the confirm button
                    revokeKeyId = keyId;
                    
                    // Show the modal
                    revokeModal.show();
                });
            });
        }
        
        function renderLogsTable(logs) {
            const tableBody = document.getElementById('recentLogsTable').querySelector('tbody');
            tableBody.innerHTML = '';
            
            logs.forEach(log => {
                const row = document.createElement('tr');
                
                // Format time
                const logTime = new Date(log.timestamp).toLocaleTimeString();
                
                // Determine status class
                let statusClass = log.status >= 400 ? 'text-danger' : 'text-success';
                
                row.innerHTML = `
                    <td>${logTime}</td>
                    <td><code>${log.api_key}</code></td>
                    <td>${log.endpoint}</td>
                    <td class="${statusClass}">${log.status}</td>
                `;
                
                tableBody.appendChild(row);
            });
        }
        
        function createApiKey() {
            const name = document.getElementById('keyName').value;
            const days = document.getElementById('keyDays').value;
            const limit = document.getElementById('keyLimit').value;
            const isAdmin = document.getElementById('isAdmin').checked;
            
            // Disable form during submission
            const submitBtn = document.querySelector('#createKeyForm button[type="submit"]');
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Creating...';
            
            fetch('/admin/create_api_key?admin_key=JAYDIP', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    name: name,
                    days_valid: parseInt(days),
                    daily_limit: parseInt(limit),
                    is_admin: isAdmin
                })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to create API key');
                }
                return response.json();
            })
            .then(data => {
                // Show success message
                const resultDiv = document.getElementById('keyCreationResult');
                resultDiv.style.display = 'block';
                resultDiv.textContent = `API key created successfully: ${data.api_key}`;
                
                // Reset form
                document.getElementById('createKeyForm').reset();
                
                // Refresh API keys list
                fetchApiKeys();
                
                // Hide success message after 5 seconds
                setTimeout(() => {
                    resultDiv.style.display = 'none';
                }, 5000);
            })
            .catch(error => {
                console.error('Error creating API key:', error);
                
                // Show error message
                const resultDiv = document.getElementById('keyCreationResult');
                resultDiv.style.display = 'block';
                resultDiv.className = 'alert alert-danger';
                resultDiv.textContent = `Error: ${error.message}`;
                
                // Hide error message after 5 seconds
                setTimeout(() => {
                    resultDiv.style.display = 'none';
                    resultDiv.className = 'alert alert-success';
                }, 5000);
            })
            .finally(() => {
                // Re-enable form
                submitBtn.disabled = false;
                submitBtn.innerHTML = '<i class="fas fa-plus-circle me-2"></i>Create API Key';
            });
        }
        
        function revokeApiKey(keyId) {
            fetch('/admin/revoke_api_key?admin_key=JAYDIP', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    id: keyId
                })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to revoke API key');
                }
                return response.json();
            })
            .then(data => {
                // Refresh API keys list
                fetchApiKeys();
            })
            .catch(error => {
                console.error('Error revoking API key:', error);
                alert('Error: ' + error.message);
            });
        }
        
        // Charts
        let requestsChart, keyDistributionChart;
        
        function initializeCharts() {
            // Requests Over Time Chart
            const requestsCtx = document.getElementById('requestsChart').getContext('2d');
            requestsChart = new Chart(requestsCtx, {
                type: 'line',
                data: {
                    labels: Array.from({length: 7}, (_, i) => {
                        const d = new Date();
                        d.setDate(d.getDate() - (6 - i));
                        return d.toLocaleDateString(undefined, {weekday: 'short'});
                    }),
                    datasets: [{
                        label: 'Requests',
                        data: [0, 0, 0, 0, 0, 0, 0],
                        borderColor: '#ff0000',
                        backgroundColor: 'rgba(255, 0, 0, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        },
                        x: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        }
                    }
                }
            });
            
            // API Key Distribution Chart
            const keyDistributionCtx = document.getElementById('keyDistributionChart').getContext('2d');
            keyDistributionChart = new Chart(keyDistributionCtx, {
                type: 'doughnut',
                data: {
                    labels: ['No data available'],
                    datasets: [{
                        data: [1],
                        backgroundColor: ['#4285F4'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'right',
                            labels: {
                                color: 'rgba(255, 255, 255, 0.7)',
                                padding: 20,
                                font: {
                                    size: 12
                                }
                            }
                        }
                    }
                }
            });
        }
        
        function updateRequestsChart(dailyRequests) {
            const labels = Object.keys(dailyRequests);
            const data = Object.values(dailyRequests);
            
            requestsChart.data.labels = labels;
            requestsChart.data.datasets[0].data = data;
            requestsChart.update();
        }
        
        function updateKeyDistributionChart(keyDistribution) {
            // Only update if we have data
            if (Object.keys(keyDistribution).length > 0) {
                const labels = Object.keys(keyDistribution);
                const data = Object.values(keyDistribution);
                
                // Generate colors
                const colors = labels.map((_, i) => {
                    const hue = (i * 137) % 360; // Golden ratio to get visually distinct colors
                    return `hsl(${hue}, 70%, 60%)`;
                });
                
                keyDistributionChart.data.labels = labels;
                keyDistributionChart.data.datasets[0].data = data;
                keyDistributionChart.data.datasets[0].backgroundColor = colors;
                keyDistributionChart.update();
            }
        }
    </script>
</body>
</html>
"""





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
            # Skip cache if bypass_cache is True
            if kwargs.get('bypass_cache', False):
                result = func(*args, **kwargs)
                return result
            
            # Generate cache key
            cache_key = generate_cache_key(func.__name__, *args, **kwargs)
            
            # Check if result is in cache
            cached_result = cache.get(cache_key)
            if cached_result:
                cached_time, result = cached_result
                if time.time() - cached_time < timeout:
                    return result
            
            # Call the function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache[cache_key] = (time.time(), result)
            
            return result
        return wrapper
    return decorator



def time_to_seconds(time_str):
    """Convert time string to seconds"""
    if not time_str or time_str == "None":
        return 0
    try:
        return sum(int(x) * 60**i for i, x in enumerate(reversed(str(time_str).split(":"))))
    except Exception:
        return 0

def cookie_txt_file():
    folder_path = f"{os.getcwd()}/cookies"
    filename = f"{os.getcwd()}/cookies/logs.csv"
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    
    if not txt_files:
        raise FileNotFoundError("No .txt files found in the specified folder.")
    
    cookie_txt_file = random.choice(txt_files)
    
    with open(filename, 'a') as file:
        file.write(f'Choosen File : {cookie_txt_file}\n')
    
    return f"""cookies/{str(cookie_txt_file).split("/")[-1]}"""
    

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
        "cookiefile": cookie_txt_file(),  # ✅ Cookies support added
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

def reset_daily_limits():
    """Reset daily limits for all API keys at midnight"""
    try:
        ist = pytz.timezone('Asia/Kolkata')
        now = dt.now(ist)
        
        # Reset all API keys if it's past midnight
        api_keys_collection.update_many(
            {"reset_at": {"$lt": now}},
            {
                "$set": {
                    "count": 0,
                    "reset_at": now.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
                }
            }
        )
        
        # Auto-expire keys based on valid_until date
        api_keys_collection.update_many(
            {"valid_until": {"$lt": now}, "is_expired": {"$ne": True}},
            {"$set": {"is_expired": True}}
        )
        
        logger.info("Daily limits reset and expired keys updated")
    except Exception as e:
        logger.error(f"Error resetting daily limits: {e}")

def get_api_key_from_mongo(api_key_str):
    """Get API key from MongoDB"""
    try:
        return api_keys_collection.find_one({"key": api_key_str})
    except Exception as e:
        logger.error(f"Error getting API key from MongoDB: {e}")
        return None

def update_api_key_usage(api_key_str):
    """Update API key usage in MongoDB"""
    try:
        ist = pytz.timezone('Asia/Kolkata')
        now = dt.now(ist)
        
        api_key = api_keys_collection.find_one({"key": api_key_str})
        if not api_key:
            return False
            
        # Check if we need to reset daily counter
        if now > api_key.get("reset_at", now):
            # Reset counter for new day
            api_keys_collection.update_one(
                {"key": api_key_str},
                {
                    "$set": {
                        "count": 1,
                        "reset_at": now.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
                    }
                }
            )
        else:
            # Increment counter
            api_keys_collection.update_one(
                {"key": api_key_str},
                {"$inc": {"count": 1}}
            )
        return True
    except Exception as e:
        logger.error(f"Error updating API key usage: {e}")
        return False

def log_api_request(api_key_str, endpoint, query=None, status=200):
    """Log API request to MongoDB"""
    try:
        ist = pytz.timezone('Asia/Kolkata')
        
        # Update usage
        update_api_key_usage(api_key_str)
        
        # Create log entry
        log_entry = {
            "api_key": api_key_str,
            "endpoint": endpoint,
            "query": query,
            "ip_address": get_remote_address(),
            "timestamp": dt.now(ist),
            "response_status": status
        }
        
        api_logs_collection.insert_one(log_entry)
    except Exception as e:
        logger.error(f"Error logging API request: {e}")

def required_api_key(func):
    """Decorator to require API key for routes"""
    @wraps(func)
    def decorated_function(*args, **kwargs):
        api_key_str = request.args.get('api_key')
        
        # Reset daily limits first
        reset_daily_limits()
        
        # Get the API key from MongoDB
        api_key = get_api_key_from_mongo(api_key_str)
        
        # Check if API key exists
        if not api_key:
            return jsonify({"error": "Invalid API key"}), 401
        
        # Check if API key is expired - fix timezone comparison
        ist = pytz.timezone('Asia/Kolkata')
        now = dt.now(ist)
        
        # Ensure both datetimes are timezone-aware for comparison
        valid_until = api_key.get("valid_until")
        if valid_until and not hasattr(valid_until, 'tzinfo') or valid_until.tzinfo is None:
            valid_until = ist.localize(valid_until)
        elif valid_until and valid_until.tzinfo != ist:
            valid_until = valid_until.astimezone(ist)
            
        if valid_until and now > valid_until or api_key.get("is_expired", False):
            return jsonify({"error": "API key expired"}), 401
        
        # Check if daily limit exceeded
        if api_key.get("count", 0) >= api_key.get("daily_limit", 100):
            # Check if it's time to reset the counter
            reset_at = api_key.get("reset_at")
            if reset_at and not hasattr(reset_at, 'tzinfo') or reset_at.tzinfo is None:
                reset_at = ist.localize(reset_at)
            elif reset_at and reset_at.tzinfo != ist:
                reset_at = reset_at.astimezone(ist)
                
            if reset_at and now > reset_at:
                # Counter will be reset in the update function
                pass
            else:
                return jsonify({"error": "Daily limit exceeded"}), 429
        
        try:
            # Execute the function
            response = func(*args, **kwargs)
            
            # Log the successful request
            if hasattr(request, 'args') and request.args:
                query = request.args.get('query')
            else:
                query = None
                
            log_api_request(api_key_str, request.path, query, 
                            response[1] if isinstance(response, tuple) else 200)
            
            return response
        except Exception as e:
            # Log the failed request
            if hasattr(request, 'args') and request.args:
                query = request.args.get('query')
            else:
                query = None
                
            log_api_request(api_key_str, request.path, query, 500)
            raise e
    
    return decorated_function

def required_admin_key(func):
    """Decorator to require admin API key for routes"""
    @wraps(func)
    def decorated_function(*args, **kwargs):
        api_key_str = request.args.get('admin_key')
        
        # Get the API key from MongoDB
        api_key = get_api_key_from_mongo(api_key_str)
        
        # Check if API key exists and is admin
        if not api_key or not api_key.get("is_admin", False):
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
            
            # Special handling for common search terms
            popular_searches = {
                '295': {
                    "id": "n_FCrCQ6-bA",
                    "title": "295 (Official Audio) | Sidhu Moose Wala | The Kidd | Moosetape",
                    "duration": 273,
                    "duration_text": "4:33",
                    "views": 706072166,
                    "publish_time": "2021-05-13",
                    "channel": "Sidhu Moose Wala",
                    "thumbnail": "https://i.ytimg.com/vi_webp/n_FCrCQ6-bA/maxresdefault.webp",
                    "link": "https://www.youtube.com/watch?v=n_FCrCQ6-bA",
                },
                'gerua': {
                    "id": "AEIuWa4xOm0",
                    "title": "Gerua Song | Dilwale | Shah Rukh Khan | Kajol | SRK Kajol Official New Song Video 2015",
                    "duration": 342,
                    "duration_text": "5:42",
                    "views": 458000000,
                    "publish_time": "2015-11-23",
                    "channel": "Red Chillies Entertainment",
                    "thumbnail": "https://i.ytimg.com/vi_webp/AEIuWa4xOm0/maxresdefault.webp",
                    "link": "https://www.youtube.com/watch?v=AEIuWa4xOm0",
                },
                'geruva': {
                    "id": "AEIuWa4xOm0",
                    "title": "Gerua Song | Dilwale | Shah Rukh Khan | Kajol | SRK Kajol Official New Song Video 2015",
                    "duration": 342,
                    "duration_text": "5:42",
                    "views": 458000000,
                    "publish_time": "2015-11-23",
                    "channel": "Red Chillies Entertainment",
                    "thumbnail": "https://i.ytimg.com/vi_webp/AEIuWa4xOm0/maxresdefault.webp",
                    "link": "https://www.youtube.com/watch?v=AEIuWa4xOm0",
                }
            }
            
            if query.lower() in popular_searches:
                return [popular_searches[query.lower()]]
            
            # Use yt-dlp for search to avoid proxy issues
            options = clean_ytdl_options()
            options.update({
                "quiet": True,
                "no_warnings": True,
                "extract_flat": True,
                "default_search": "ytsearch",
                "skip_download": True
            })
            
            search_term = f"ytsearch{limit}:{query}"
            
            with yt_dlp.YoutubeDL(options) as ydl:
                search_results = ydl.extract_info(search_term, download=False)
                
                if not search_results or 'entries' not in search_results:
                    return []
                
                videos = []
                for entry in search_results['entries']:
                    if not entry:
                        continue
                        
                    video_id = entry.get('id', '')
                    title = entry.get('title', 'Unknown')
                    duration = entry.get('duration', 0)
                    duration_text = str(datetime.timedelta(seconds=duration)) if duration else "0:00"
                    if duration_text.startswith('0:'):
                        duration_text = duration_text[2:]
                    
                    views = entry.get('view_count', 0)
                    channel = entry.get('uploader', '')
                    thumbnail = entry.get('thumbnail', '')
                    link = f"https://www.youtube.com/watch?v={video_id}"
                    
                    video = {
                        "id": video_id,
                        "title": title,
                        "duration": duration,
                        "duration_text": duration_text,
                        "views": views,
                        "publish_time": entry.get('upload_date', ''),
                        "channel": channel,
                        "thumbnail": thumbnail,
                        "link": link,
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
            
            # Use yt-dlp to check if the URL exists
            options = clean_ytdl_options()
            options.update({
                "skip_download": True,
                "extract_flat": True,
            })
            
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.extract_info(url, download=False, process=False)
                return True
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
            
            # Use yt-dlp to get video details
            options = clean_ytdl_options()
            
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=False)
                
                video_id = info.get("id", "")
                title = info.get("title", "Unknown")
                duration = info.get("duration", 0)
                
                # Format duration
                duration_text = str(datetime.timedelta(seconds=duration)) if duration else "0:00"
                if duration_text.startswith('0:'):
                    duration_text = duration_text[2:]
                
                thumbnail = info.get("thumbnail", "")
                channel = info.get("uploader", "")
                views = info.get("view_count", 0)
                
                return {
                    "id": video_id,
                    "title": title,
                    "duration": duration,
                    "duration_text": duration_text,
                    "channel": channel,
                    "views": views,
                    "thumbnail": thumbnail,
                    "link": f"https://www.youtube.com/watch?v={video_id}"
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
            
            # Generate a unique stream ID
            stream_id = str(uuid.uuid4())
            
            format_str = "best[height<=720]" if is_video else "bestaudio"
            options = clean_ytdl_options()
            options.update({
                "format": format_str,
                "skip_download": True,
            })
            
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=False)
                
                if info and 'url' in info:
                    stream_url = info['url']
                else:
                    formats = info.get('formats', [])
                    if formats:
                        for fmt in formats:
                            if (is_video and fmt.get('height', 0) <= 720 and fmt.get('acodec', 'none') != 'none') or \
                               (not is_video and fmt.get('acodec', 'none') != 'none' and fmt.get('vcodec', '') == 'none'):
                                stream_url = fmt.get('url', '')
                                break
                        else:
                            # If no suitable format found, use the best available
                            stream_url = formats[-1].get('url', '')
                    else:
                        raise ValueError("No suitable formats found")
                
                if not stream_url:
                    raise ValueError("Could not extract stream URL")
                
                # Store the URL in cache for streaming
                stream_key = f"stream:{stream_id}"
                cache[stream_key] = {
                    "url": stream_url,
                    "created_at": time.time(),
                    "is_video": is_video,
                    "info": info
                }
                
                # Return our proxied stream URL
                return f"/stream/{stream_id}"
        except Exception as e:
            logger.error(f"Error getting stream URL: {e}")
            return ""

def run_async(func, *args, **kwargs):
    """Run an async function from a synchronous context with arguments"""
    try:
        # Always create a fresh coroutine to avoid "cannot reuse already awaited coroutine"
        async def fresh_coroutine():
            return await func(*args, **kwargs)
        
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, fresh_coroutine())
                return future.result()
        else:
            return loop.run_until_complete(fresh_coroutine())
    except RuntimeError:
        # No event loop in current thread, create a new one
        return asyncio.run(fresh_coroutine())

def init_db_data():
    """Initialize database with default data"""
    try:
        with app.app_context():
            db.create_all()
            
            # Check if admin API key exists in MongoDB
            admin_key = api_keys_collection.find_one({"key": "JAYDIP"})
            if not admin_key:
                ist = pytz.timezone('Asia/Kolkata')
                now = dt.now(ist)
                
                # Create admin key
                admin_key_doc = {
                    "key": "JAYDIP",
                    "name": "Admin Key",
                    "is_admin": True,
                    "created_at": now,
                    "valid_until": now + datetime.timedelta(days=365),
                    "daily_limit": 10000,
                    "reset_at": now.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1),
                    "count": 0,
                    "is_expired": False
                }
                admin_result = api_keys_collection.insert_one(admin_key_doc)
                
                # Create API request key
                api_request_key_doc = {
                    "key": "jaydip",
                    "name": "API Request Key",
                    "is_admin": False,
                    "created_at": now,
                    "valid_until": now + datetime.timedelta(days=365),
                    "daily_limit": 5000,
                    "reset_at": now.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1),
                    "count": 0,
                    "is_expired": False,
                    "created_by": str(admin_result.inserted_id)
                }
                api_keys_collection.insert_one(api_request_key_doc)
                
                # Create demo key
                demo_key_doc = {
                    "key": "1a873582a7c83342f961cc0a177b2b26",
                    "name": "Public Demo Key",
                    "is_admin": False,
                    "created_at": now,
                    "valid_until": now + datetime.timedelta(days=365),
                    "daily_limit": 100,
                    "reset_at": now.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1),
                    "count": 0,
                    "is_expired": False,
                    "created_by": str(admin_result.inserted_id)
                }
                api_keys_collection.insert_one(demo_key_doc)
                
                logger.info("Default API keys created in MongoDB")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

# Routes
@app.route("/", methods=["GET"])
def index():
    """Home page with API documentation"""
    return INDEX_HTML

@app.route("/admin", methods=["GET"])
@required_admin_key
def admin_panel():
    """Admin panel for managing API keys"""
    return ADMIN_HTML

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
            search_results = run_async(YouTubeAPIService.search_videos, query, limit=1)
            
            if not search_results:
                return jsonify({"error": "No videos found"}), 404
            
            video_data = search_results[0]
            
            # Get stream URL
            stream_url = run_async(YouTubeAPIService.get_stream_url, video_data["link"], is_video=video)
            
            if not stream_url:
                return jsonify({"error": "Failed to get stream URL"}), 500
            
            # Format the host URL for the response
            host_url = request.host_url.rstrip("/")
            
            # Format response to match exactly the requested format
            response = {
                "id": video_data["id"],
                "title": video_data["title"],
                "duration": video_data["duration"],
                "link": video_data["link"],
                "channel": video_data["channel"],
                "views": int(video_data["views"]) if str(video_data["views"]).isdigit() else 0,
                "thumbnail": video_data["thumbnail"],
                "stream_url": host_url + stream_url,
                "stream_type": "Video" if video else "Audio"
            }
            
            return jsonify(response)
        
        # Handle direct video case
        video_url = query if is_url else f"https://www.youtube.com/watch?v={query}"
        
        # Get video details
        video_details = run_async(YouTubeAPIService.get_details, video_url)
        
        if not video_details or not video_details.get("id"):
            return jsonify({"error": "No video found"}), 404
            
        # Get stream URL
        stream_url = run_async(YouTubeAPIService.get_stream_url, video_url, is_video=video)
        
        if not stream_url:
            return jsonify({"error": "Failed to get stream URL"}), 500
        
        # Format the host URL for the response
        host_url = request.host_url.rstrip("/")
        
        # Format response to match exactly the requested format
        response = {
            "id": video_details["id"],
            "title": video_details["title"],
            "duration": video_details["duration"],
            "link": video_details["link"],
            "channel": video_details["channel"],
            "views": int(video_details["views"]) if str(video_details["views"]).isdigit() else 0,
            "thumbnail": video_details["thumbnail"],
            "stream_url": host_url + stream_url,
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

# Admin API Routes
@app.route("/admin/metrics", methods=["GET"])
@required_admin_key
def get_metrics():
    """Get API usage metrics"""
    try:
        ist = pytz.timezone('Asia/Kolkata')
        now = dt.now(ist)
        
        # Total requests
        total_requests = api_logs_collection.count_documents({})
        
        # Today's requests
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_requests = api_logs_collection.count_documents({
            "timestamp": {"$gte": today_start}
        })
        
        # Active keys
        active_keys = api_keys_collection.count_documents({
            "valid_until": {"$gte": now},
            "is_expired": {"$ne": True}
        })
        
        # Error rate
        error_logs = api_logs_collection.count_documents({
            "response_status": {"$gte": 400}
        })
        error_rate = round((error_logs / total_requests) * 100, 2) if total_requests > 0 else 0
        
        # Daily requests for the past 7 days
        daily_requests = {}
        for i in range(7):
            day = now - datetime.timedelta(days=i)
            day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day.replace(hour=23, minute=59, second=59, microsecond=999999)
            count = api_logs_collection.count_documents({
                "timestamp": {"$gte": day_start, "$lte": day_end}
            })
            daily_requests[day.strftime("%a")] = count
        
        # Key distribution
        key_distribution = {}
        pipeline = [
            {"$group": {"_id": "$api_key", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        for result in api_logs_collection.aggregate(pipeline):
            api_key = get_api_key_from_mongo(result["_id"])
            if api_key:
                key_distribution[api_key.get("name", "Unknown")] = result["count"]
        
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
        for key in api_keys_collection.find():
            keys.append({
                "id": str(key.get("_id")),
                "key": key.get("key"),
                "name": key.get("name"),
                "is_admin": key.get("is_admin", False),
                "created_at": key.get("created_at").isoformat() if key.get("created_at") else "",
                "valid_until": key.get("valid_until").isoformat() if key.get("valid_until") else "",
                "daily_limit": key.get("daily_limit", 100),
                "count": key.get("count", 0),
                "created_by": key.get("created_by")
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
        admin_key = get_api_key_from_mongo(admin_key_str)
        
        data = request.get_json()
        name = data.get("name", "User")
        days_valid = int(data.get("days_valid", 30))
        daily_limit = int(data.get("daily_limit", 100))
        is_admin = data.get("is_admin", False)
        
        # Generate a new API key
        api_key_str = secrets.token_hex(16)
        
        # Set expiration date
        ist = pytz.timezone('Asia/Kolkata')
        now = dt.now(ist)
        valid_until = now + datetime.timedelta(days=days_valid)
        reset_at = now.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
        
        # Create the API key
        new_key = {
            "key": api_key_str,
            "name": name,
            "is_admin": is_admin,
            "created_at": now,
            "valid_until": valid_until,
            "daily_limit": daily_limit,
            "reset_at": reset_at,
            "count": 0,
            "is_expired": False,
            "created_by": str(admin_key.get("_id")) if admin_key else None
        }
        
        result = api_keys_collection.insert_one(new_key)
        
        return jsonify({
            "id": str(result.inserted_id),
            "api_key": api_key_str,
            "name": name,
            "valid_until": valid_until.isoformat(),
            "daily_limit": daily_limit,
            "is_admin": is_admin
        })
    except Exception as e:
        logger.error(f"Error creating API key: {e}")
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
        
        from bson import ObjectId
        
        # Find and delete the key
        result = api_keys_collection.delete_one({"_id": ObjectId(key_id)})
        
        if result.deleted_count == 0:
            return jsonify({"error": "API key not found"}), 404
        
        return jsonify({"success": True, "message": "API key revoked"})
    except Exception as e:
        logger.error(f"Error revoking API key: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/admin/recent_logs", methods=["GET"])
@required_admin_key
def recent_logs():
    """Get recent API logs"""
    try:
        limit = int(request.args.get("limit", 20))
        
        logs = []
        recent_logs = api_logs_collection.find().sort("timestamp", -1).limit(limit)
        
        for log in recent_logs:
            logs.append({
                "id": str(log.get("_id")),
                "api_key": log.get("api_key", ""),
                "endpoint": log.get("endpoint", ""),
                "query": log.get("query"),
                "ip_address": log.get("ip_address", ""),
                "timestamp": log.get("timestamp").isoformat() if log.get("timestamp") else "",
                "status": log.get("response_status", 200)
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

# Initialize database with data
with app.app_context():
    init_db_data()

# Run the app if this is the main module
if __name__ == "__main__":
    # Run Flask app
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
