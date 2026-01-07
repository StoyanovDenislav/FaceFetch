# Secure Camera Setup Guide

## Overview

The camera streamer now includes **token-based authentication** to prevent unauthorized access to your camera feeds.

## Quick Start (Secure)

### Step 1: Start the Camera Streamer

```powershell
python camera_streamer.py
```

The streamer will generate a secure random token and display it:

```
⚠️  SECURITY TOKEN (keep this secret!):
  abc123xyz789...

Set this in your environment:
  PowerShell: $env:CAMERA_STREAM_TOKEN="abc123xyz789..."
```

**Important:** Save this token securely!

### Step 2A: Run with Docker (Recommended)

```powershell
# Set the authentication token
$env:CAMERA_STREAM_TOKEN="abc123xyz789..."  # Use the token from Step 1

# Set the camera stream URL (without token, it will be added automatically)
$env:CAMERA_STREAM_URL="http://host.docker.internal:8080/camera/usb_0/stream"

# Start Docker
docker-compose up
```

### Step 2B: Run Directly on Windows

```powershell
# Set the authentication token
$env:CAMERA_STREAM_TOKEN="abc123xyz789..."

# Run with the stream URL
python web_server.py --stream=http://localhost:8080/camera/usb_0/stream
```

Access the app at: http://localhost:5000

## Security Features

### 🔒 Token-Based Authentication

- **Enabled by default** - All camera stream requests require a valid token
- **Random token generation** - Automatically generates a secure 32-character token
- **Token in URL or Header** - Supports both query parameter and `X-Camera-Token` header
- **Per-session tokens** - Each time you start the streamer, a new token is generated

### 🛡️ Access Control

- Only requests with the correct token can access camera streams
- Invalid requests return 403 Forbidden
- Token is required for:
  - Video streams (`/camera/{id}/stream`)
  - Snapshots (`/camera/{id}/snapshot`)
  - Camera list API (`/`)

### 📝 Best Practices

1. **Never commit tokens to git** - They're generated at runtime
2. **Use environment variables** - Don't hardcode tokens in your code
3. **Restrict network access** - Use firewall rules to limit access to port 8080
4. **Use HTTPS in production** - For remote access, put behind nginx/reverse proxy with SSL

## Advanced Configuration

### Using a Custom Token

Instead of a random token, you can set your own:

```powershell
# Set a custom token before starting the streamer
$env:CAMERA_STREAM_TOKEN="your-secret-token-here"

# Start the streamer
python camera_streamer.py
```

The streamer will use your token instead of generating one.

### Disabling Authentication (NOT RECOMMENDED)

For local testing only, you can disable authentication:

```powershell
$env:CAMERA_STREAM_AUTH="false"
python camera_streamer.py
```

⚠️ **WARNING:** Only do this on trusted networks. Your camera feed will be publicly accessible!

### Using with curl/API Clients

Query parameter method:

```bash
curl http://localhost:8080/camera/usb_0/snapshot?token=abc123xyz789
```

Header method (more secure):

```bash
curl -H "X-Camera-Token: abc123xyz789" http://localhost:8080/camera/usb_0/snapshot
```

## Network Security

### Local Network Only (Default)

The streamer binds to `0.0.0.0:8080`, making it accessible on your local network. This is fine for home/office use.

### Exposing to Internet (Advanced)

If you need remote access:

1. **Use a reverse proxy** (nginx/caddy) with HTTPS
2. **Set up VPN** - Access your network securely via VPN
3. **Use Cloudflare Tunnel** - Zero-trust access without port forwarding

Example nginx config:

```nginx
server {
    listen 443 ssl;
    server_name camera.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header X-Camera-Token $http_x_camera_token;
    }
}
```

### Firewall Configuration

Windows Firewall rule to limit access:

```powershell
# Allow only local network
New-NetFirewallRule -DisplayName "Camera Streamer" `
    -Direction Inbound -LocalPort 8080 -Protocol TCP `
    -Action Allow -Profile Private
```

## Docker Security

When using Docker, the token is passed via environment variables:

```yaml
# docker-compose.yml
environment:
  - CAMERA_STREAM_URL=http://host.docker.internal:8080/camera/usb_0/stream
  - CAMERA_STREAM_TOKEN=${CAMERA_STREAM_TOKEN}
```

The token never appears in the docker-compose.yml file itself, only in your environment.

## Troubleshooting

### "Invalid or missing authentication token"

- Check that `CAMERA_STREAM_TOKEN` is set correctly
- Make sure the token matches what the streamer displayed
- Try including the token in the URL: `?token=abc123`

### Token changes every restart

- This is by design for security
- Set a custom token via `CAMERA_STREAM_TOKEN` environment variable before starting the streamer

### Still accessible without token

- Check that `CAMERA_STREAM_AUTH` is not set to "false"
- Restart the streamer to ensure settings are applied

## Security Checklist

- ✅ Use token authentication (default)
- ✅ Store token in environment variables, not code
- ✅ Use firewall rules to restrict network access
- ✅ Don't expose port 8080 directly to internet
- ✅ Use HTTPS/SSL for remote access
- ✅ Rotate tokens regularly
- ✅ Monitor access logs for suspicious activity
- ❌ Don't commit tokens to git
- ❌ Don't disable authentication on untrusted networks
- ❌ Don't share camera stream URLs publicly

## Example Complete Setup

```powershell
# Terminal 1: Start camera streamer
python camera_streamer.py
# Copy the displayed token

# Terminal 2: Set environment and run Docker
$env:CAMERA_STREAM_TOKEN="abc123xyz789..."  # Token from Terminal 1
$env:CAMERA_STREAM_URL="http://host.docker.internal:8080/camera/usb_0/stream"
docker-compose up
```

Now your camera feed is:

- ✅ Encrypted with token authentication
- ✅ Only accessible with the correct token
- ✅ Safe from unauthorized access
- ✅ Ready for facial recognition processing

## Need More Security?

Consider these additional measures:

- **mTLS (Mutual TLS)** - Client certificate authentication
- **VPN** - Put everything behind a VPN
- **Network Segmentation** - Isolate camera network from internet
- **Rate Limiting** - Prevent brute force attacks on tokens
- **IP Whitelisting** - Only allow specific IPs
