# Camera Setup Guide

## Overview

This setup allows Docker containers to access cameras by running a camera streamer on your host machine that broadcasts camera feeds over HTTP.

## Quick Start

### Step 1: Start the Camera Streamer (on Windows host)

```powershell
python camera_streamer.py
```

This will:

- Automatically discover all USB cameras and Raspberry Pi cameras
- Stream them on port 8080
- Show you the stream URLs

Example output:

```
Camera Streamer is running!
Local IP: 192.168.1.100

Available Streams:
  Camera: usb_0
    Stream:   http://192.168.1.100:8080/camera/usb_0/stream
    Snapshot: http://192.168.1.100:8080/camera/usb_0/snapshot
```

### Step 2: Run the Facial Recognition App

#### Option A: With Docker

1. Set the camera stream URL:

```powershell
$env:CAMERA_STREAM_URL="http://host.docker.internal:8080/camera/usb_0/stream"
```

2. Start Docker:

```powershell
docker-compose up
```

3. Access the app at: http://localhost:5000

#### Option B: Directly on Windows (without Docker)

```powershell
python web_server.py --stream=http://localhost:8080/camera/usb_0/stream
```

Access the app at: http://localhost:5000

## Features

### Camera Streamer (`camera_streamer.py`)

- **Auto-discovery**: Finds all USB and Raspberry Pi cameras
- **Multi-camera support**: Streams multiple cameras simultaneously
- **MJPEG streaming**: Standard format compatible with OpenCV
- **RESTful API**: JSON endpoint to list available cameras
- **Cross-platform**: Works on Windows, Linux, and Raspberry Pi

### Supported Camera Types

- USB webcams (any index)
- Raspberry Pi Camera Module (v1, v2, HQ)
- Any camera supported by OpenCV

## API Endpoints

### Camera Streamer (port 8080)

- `GET /` - List all cameras (JSON)
- `GET /camera/{camera_id}/stream` - MJPEG video stream
- `GET /camera/{camera_id}/snapshot` - Single frame (JPEG)

Example camera IDs:

- `usb_0`, `usb_1` - USB cameras
- `picamera_0` - Raspberry Pi camera

### Facial Recognition App (port 5000)

- `GET /` - Web interface
- `GET /video_feed` - Live video with face recognition
- `GET /api/detections` - Current face detections (JSON)
- `GET /api/status` - System status (JSON)

## Advanced Usage

### Multiple Cameras

The streamer supports multiple cameras. Each gets a unique ID:

```powershell
# Camera 0
http://localhost:8080/camera/usb_0/stream

# Camera 1
http://localhost:8080/camera/usb_1/stream

# Raspberry Pi camera
http://localhost:8080/camera/picamera_0/stream
```

### Network Access

The camera streamer is accessible from other devices on your network:

```powershell
# From another computer
http://YOUR_IP:8080/camera/usb_0/stream
```

### Custom Configuration

Edit `docker-compose.yml` to change the default camera:

```yaml
environment:
  - CAMERA_STREAM_URL=http://host.docker.internal:8080/camera/usb_1/stream
```

## Troubleshooting

### Camera not detected

- Make sure the camera is not in use by another application
- Try unplugging and replugging the USB camera
- For Raspberry Pi camera, enable it with `sudo raspi-config`

### Stream not accessible from Docker

- Verify the streamer is running on the host
- Make sure port 8080 is not blocked by firewall
- Use `host.docker.internal` instead of `localhost` in Docker

### Poor performance

- Reduce the number of cameras
- Lower the resolution in `camera_streamer.py`:
  ```python
  self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
  self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
  ```

### "Cannot open camera" error

- Check if the camera is already in use
- Try a different camera index (0, 1, 2, etc.)
- Verify the camera works with other applications

## Architecture

```
┌─────────────────┐
│  Windows Host   │
│                 │
│ camera_streamer │ ← Accesses USB/camera devices
│   (port 8080)   │
└────────┬────────┘
         │ HTTP/MJPEG
         ↓
┌────────────────────┐
│  Docker Container  │
│                    │
│   web_server.py    │ ← Connects to network stream
│    (port 5000)     │
└────────────────────┘
```

This architecture:

- ✅ Works on Windows, Linux, and macOS
- ✅ No USB passthrough needed
- ✅ Cameras accessible to multiple containers
- ✅ Standard HTTP protocol
- ✅ Works over network

## Notes

- The camera streamer must run on the same machine as the cameras
- Docker containers can access it via `host.docker.internal`
- Other devices on the network can access it via your IP address
- Streams are not encrypted (use VPN for remote access)
