# FaceFetch Authentication System - Full Implementation

## Overview
The FaceFetch authentication system is now fully functional with complete backend integration.

## Authentication Flow

### 1. Registration
- **URL**: `/register`
- **Method**: GET (show form) or POST (submit form)
- **Form Fields**:
  - `firstName`: User's first name (required)
  - `lastName`: User's last name (required)
  - `email`: User's email address (required, must be unique)
  - `password`: Password (required, minimum 8 characters)
  - `company`: Company name (optional)

**Example Request:**
```bash
curl -X POST http://localhost:5000/register \
  -d "firstName=John&lastName=Doe&email=john@example.com&password=Password123!"
```

**Success Response (201):**
```json
{
  "message": "Registration successful"
}
```

**Error Responses:**
- `400`: Missing required fields or password too short
- `409`: Email already registered

### 2. Login
- **URL**: `/login`
- **Method**: GET (show form) or POST (submit form)
- **Form Fields**:
  - `email`: User's email address (required)
  - `password`: User's password (required)
  - `rememberMe`: Boolean to remember email (optional)

**Example Request:**
```bash
curl -X POST http://localhost:5000/login \
  -d "email=john@example.com&password=Password123!"
```

**Success Response (302):**
- Redirects to `/` (dashboard)
- Sets session cookie with user information

**Error Responses:**
- `400`: Missing required fields
- `401`: Invalid email or password

### 3. Logout
- **URL**: `/logout`
- **Method**: GET
- **Result**: Clears session and redirects to login page

**Example Request:**
```bash
curl http://localhost:5000/logout
```

## API Endpoints (Require Authentication)

All API endpoints require authentication. If not authenticated, they return `401 Unauthorized`.

### Check Authentication Status
- **URL**: `/api/auth/status`
- **Method**: GET
- **Auth Required**: No

**Response:**
```json
{
  "authenticated": true,
  "user_id": 1,
  "user_email": "john@example.com",
  "user_name": "John Doe"
}
```

### Get User Profile
- **URL**: `/api/user/profile`
- **Method**: GET
- **Auth Required**: Yes

**Response:**
```json
{
  "id": 1,
  "email": "john@example.com",
  "first_name": "John",
  "last_name": "Doe",
  "company": "Acme Corp",
  "created_at": "2026-01-12 10:30:00"
}
```

### Get Detection Results
- **URL**: `/api/detections`
- **Method**: GET
- **Auth Required**: Yes

**Response:**
```json
{
  "timestamp": 1234567890,
  "total_faces": 2,
  "faces": [
    {
      "face_id": "face_1",
      "name": "John Smith",
      "confidence": 0.95,
      "state": "live",
      "is_live": true
    }
  ]
}
```

### Get Detection History
- **URL**: `/api/history`
- **Method**: GET
- **Auth Required**: Yes

**Response:**
```json
{
  "total_entries": 10,
  "history": [
    {
      "timestamp": 1234567890,
      "face_id": "face_1",
      "state": "live",
      "name": "John Smith",
      "confidence": 0.95,
      "is_live": true
    }
  ]
}
```

### Clear Detection History
- **URL**: `/api/history/clear`
- **Method**: POST
- **Auth Required**: Yes

**Response:**
```json
{
  "status": "success",
  "message": "History cleared"
}
```

### Get System Status
- **URL**: `/api/status`
- **Method**: GET
- **Auth Required**: Yes

**Response:**
```json
{
  "status": "running",
  "camera_type": "USB/Webcam",
  "camera_fps": 30.5,
  "known_faces": 5,
  "faces_loaded": ["John Smith", "Jane Doe"],
  "threads_active": true
}
```

### Get Security Alerts
- **URL**: `/api/alerts`
- **Method**: GET
- **Auth Required**: Yes
- **Query Parameters**:
  - `unacknowledged`: Set to "true" to get only unacknowledged alerts (optional)

**Response:**
```json
{
  "total_alerts": 3,
  "alerts": [
    {
      "id": 1,
      "type": "unknown_face_detected",
      "severity": "high",
      "timestamp": 1234567890,
      "message": "Unknown face detected",
      "acknowledged": false
    }
  ]
}
```

### Acknowledge Alert
- **URL**: `/api/alerts/<alert_id>/acknowledge`
- **Method**: POST
- **Auth Required**: Yes

**Response:**
```json
{
  "status": "success",
  "message": "Alert 1 acknowledged"
}
```

### Clear All Alerts
- **URL**: `/api/alerts/clear`
- **Method**: POST
- **Auth Required**: Yes

**Response:**
```json
{
  "status": "success",
  "message": "All alerts cleared"
}
```

### Video Feed Stream
- **URL**: `/video_feed`
- **Method**: GET
- **Auth Required**: Yes
- **Content-Type**: multipart/x-mixed-replace (MJPEG stream)

## Database Schema

### Users Table
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    company TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

## Session Configuration

Sessions are configured with:
- **Secure Cookies**: HTTPOnly, Secure, SameSite=Lax
- **Secret Key**: Configurable via `SECRET_KEY` environment variable
- **Default Secret**: "dev-secret-key-change-in-production" (change in production!)

## Password Security

- Passwords are hashed using `werkzeug.security.generate_password_hash()`
- Uses bcrypt-based hashing with salt
- Verification uses `werkzeug.security.check_password_hash()`
- Minimum password length: 8 characters

## Frontend Features

### Login Page (`login_page.html`)
- Email and password input
- "Remember me" checkbox (stores email in localStorage)
- Error message display
- Loading indicator
- Social login buttons (UI ready, not implemented)
- Forgot password link (UI ready)

### Register Page (`register_page.html`)
- First name, last name, email, password fields
- Company field (optional)
- Password strength indicator (visual feedback)
- Password confirmation validation
- Terms and privacy agreement checkbox
- Error and success message display
- Loading indicator

### Dashboard (`index.html`)
- User info display in header with avatar
- Logout button
- Protected by login requirement
- User name and email shown
- Avatar with initials

## Environment Variables

```bash
# Secret key for session encryption (required in production)
SECRET_KEY=your-secret-key-here

# Flask settings
FLASK_ENV=production
FLASK_DEBUG=False
```

## Testing the System

### Create a Test Account
```bash
curl -X POST http://localhost:5000/register \
  -d "firstName=Test&lastName=User&email=test@example.com&password=TestPassword123!"
```

### Login with Test Account
```bash
curl -c cookies.txt -X POST http://localhost:5000/login \
  -d "email=test@example.com&password=TestPassword123!"
```

### Test Protected API
```bash
curl -b cookies.txt http://localhost:5000/api/user/profile
```

### Logout
```bash
curl -b cookies.txt http://localhost:5000/logout
```

## Security Considerations

1. **HTTPS**: Use HTTPS in production (set `SESSION_COOKIE_SECURE=True`)
2. **Secret Key**: Change the default secret key in production
3. **Database**: Ensure `facefetch_users.db` is backed up and secured
4. **Password Policy**: Consider implementing stronger password requirements
5. **Rate Limiting**: Consider implementing rate limiting on login/register endpoints
6. **CORS**: Review CORS settings if accessing from frontend domain

## Troubleshooting

### "Email already registered" error
- The email is already in the system
- Try with a different email address
- Check the SQLite database: `sqlite3 facefetch_users.db "SELECT * FROM users;"`

### "Invalid email or password" error
- Verify the email and password are correct
- Check that the account was created successfully
- Ensure no typos in credentials

### Session not persisting
- Check that cookies are enabled in browser
- Verify `SECRET_KEY` is set correctly
- Check browser console for cookie errors

### Database locked error
- Ensure only one instance of the application is running
- Delete `facefetch_users.db-journal` if it exists
- Restart the application

