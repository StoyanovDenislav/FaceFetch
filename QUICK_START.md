# FaceFetch Authentication System - Quick Start Guide

## Installation

### 1. Install Required Dependencies

The authentication system requires werkzeug for password hashing. Make sure these are in your `requirements.txt`:

```
Flask==3.1.2
Werkzeug==3.1.4
```

These are already included in the project's requirements.txt.

### 2. Update Requirements
```bash
pip install -r requirements.txt
```

## Running the Application

### 1. Start the Web Server
```bash
python web_server.py
```

You should see output like:
```
==================================================
Facial Recognition Web Server (Optimized)
==================================================
...
Access the web interface at:
  http://localhost:5000
```

### 2. Open in Browser

Navigate to: `http://localhost:5000`

You will be redirected to the login page automatically.

## First Time Usage

### Create an Account

1. Click "Sign up here" on the login page
2. Fill in your details:
   - First Name: Your first name
   - Last Name: Your last name
   - Email: Your email address
   - Password: At least 8 characters (with uppercase, lowercase, numbers, special chars for strong password)
   - Company: Optional
3. Check the Terms and Privacy agreement
4. Click "Create Account"
5. You'll be redirected to login

### Login

1. Enter your email address
2. Enter your password
3. Optionally check "Remember me" to save your email
4. Click "Sign In"

### Access Dashboard

After login, you'll see the facial recognition dashboard with:
- Live camera feed
- System status
- Detected faces
- Detection history
- Security alerts

## Features

### User Authentication
- Secure password hashing with bcrypt
- Session management with secure cookies
- Password strength indicator during registration
- Email validation
- Unique email enforcement

### Protected Resources
All API endpoints and video feed require authentication:
- `/` - Dashboard
- `/video_feed` - Live camera stream
- `/api/detections` - Detection results
- `/api/history` - Detection history
- `/api/status` - System status
- `/api/alerts` - Security alerts

### User Session
- User information displayed in dashboard header
- Avatar with name initials
- "Remember me" functionality (saves email in browser)
- Logout button

## Database (MySQL)

The authentication system now uses MySQL. Configure connection via environment variables (see `.env.example`):

```
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=facefetch
MYSQL_PASSWORD=facefetch
MYSQL_DB=facefetch
```

On first run, the application will automatically create the database (if missing) and the `users` table.

### Viewing Users
Use any MySQL client (Workbench, CLI, or admin UI). Example with MySQL CLI:
```bash
mysql -h localhost -u facefetch -p -D facefetch -e "SELECT email, first_name, last_name, created_at FROM users;"
```

### Deleting a User (if needed)
```bash
mysql -h localhost -u facefetch -p -D facefetch -e "DELETE FROM users WHERE email='user@example.com';"
```

## Testing

### Run Automated Tests
```bash
python test_auth.py
```

This will test:
- User registration
- Login/logout
- Protected API endpoints
- Authentication status
- User profile retrieval

### Manual Testing with cURL

**Register:**
```bash
curl -X POST http://localhost:5000/register \
  -d "firstName=John&lastName=Doe&email=john@example.com&password=Password123!" \
  -d "company=MyCompany"
```

**Login:**
```bash
curl -c cookies.txt -X POST http://localhost:5000/login \
  -d "email=john@example.com&password=Password123!"
```

**Check Auth Status:**
```bash
curl -b cookies.txt http://localhost:5000/api/auth/status
```

**Get User Profile:**
```bash
curl -b cookies.txt http://localhost:5000/api/user/profile
```

**Get System Status:**
```bash
curl -b cookies.txt http://localhost:5000/api/status
```

## Common Issues

### "Email already registered"
- The email is already in the system
- Use a different email or delete the user from the database

### "Invalid email or password"
- Double-check your credentials
- Ensure you created an account first

### Session not working
- Clear browser cookies
- Check browser console for errors
- Ensure cookies are enabled

### Port 5000 already in use
Change the port in `web_server.py`:
```python
serve(app, host='0.0.0.0', port=5001, threads=backend_threads)
```

## Production Deployment

### Security Checklist
- [ ] Set `SECRET_KEY` environment variable
- [ ] Enable HTTPS (set `SESSION_COOKIE_SECURE = True`)
- [ ] Use a production WSGI server (Waitress is already included)
- [ ] Backup the SQLite database regularly
- [ ] Use a stronger password policy
- [ ] Consider rate limiting on login/register endpoints
- [ ] Set up database encryption
- [ ] Review CORS policies

### Environment Setup
```bash
export SECRET_KEY="your-secure-random-key-here"
export FLASK_ENV="production"
```

### Run with Waitress
The application automatically uses Waitress (production WSGI server) if available. It will show:
```
Using Waitress production server
```

## File Structure

```
FaceFetch/
├── web_server.py                 # Main Flask application with auth routes
├── requirements.txt              # Python dependencies
├── facefetch_users.db           # SQLite database (created automatically)
├── test_auth.py                 # Authentication testing script
├── AUTHENTICATION_GUIDE.md       # Detailed API documentation
├── QUICK_START.md               # This file
├── templates/
│   ├── login_page.html          # Login form
│   ├── register_page.html       # Registration form
│   ├── logout_page.html         # Logout confirmation
│   └── index.html               # Dashboard (protected)
└── ...other files...
```

## Next Steps

1. **Configure Email (Optional)**
   - Implement password reset functionality
   - Send welcome emails

2. **User Profile Management**
   - Add profile edit page
   - Password change functionality
   - Account deletion

3. **Advanced Security**
   - Two-factor authentication
   - OAuth/Social login
   - API key management for external integrations

4. **User Management (Admin Panel)**
   - User list view
   - Account management
   - Activity logging

## Support

For issues or questions:
1. Check `AUTHENTICATION_GUIDE.md` for detailed API documentation
2. Review error messages in the browser console
3. Check Flask application logs in terminal
4. Run `test_auth.py` to validate the system

## Password Requirements

During registration, passwords should:
- Be at least 8 characters long
- Contain uppercase letters (A-Z)
- Contain lowercase letters (a-z)
- Contain numbers (0-9)
- Contain special characters (!@#$%^&*)

The strength indicator shows:
- **Weak**: Less than 8 characters or minimal variety
- **Medium**: 8+ characters with some variety
- **Strong**: 8+ characters with all character types

