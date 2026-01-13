# FaceFetch Authentication System

## Overview
A complete authentication system has been implemented for the FaceFetch facial recognition application. The system includes user registration, login, and logout functionality with secure password handling.

## Features

### 1. **Login Page** (`login_page.html`)
- Email and password authentication
- "Remember me" functionality (saves email locally)
- Error messages for invalid credentials
- Responsive design with gradient background
- Social login buttons (Google & GitHub - ready for integration)
- Forgot password link
- Sign up redirect
- Loading spinner during authentication
- Password field with secure input masking

### 2. **Register Page** (`register_page.html`)
- User registration with email and password
- Password strength indicator (weak/medium/strong)
- Real-time password strength checking
- Password confirmation validation
- First name, last name, and optional company fields
- Terms and privacy policy agreement checkbox
- Email validation
- Success and error messages
- Responsive design
- Auto-redirect to login after successful registration

### 3. **Logout Page** (`logout_page.html`)
- Confirmation page after logout
- Links back to login page
- FaceFetch feature overview
- Professional design with wave emoji
- Session cleanup on logout

### 4. **Dashboard Update** (`index.html`)
- User info display in header (name and email)
- User avatar with initials
- Logout button
- Session-based user tracking
- Responsive header layout
- Elegant dropdown-style user info panel

## Backend Implementation

### Database (MySQL)
- Automatic database and table creation on first startup
- Configure connection via environment variables:
  - `MYSQL_HOST` (default: `localhost`)
  - `MYSQL_PORT` (default: `3306`)
  - `MYSQL_USER` (default: `facefetch`)
  - `MYSQL_PASSWORD` (default: `facefetch`)
  - `MYSQL_DB` (default: `facefetch`)
- Users table schema:
  - `id` INT AUTO_INCREMENT PRIMARY KEY
  - `email` VARCHAR(255) UNIQUE NOT NULL
  - `password_hash` VARCHAR(255) NOT NULL
  - `first_name` VARCHAR(100) NOT NULL
  - `last_name` VARCHAR(100) NOT NULL
  - `company` VARCHAR(255)
  - `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP

### Flask Routes

#### `GET/POST /login`
- **GET**: Serves the login page
- **POST**: Processes login credentials
  - Validates email and password
  - Checks against database
  - Creates session on success
  - Returns JSON error on failure

#### `GET/POST /register`
- **GET**: Serves the registration page
- **POST**: Processes registration
  - Validates all required fields
  - Checks password strength (min 8 characters)
  - Verifies email uniqueness
  - Hashes password with `werkzeug.security`
  - Returns success/error JSON
  - Prevents duplicate email registration

#### `GET /logout`
- Clears user session
- Redirects to login page
- Serves logout confirmation page

### Security Features
- Session-based authentication using Flask sessions
- Password hashing with `werkzeug.security.generate_password_hash`
- Secure session cookies (`httponly`, `secure`, `samesite`)
- CSRF protection ready
- Protected endpoints requiring login
- Password strength requirements (minimum 8 characters)

## Configuration

### Environment Variables
```python
SECRET_KEY   # Set this to a strong secret key in production
# MySQL Connection
MYSQL_HOST   # e.g., 'localhost'
MYSQL_PORT   # e.g., '3306'
MYSQL_USER   # e.g., 'facefetch'
MYSQL_PASSWORD # e.g., 'facefetch'
MYSQL_DB     # e.g., 'facefetch'
```

### Session Configuration
```python
SESSION_COOKIE_SECURE = True      # HTTPS only in production
SESSION_COOKIE_HTTPONLY = True    # Prevent JavaScript access
SESSION_COOKIE_SAMESITE = 'Lax'   # CSRF protection
```

## Database Initialization

The MySQL database and `users` table are automatically created on the first run:
```python
init_db()  # Called at application startup
```

## Usage Example

### Starting the Application
```bash
python web_server.py
```

### User Registration Flow
1. User navigates to `/register`
2. Fills in registration form
3. Password strength is shown in real-time
4. Upon submission, user data is stored in database
5. Redirects to login page

### User Login Flow
1. User navigates to `/login`
2. Enters credentials
3. System validates against database
4. Creates session on success
5. Redirects to dashboard
6. Displays user info in header

### User Logout Flow
1. User clicks "Logout" button in header
2. Session is cleared
3. Redirects to logout confirmation page
4. User can log back in

## Styling
- Consistent gradient background: `linear-gradient(135deg, #667eea 0%, #764ba2 100%)`
- Professional white cards with shadow effects
- Responsive design (mobile-first)
- Smooth transitions and animations
- Accent colors:
  - Primary: `#667eea` (Purple)
  - Secondary: `#764ba2` (Dark purple)
  - Success: `#28a745` (Green)
  - Error: `#dc3545` (Red)
  - Warning: `#ffc107` (Yellow)

## Future Enhancements

### Ready for Implementation
- [ ] Social login (Google, GitHub, OAuth2)
- [ ] Email verification
- [ ] Password reset functionality
- [ ] Two-factor authentication (2FA)
- [ ] User profile management
- [ ] Role-based access control
- [ ] Remember me with secure tokens
- [ ] Login history tracking
- [ ] Session management (multiple devices)

## Browser Support
- Chrome/Chromium 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## API Response Format

### Successful Login
```json
{
  "redirect": "/"
}
```

### Login Error
```json
{
  "message": "Invalid email or password"
}
```

### Registration Success
```json
{
  "message": "Registration successful"
}
```

### Registration Error
```json
{
  "message": "Email already registered"
}
```

## Security Checklist
- ✅ Password hashing (bcrypt via werkzeug)
- ✅ Secure session configuration
- ✅ HTTPS-ready configuration
- ✅ CSRF protection structure
- ✅ Input validation
- ✅ SQL injection prevention (parameterized queries)
- ✅ Session timeout configuration
- ⚠️ TODO: Add rate limiting for login attempts
- ⚠️ TODO: Add account lockout after failed attempts
- ⚠️ TODO: Email verification before account activation

## File Structure
```
FaceFetch/
├── web_server.py           (Updated with auth routes)
├── templates/
│   ├── index.html          (Updated with user header)
│   ├── login_page.html     (New)
│   ├── register_page.html  (New)
│   └── logout_page.html    (New)
└── facefetch_users.db      (Auto-created on first run)
```

## Dependencies Added
```python
from flask import session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
```

## Testing Checklist
- [ ] User registration with valid data
- [ ] Registration with duplicate email
- [ ] Password strength validation
- [ ] Login with correct credentials
- [ ] Login with incorrect password
- [ ] Session persistence
- [ ] Logout functionality
- [ ] Protected dashboard access
- [ ] Mobile responsiveness
- [ ] Error message display
