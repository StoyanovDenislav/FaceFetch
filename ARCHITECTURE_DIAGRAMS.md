# FaceFetch Authentication System - Architecture & Flow Diagrams

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser / Client                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │ Login Page   │    │Register Page │    │  Dashboard   │    │
│  │              │    │              │    │              │    │
│  │ - Email      │    │ - First Name │    │ - User Info  │    │
│  │ - Password   │    │ - Last Name  │    │ - Video Feed │    │
│  │ - Remember   │    │ - Email      │    │ - Statistics │    │
│  │   Me         │    │ - Password   │    │ - History    │    │
│  │ - Login BTN  │    │ - Company    │    │ - Logout BTN │    │
│  │              │    │ - Terms      │    │              │    │
│  │ - Register   │    │ - Register   │    │ (Protected)  │    │
│  │   Link       │    │   BTN        │    │              │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│         │                   │                     │             │
│         └───────────────────┼─────────────────────┘             │
│                             │ HTTP/AJAX                         │
└─────────────────────────────┼─────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Flask Web Server                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                  Route Handlers                           │ │
│  │                                                           │ │
│  │  GET  /            ──────┬──► Check Session              │ │
│  │       /login              │   ├─► Valid? → Dashboard     │ │
│  │  POST /login              │   └─► Invalid? → Login Page  │ │
│  │       /register           │                              │ │
│  │  POST /register     ──────┤   Validate Input             │ │
│  │       /logout             │   ├─► Valid? → Hash Pass &   │ │
│  │  GET  /logout       ──────┤       Store User             │ │
│  │                           │   └─► Error? → Error Msg    │ │
│  │  Protected Routes:        │                              │ │
│  │  /video_feed       ──────┬┴─► Check @login_required     │ │
│  │  /api/*                  │   ├─► Valid? → Return Data   │ │
│  │                          │   └─► Invalid? → 401 Error   │ │
│  └───────────────────────────────────────────────────────────┘ │
│                             │                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              Session Management                          │ │
│  │                                                           │ │
│  │  ├─ user_id: Integer (Primary Key)                       │ │
│  │  ├─ user_email: String (User's Email)                    │ │
│  │  ├─ user_name: String (First + Last Name)                │ │
│  │  └─ Session Cookie (Secure, HTTPOnly, SameSite=Lax)     │ │
│  └───────────────────────────────────────────────────────────┘ │
│                             │                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │            Password Hashing & Verification                │ │
│  │                                                           │ │
│  │  Registration: Plain Text → Bcrypt Hash → DB             │ │
│  │  Login: Plain Text + DB Hash → Verify → Valid/Invalid    │ │
│  │                                                           │ │
│  │  Functions:                                              │ │
│  │  ├─ generate_password_hash(password)                     │ │
│  │  └─ check_password_hash(hash, password)                  │ │
│  └───────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SQLite Database                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Table: users                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ id │ email         │ password_hash │ first_name │ ...  │  │
│  ├────┼───────────────┼───────────────┼────────────┼─────┤  │
│  │ 1  │ john@ex.com   │ $2b$12$...    │ John       │ ... │  │
│  │ 2  │ jane@ex.com   │ $2b$12$...    │ Jane       │ ... │  │
│  │ 3  │ bob@ex.com    │ $2b$12$...    │ Bob        │ ... │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Columns:                                                       │
│  ├─ id: PRIMARY KEY AUTO INCREMENT                            │
│  ├─ email: UNIQUE NOT NULL                                    │
│  ├─ password_hash: NOT NULL (Bcrypt)                          │
│  ├─ first_name: NOT NULL                                      │
│  ├─ last_name: NOT NULL                                       │
│  ├─ company: TEXT (Optional)                                  │
│  └─ created_at: TIMESTAMP DEFAULT                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Authentication Flow Diagram

```
                            START
                              │
                              ▼
                    ┌─────────────────┐
                    │ User Visits App │
                    │ (Not Logged In) │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────────┐
                    │ Redirect to /login  │
                    │ (By @login_required │
                    │  decorator or code) │
                    └────────┬────────────┘
                             │
                  ┌──────────┴──────────┐
                  │                     │
                  ▼                     ▼
        ┌──────────────────┐  ┌──────────────────┐
        │  Existing User   │  │   New User?      │
        │                  │  │                  │
        │ Click "Sign In"  │  │ Click "Sign Up" │
        └────────┬─────────┘  └────────┬─────────┘
                 │                     │
                 ▼                     ▼
        ┌──────────────────┐  ┌──────────────────┐
        │  Login Form      │  │  Register Form   │
        │                  │  │                  │
        │ - Email          │  │ - First Name     │
        │ - Password       │  │ - Last Name      │
        │ - Remember Me    │  │ - Email          │
        └────────┬─────────┘  │ - Password       │
                 │            │ - Company        │
                 │            │ - Confirm Pass   │
                 │            │ - Terms Agree    │
                 │            └────────┬─────────┘
                 │                     │
                 │   ┌─────────────────┘
                 │   │
                 ▼   ▼
        ┌──────────────────────────────┐
        │  Validate Input              │
        │                              │
        │  ├─ Email format (both)      │
        │  ├─ Required fields (both)   │
        │  ├─ Password 8+ chars        │
        │  ├─ Passwords match (reg)    │
        │  ├─ Terms agreed (reg)       │
        │  └─ Unique email (reg)       │
        └────────┬─────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
    ┌────────────┐    ┌────────────────┐
    │ Invalid ✗  │    │ Valid ✓         │
    └────────┬───┘    └────────┬────────┘
             │                 │
             ▼                 ▼
    ┌────────────────┐   ┌──────────────────┐
    │ Show Error Msg │   │ Hash Password    │
    │ & Stay on Form │   │ (Bcrypt + Salt)  │
    └────────────────┘   └────────┬─────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
            ┌───────────────────┐      ┌──────────────────┐
            │ LOGIN Path        │      │ REGISTER Path    │
            │                   │      │                  │
            │ 1. Check DB for   │      │ 1. Check Email   │
            │    email          │      │    Unique        │
            │ 2. Compare        │      │ 2. Create User   │
            │    password hash  │      │    in DB         │
            └────────┬──────────┘      └────────┬─────────┘
                     │                         │
         ┌───────────┴──────────┐              │
         │                      │              │
         ▼                      ▼              ▼
    ┌─────────┐         ┌──────────┐   ┌────────────────┐
    │ Valid ✓ │         │Invalid ✗ │   │ Success ✓      │
    │         │         │          │   │                │
    │ Create  │         │ Return   │   │ Show Success   │
    │ Session │         │ 401 Error│   │ Msg            │
    └────┬────┘         └──────────┘   └────────┬───────┘
         │                                      │
         ▼                                      ▼
    ┌─────────────────┐             ┌──────────────────┐
    │ Store in        │             │ Redirect to      │
    │ Session:        │             │ /login (after    │
    │ - user_id       │             │ 2 seconds)       │
    │ - user_email    │             └────────┬─────────┘
    │ - user_name     │                      │
    └────────┬────────┘                      │
             │          ┌────────────────────┘
             │          │
             ▼          ▼
        ┌──────────────────────┐
        │ Show Login Form      │
        │ (for new users or    │
        │  redirect from reg)  │
        └────────┬─────────────┘
                 │
                 ├─ User enters credentials
                 │  (cycle repeats from Login Path)
                 │
                 └──► Credentials Valid
                      │
                      ▼
        ┌──────────────────────────┐
        │ Session Created          │
        │                          │
        │ Cookie Set:              │
        │ - secure=True            │
        │ - httponly=True          │
        │ - samesite=Lax           │
        └────────┬─────────────────┘
                 │
                 ▼
        ┌──────────────────────────┐
        │ Redirect to /            │
        │ (Dashboard)              │
        └────────┬─────────────────┘
                 │
                 ▼
        ┌──────────────────────────┐
        │ Dashboard Loaded         │
        │                          │
        │ ✓ Check @login_required  │
        │ ✓ Session Valid          │
        │ ✓ Display User Info      │
        │ ✓ Load Video Feed        │
        │ ✓ API Access Allowed     │
        │                          │
        │ User can:                │
        │ ├─ View Live Feed        │
        │ ├─ Access APIs           │
        │ ├─ See Statistics        │
        │ ├─ View History          │
        │ └─ Click Logout          │
        └────────┬─────────────────┘
                 │
                 ▼
        ┌──────────────────────────┐
        │ User Clicks Logout       │
        └────────┬─────────────────┘
                 │
                 ▼
        ┌──────────────────────────┐
        │ Clear Session            │
        │ - Remove all data        │
        │ - Invalidate Cookie      │
        └────────┬─────────────────┘
                 │
                 ▼
        ┌──────────────────────────┐
        │ Redirect to /login       │
        │ (Logout Page first)      │
        └────────┬─────────────────┘
                 │
                 ▼
                DONE
```

## Protected Resource Access Flow

```
User with Session                    User without Session
(Logged In)                          (Not Logged In)
      │                                    │
      │                                    │
      ▼                                    ▼
Request /api/detections          Request /api/detections
      │                                    │
      ▼                                    ▼
Check Session                    Check Session
(session['user_id'])             (session['user_id'])
      │                                    │
      ▼                                    ▼
   FOUND                              NOT FOUND
      │                                    │
      ▼                                    ▼
@login_required                  @login_required
Passes ✓                         Fails ✗
      │                                    │
      ▼                                    ▼
Execute API Handler              Return 401 Unauthorized
      │                                    │
      ▼                                    ▼
Return JSON Data                 Return Error JSON
      │                                    │
      ▼                                    ▼
User receives data               Browser shows error
with HTTP 200                    message and offers
                                 to login again
```

## Security Layers

```
┌─────────────────────────────────────────────────────┐
│              Browser / Client                       │
│                                                     │
│  HTTPS Transport Layer                              │
│  └─ Encrypts all communication                      │
│                                                     │
│  CORS Policy (if configured)                        │
│  └─ Controls cross-origin requests                  │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│              Web Server                             │
│                                                     │
│  1. Authentication Check                            │
│     ├─ Session validation                           │
│     └─ @login_required decorator                    │
│                                                     │
│  2. Input Validation                                │
│     ├─ Email format check                           │
│     ├─ Password length check                        │
│     ├─ Required field check                         │
│     └─ SQL injection prevention                     │
│                                                     │
│  3. Password Security                               │
│     ├─ Bcrypt hashing                               │
│     ├─ Automatic salt generation                    │
│     └─ Secure comparison                            │
│                                                     │
│  4. Session Security                                │
│     ├─ Secure flag (HTTPS only)                     │
│     ├─ HTTPOnly flag (JS protected)                 │
│     └─ SameSite=Lax flag (CSRF protection)          │
│                                                     │
│  5. Database Security                               │
│     ├─ Parameterized queries                        │
│     ├─ Password never stored plaintext              │
│     └─ Email uniqueness enforced                    │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│              Database                               │
│                                                     │
│  SQLite with:                                       │
│  ├─ Unique constraints on email                     │
│  ├─ Hashed passwords                                │
│  ├─ Timestamp tracking                              │
│  └─ Indexed lookups                                 │
└─────────────────────────────────────────────────────┘
```

## Session Lifecycle

```
Login Successful
      │
      ▼
┌─────────────────────────────────┐
│ Session Created                 │
│ session['user_id'] = 1          │
│ session['user_email'] = ...     │
│ session['user_name'] = ...      │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Session Cookie Set              │
│ - Name: session                 │
│ - Value: encrypted session data │
│ - Secure=True (HTTPS only)      │
│ - HttpOnly=True (No JS access)  │
│ - SameSite=Lax (CSRF protection)│
└──────────────┬──────────────────┘
               │
               ▼
        ┌──────────────┐
        │ User Browsing│
        │   Dashboard  │
        └──────┬───────┘
               │
        Each Request:
        ├─ Browser sends session cookie
        ├─ Server validates session
        ├─ Request proceeds if valid
        └─ Server updates session
               │
               ▼
        ┌──────────────────┐
        │ User Clicks      │
        │ Logout Button    │
        └──────┬───────────┘
               │
               ▼
        ┌──────────────────────────┐
        │ Session Cleared          │
        │ session.clear()          │
        │ - Removes user_id        │
        │ - Removes user_email     │
        │ - Removes user_name      │
        └──────┬───────────────────┘
               │
               ▼
        ┌──────────────────────────┐
        │ Cookie Invalidated       │
        │ - Set Max-Age=0          │
        │ - Browser deletes cookie │
        └──────┬───────────────────┘
               │
               ▼
        ┌──────────────────────────┐
        │ Redirect to /login       │
        │ (Not Authenticated)      │
        └──────────────────────────┘
```

## Database Query Flow

### Registration Flow
```
User Input: Email, Password, Name
      │
      ▼
validate_email_format()
      │
      ▼
hash_password(password, bcrypt + salt)
      │
      ▼
INSERT INTO users (email, password_hash, first_name, last_name, ...)
      │
      ├─ Unique Constraint Check
      │  └─ Email already exists? → 409 Conflict
      │
      └─ Success
         └─ Return: User Created
```

### Login Flow
```
User Input: Email, Password
      │
      ▼
SELECT * FROM users WHERE email = ?
      │
      ├─ User found?
      │  ├─ Yes → Continue
      │  └─ No → 401 Invalid Credentials
      │
      ▼
check_password_hash(password, stored_hash)
      │
      ├─ Match?
      │  ├─ Yes → Create Session → Redirect /
      │  └─ No → 401 Invalid Credentials
      │
      └─ Complete
```

