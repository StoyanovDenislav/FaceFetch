"""
Authentication API Routes
Handles user login, registration, and session management
"""
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session, make_response, current_app
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector
from mysql.connector import errorcode
import os
from functools import wraps
import jwt
import datetime
import hashlib

auth_bp = Blueprint('auth', __name__)

# JWT Configuration
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', os.environ.get('SECRET_KEY', 'dev-jwt-secret'))
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Database configuration for Users
# Using shared 'facefetch' database
MYSQL_HOST = os.environ.get('DB_HOST', os.environ.get('MYSQL_HOST', 'localhost'))
MYSQL_PORT = int(os.environ.get('DB_PORT', os.environ.get('MYSQL_PORT', '3306')))
MYSQL_USER = os.environ.get('DB_USER', os.environ.get('MYSQL_USER', 'root'))
MYSQL_PASSWORD = os.environ.get('DB_PASSWORD', os.environ.get('MYSQL_PASSWORD', ''))
MYSQL_DB_USERS = os.environ.get('DB_NAME', 'facefetch')

def _get_user_db_connection(database: bool = True):
    """Create a MySQL connection to the user database."""
    cfg = {
        'host': MYSQL_HOST,
        'port': MYSQL_PORT,
        'user': MYSQL_USER,
        'password': MYSQL_PASSWORD,
    }
    if database:
        cfg['database'] = MYSQL_DB_USERS
    return mysql.connector.connect(**cfg)

def init_auth_db():
    """Initialize User database and tables."""
    # Ensure database exists
    conn = _get_user_db_connection(database=False)
    try:
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DB_USERS} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        conn.commit()
    finally:
        cur.close()
        conn.close()

    # Ensure table exists with merged schema
    conn = _get_user_db_connection(database=True)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(255) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                first_name VARCHAR(100) NOT NULL,
                last_name VARCHAR(100) NOT NULL,
                company VARCHAR(255),
                name VARCHAR(120) GENERATED ALWAYS AS (CONCAT(first_name, ' ', last_name)) VIRTUAL,
                active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()

def generate_tokens(user_id):
    """Generate access and refresh tokens"""
    access_token = jwt.encode({
        'user_id': user_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    }, JWT_SECRET_KEY, algorithm='HS256')
    
    # Refresh token (random string)
    refresh_token = hashlib.sha256(os.urandom(64)).hexdigest()
    
    return access_token, refresh_token

def store_refresh_token(user_id, refresh_token):
    """Store refresh token hash in database"""
    token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
    expires_at = datetime.datetime.utcnow() + datetime.timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    conn = _get_user_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO refresh_tokens (user_id, token_hash, expires_at) VALUES (%s, %s, %s)',
            (user_id, token_hash, expires_at)
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def remove_refresh_token(refresh_token):
    """Remove refresh token from database"""
    token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
    conn = _get_user_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM refresh_tokens WHERE token_hash = %s', (token_hash,))
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def login_required(f):
    """Decorator to require login via JWT cookie"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.cookies.get('access_token')
        
        if not token:
            if request.path.startswith('/api/'):
                return jsonify({'message': 'Authentication required'}), 401
            return redirect(url_for('auth.login'))
        
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
            # Add user_id to session and hydrate display info from DB for templates
            session['user_id'] = payload['user_id']
            if 'user_name' not in session or 'user_role' not in session:
                user = get_user_by_id(payload['user_id'])
                if user:
                    # Prefer full name, fall back to email if missing
                    display_name = f"{user.get('first_name','').strip()} {user.get('last_name','').strip()}".strip()
                    session['user_name'] = display_name or user.get('email', 'User')
                    session['user_role'] = user.get('role', 'viewer')
        except jwt.ExpiredSignatureError:
            if request.path.startswith('/api/'):
                return jsonify({'message': 'Token expired'}), 401
            return redirect(url_for('auth.login'))
        except jwt.InvalidTokenError:
            if request.path.startswith('/api/'):
                return jsonify({'message': 'Invalid token'}), 401
            return redirect(url_for('auth.login'))
            
        return f(*args, **kwargs)
    return decorated_function

def get_user_by_email(email):
    """Get user by email from DB."""
    conn = _get_user_db_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT id, email, password_hash, first_name, last_name, company, role, created_at FROM users WHERE email = %s', (email,))
        return cursor.fetchone()
    finally:
        cursor.close()
        conn.close()

def get_user_by_id(user_id):
    """Get user by ID from DB."""
    conn = _get_user_db_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT id, email, first_name, last_name, company, role, created_at FROM users WHERE id = %s', (user_id,))
        return cursor.fetchone()
    finally:
        cursor.close()
        conn.close()

def create_user(email, password, first_name, last_name, company='', role='viewer'):
    """Create a new user. Role defaults to 'viewer'."""
    conn = _get_user_db_connection()
    try:
        cursor = conn.cursor()
        password_hash = generate_password_hash(password)
        cursor.execute(
            'INSERT INTO users (email, password_hash, first_name, last_name, company, role) VALUES (%s, %s, %s, %s, %s, %s)',
            (email, password_hash, first_name, last_name, company, role)
        )
        conn.commit()
        return True
    except mysql.connector.Error as e:
        if e.errno == errorcode.ER_DUP_ENTRY:
            return False
        raise
    finally:
        cursor.close()
        conn.close()

def role_required(required_role):
    """Decorator to require a specific role"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_role' not in session:
                 # Try to refresh from DB if we have user_id but lost role in session (unlikely)
                 if 'user_id' in session:
                     user = get_user_by_id(session['user_id'])
                     if user:
                         session['user_role'] = user['role']
                     else:
                         return jsonify({'message': 'Access denied'}), 403
                 else:
                    return jsonify({'message': 'Authentication required'}), 401
            
            # Simple check: operator can do everything, viewer is limited
            # If required is 'operator' and user is 'viewer', deny
            if required_role == 'operator' and session.get('user_role') != 'operator':
                return jsonify({'message': 'Operator access required'}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# --- Routes ---

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Login route"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        
        if not email or not password:
            return jsonify({'message': 'Email and password are required'}), 400
        
        user = get_user_by_email(email)
        
        if user and check_password_hash(user['password_hash'], password):
            # Generate JWT tokens
            access_token, refresh_token = generate_tokens(user['id'])
            
            # Store refresh token in DB
            store_refresh_token(user['id'], refresh_token)
            
            # Create response
            response = make_response(redirect(url_for('index')))
            
            # Set cookies (HttpOnly, Secure)
            response.set_cookie(
                'access_token', 
                access_token, 
                httponly=True, 
                secure=True, 
                samesite='Lax', # Lax needed for top-level navigation redirects 
                max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60
            )
            
            # Refresh token cookie (stricter path)
            response.set_cookie(
                'refresh_token',
                refresh_token,
                httponly=True,
                secure=True,
                samesite='Lax',
                path='/api/auth', # Only send to auth routes
                max_age=REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
            )
            
            # Keep minimal session for template rendering (or remove if using g.user)
            session['user_id'] = user['id']
            session['user_name'] = f"{user['first_name']} {user['last_name']}"
            session['user_role'] = user['role']
            
            return response
        else:
            return jsonify({'message': 'Invalid email or password'}), 401
    
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    return render_template('login_page.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Register route"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        first_name = request.form.get('firstName', '').strip()
        last_name = request.form.get('lastName', '').strip()
        company = request.form.get('company', '').strip()
        # Optional: Allow selecting role if not public, but for now default 'viewer'
        
        # Validation
        if not all([email, password, first_name, last_name]):
            return jsonify({'message': 'All required fields must be filled'}), 400
        
        # Email validation
        if '@' not in email or '.' not in email:
            return jsonify({'message': 'Please enter a valid email address'}), 400
        
        if len(password) < 8:
            return jsonify({'message': 'Password must be at least 8 characters long'}), 400
        
        # Default role is viewer
        if not create_user(email, password, first_name, last_name, company, role='viewer'):
            return jsonify({'message': 'Email already registered'}), 409
        
        return jsonify({'message': 'Registration successful'}), 201
    
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    return render_template('register_page.html')

@auth_bp.route('/logout')
def logout():
    """Logout route"""
    # Remove refresh token from DB if present
    refresh_token = request.cookies.get('refresh_token')
    if refresh_token:
        remove_refresh_token(refresh_token)
    
    session.clear()
    
    response = make_response(render_template('logout_page.html'))
    # Clear tokens on all paths we might have used
    response.set_cookie('access_token', '', expires=0)
    response.set_cookie('refresh_token', '', expires=0, path='/')
    response.set_cookie('refresh_token', '', expires=0, path='/api/auth')
    return response

@auth_bp.route('/api/auth/status')
def auth_status():
    """Check authentication status"""
    if 'user_id' in session:
        return jsonify({
            'authenticated': True,
            'user_id': session.get('user_id'),
            'user_email': session.get('user_email'),
            'user_name': session.get('user_name'),
            'user_role': session.get('user_role')
        })
    else:
        return jsonify({'authenticated': False}), 401

@auth_bp.route('/api/user/profile')
@login_required
def get_user_profile():
    """Get current user profile"""
    user = get_user_by_id(session.get('user_id'))
    if user:
        return jsonify({
            'id': user['id'],
            'email': user['email'],
            'first_name': user['first_name'],
            'last_name': user['last_name'],
            'company': user['company'],
            'role': user['role'],
            'created_at': str(user['created_at'])
        })
    return jsonify({'message': 'User not found'}), 404
