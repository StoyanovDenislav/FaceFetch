"""
Authentication API Routes
Handles user login, registration, and session management
"""
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector
from mysql.connector import errorcode
import os
from functools import wraps

auth_bp = Blueprint('auth', __name__)

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

def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            # If it's an API request, return JSON
            if request.path.startswith('/api/'):
                return jsonify({'message': 'Authentication required'}), 401
            # Otherwise redirect to login page
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

def get_user_by_email(email):
    """Get user by email from DB."""
    conn = _get_user_db_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT id, email, password_hash, first_name, last_name, company, created_at FROM users WHERE email = %s', (email,))
        return cursor.fetchone()
    finally:
        cursor.close()
        conn.close()

def get_user_by_id(user_id):
    """Get user by ID from DB."""
    conn = _get_user_db_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT id, email, first_name, last_name, company, created_at FROM users WHERE id = %s', (user_id,))
        return cursor.fetchone()
    finally:
        cursor.close()
        conn.close()

def create_user(email, password, first_name, last_name, company=''):
    """Create a new user."""
    conn = _get_user_db_connection()
    try:
        cursor = conn.cursor()
        password_hash = generate_password_hash(password)
        cursor.execute(
            'INSERT INTO users (email, password_hash, first_name, last_name, company) VALUES (%s, %s, %s, %s, %s)',
            (email, password_hash, first_name, last_name, company)
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
            session['user_id'] = user['id']
            session['user_email'] = user['email']
            session['user_name'] = f"{user['first_name']} {user['last_name']}"
            session.permanent = True
            return redirect(url_for('index'))
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
        
        # Validation
        if not all([email, password, first_name, last_name]):
            return jsonify({'message': 'All required fields must be filled'}), 400
        
        # Email validation
        if '@' not in email or '.' not in email:
            return jsonify({'message': 'Please enter a valid email address'}), 400
        
        if len(password) < 8:
            return jsonify({'message': 'Password must be at least 8 characters long'}), 400
        
        if not create_user(email, password, first_name, last_name, company):
            return jsonify({'message': 'Email already registered'}), 409
        
        return jsonify({'message': 'Registration successful'}), 201
    
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    return render_template('register_page.html')

@auth_bp.route('/logout')
def logout():
    """Logout route"""
    session.clear()
    return redirect(url_for('auth.login'))

@auth_bp.route('/api/auth/status')
def auth_status():
    """Check authentication status"""
    if 'user_id' in session:
        return jsonify({
            'authenticated': True,
            'user_id': session.get('user_id'),
            'user_email': session.get('user_email'),
            'user_name': session.get('user_name')
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
            'created_at': str(user['created_at'])
        })
    return jsonify({'message': 'User not found'}), 404
