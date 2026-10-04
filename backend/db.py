import os
import sqlite3
import hashlib
import json
import secrets
import pymysql

# MySQL Configuration Defaults
MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
MYSQL_DB = os.environ.get("MYSQL_DB", "lung_cancer_db")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", 3306))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB_PATH = os.path.join(BASE_DIR, "users_fallback.db")

def configure_mysql_password(password: str):
    """Dynamically set MySQL root password and test connection."""
    global MYSQL_PASSWORD
    MYSQL_PASSWORD = password
    db_status = init_db()
    if db_status and db_status.get("engine") == "MySQL":
        return {
            "success": True,
            "message": "Connected to local MySQL database (localhost:3306 / lung_cancer_db) successfully!",
            "status": db_status
        }
    else:
        return {
            "success": False,
            "error": "Failed to connect to MySQL with provided password. Please check password or verify MySQL-8 is active on localhost:3306.",
            "status": db_status
        }

def hash_password(password: str) -> str:
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def get_mysql_connection():
    """Attempt to connect to MySQL server and ensure database exists."""
    try:
        conn = pymysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            port=MYSQL_PORT,
            autocommit=True,
            connect_timeout=3
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DB}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        cursor.close()
        conn.close()

        conn = pymysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            port=MYSQL_PORT,
            autocommit=True,
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=3
        )
        return conn
    except Exception as e:
        print(f"[Database Warning] MySQL connection failed ({e}). Falling back to SQLite/Memory.")
        return None

def init_db():
    """Initialize MySQL tables, or fallback SQLite tables if MySQL is offline."""
    mysql_conn = get_mysql_connection()
    if mysql_conn:
        try:
            with mysql_conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        username VARCHAR(50) UNIQUE NOT NULL,
                        email VARCHAR(100) UNIQUE NOT NULL,
                        password VARCHAR(255) NOT NULL,
                        full_name VARCHAR(100) NOT NULL,
                        role VARCHAR(50) DEFAULT 'Doctor',
                        hospital_name VARCHAR(100) DEFAULT 'General Hospital',
                        auth_token VARCHAR(255) DEFAULT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Check if auth_token column exists (migration)
                try:
                    cursor.execute("SHOW COLUMNS FROM users LIKE 'auth_token'")
                    if not cursor.fetchone():
                        cursor.execute("ALTER TABLE users ADD COLUMN auth_token VARCHAR(255) DEFAULT NULL")
                except Exception:
                    pass

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS prediction_history (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        case_id VARCHAR(50) NOT NULL,
                        username VARCHAR(50) DEFAULT 'doctor',
                        predicted_class VARCHAR(50) NOT NULL,
                        confidence FLOAT NOT NULL,
                        probabilities_json TEXT NOT NULL,
                        model_name VARCHAR(50) DEFAULT 'ConvNeXt-Tiny',
                        model_version VARCHAR(20) DEFAULT 'v1.0',
                        gradcam_focus FLOAT DEFAULT 0.75,
                        verification_status VARCHAR(20) DEFAULT 'Pending',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS training_runs (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        run_id VARCHAR(50) NOT NULL,
                        model_name VARCHAR(50) NOT NULL,
                        dataset_name VARCHAR(100) NOT NULL,
                        epochs INT DEFAULT 20,
                        batch_size INT DEFAULT 8,
                        learning_rate FLOAT DEFAULT 0.0003,
                        metrics_json TEXT,
                        status VARCHAR(20) DEFAULT 'Completed',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Ensure default doctor account matches user database
                cursor.execute("SELECT id FROM users WHERE username = %s", ('doctor',))
                if not cursor.fetchone():
                    default_pwd = hash_password('password123')
                    cursor.execute("""
                        INSERT INTO users (username, email, password, full_name, role, hospital_name)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, ('doctor', 'nivethaselvakumar23@gmail.com', default_pwd, 'Nivetha Selvakumar', 'Senior Pulmonologist', 'General Hospital'))
                else:
                    # Update profile fields to match user screenshot
                    cursor.execute("""
                        UPDATE users SET email = %s, full_name = %s WHERE username = %s
                    """, ('nivethaselvakumar23@gmail.com', 'Nivetha Selvakumar', 'doctor'))

            mysql_conn.close()
            print("[Database] MySQL database initialized successfully!")
            return {"engine": "MySQL", "host": MYSQL_HOST, "database": MYSQL_DB, "status": "Connected"}
        except Exception as e:
            print(f"[Database Error] MySQL init table error: {e}")

    # SQLite Fallback
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT DEFAULT 'Doctor',
                hospital_name TEXT DEFAULT 'General Hospital',
                auth_token TEXT DEFAULT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add auth_token column if missing
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN auth_token TEXT DEFAULT NULL")
        except Exception:
            pass

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                username TEXT DEFAULT 'doctor',
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                probabilities_json TEXT NOT NULL,
                model_name TEXT DEFAULT 'ConvNeXt-Tiny',
                model_version TEXT DEFAULT 'v1.0',
                gradcam_focus REAL DEFAULT 0.75,
                verification_status TEXT DEFAULT 'Pending',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                epochs INTEGER DEFAULT 20,
                batch_size INTEGER DEFAULT 8,
                learning_rate REAL DEFAULT 0.0003,
                metrics_json TEXT,
                status TEXT DEFAULT 'Completed',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("SELECT id FROM users WHERE username = 'doctor'")
        if not cursor.fetchone():
            default_pwd = hash_password('password123')
            cursor.execute("""
                INSERT INTO users (username, email, password, full_name, role, hospital_name)
                VALUES ('doctor', 'nivethaselvakumar23@gmail.com', ?, 'Nivetha Selvakumar', 'Senior Pulmonologist', 'General Hospital')
            """, (default_pwd,))
        else:
            cursor.execute("""
                UPDATE users SET email = 'nivethaselvakumar23@gmail.com', full_name = 'Nivetha Selvakumar' WHERE username = 'doctor'
            """)

        conn.commit()
        conn.close()
        print("[Database] SQLite fallback database initialized!")
        return {"engine": "SQLite (Fallback)", "database": SQLITE_DB_PATH, "status": "Active (MySQL Offline)"}
    except Exception as e:
        print(f"[Database Error] SQLite fallback error: {e}")
        return {"engine": "Memory", "status": "Active"}

def register_user(username, email, password, full_name, role="Doctor", hospital_name="General Hospital"):
    """Register a new user in MySQL (or SQLite fallback)."""
    hashed_pwd = hash_password(password)
    mysql_conn = get_mysql_connection()
    
    if mysql_conn:
        try:
            with mysql_conn.cursor() as cursor:
                cursor.execute("SELECT id FROM users WHERE username = %s OR email = %s", (username, email))
                if cursor.fetchone():
                    mysql_conn.close()
                    return {"success": False, "error": "Username or email already exists."}
                
                cursor.execute("""
                    INSERT INTO users (username, email, password, full_name, role, hospital_name)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (username, email, hashed_pwd, full_name, role, hospital_name))
                user_id = cursor.lastrowid
            mysql_conn.close()
            return {
                "success": True,
                "user": {
                    "id": user_id,
                    "username": username,
                    "email": email,
                    "full_name": full_name,
                    "role": role,
                    "hospital_name": hospital_name
                },
                "db_engine": "MySQL"
            }
        except Exception as e:
            mysql_conn.close()
            return {"success": False, "error": f"MySQL registration error: {str(e)}"}

    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
        if cursor.fetchone():
            conn.close()
            return {"success": False, "error": "Username or email already exists."}
        
        cursor.execute("""
            INSERT INTO users (username, email, password, full_name, role, hospital_name)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (username, email, hashed_pwd, full_name, role, hospital_name))
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return {
            "success": True,
            "user": {
                "id": user_id,
                "username": username,
                "email": email,
                "full_name": full_name,
                "role": role,
                "hospital_name": hospital_name
            },
            "db_engine": "SQLite (Fallback)"
        }
    except Exception as e:
        return {"success": False, "error": f"Registration error: {str(e)}"}

def authenticate_user(username, password):
    """Authenticate user credentials against MySQL (or SQLite fallback) and generate auth token."""
    hashed_pwd = hash_password(password)
    token = f"TOKEN-{secrets.token_hex(24)}"
    mysql_conn = get_mysql_connection()

    if mysql_conn:
        try:
            with mysql_conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, username, email, password, full_name, role, hospital_name
                    FROM users WHERE username = %s OR email = %s
                """, (username, username))
                row = cursor.fetchone()
                if not row:
                    mysql_conn.close()
                    return {"success": False, "error": "User account not found."}
                
                if row['password'] != hashed_pwd:
                    mysql_conn.close()
                    return {"success": False, "error": "Invalid password."}

                cursor.execute("UPDATE users SET auth_token = %s WHERE id = %s", (token, row['id']))
            mysql_conn.close()

            return {
                "success": True,
                "token": token,
                "user": {
                    "id": row['id'],
                    "username": row['username'],
                    "email": row['email'],
                    "full_name": row['full_name'],
                    "role": row['role'],
                    "hospital_name": row['hospital_name'],
                    "auth_token": token
                },
                "db_engine": "MySQL"
            }
        except Exception as e:
            if mysql_conn:
                mysql_conn.close()
            print(f"[Database Error] MySQL Auth error: {e}")

    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, username, email, password, full_name, role, hospital_name
            FROM users WHERE username = ? OR email = ?
        """, (username, username))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return {"success": False, "error": "User account not found."}

        if row['password'] != hashed_pwd:
            conn.close()
            return {"success": False, "error": "Invalid password."}

        cursor.execute("UPDATE users SET auth_token = ? WHERE id = ?", (token, row['id']))
        conn.commit()
        conn.close()

        return {
            "success": True,
            "token": token,
            "user": {
                "id": row['id'],
                "username": row['username'],
                "email": row['email'],
                "full_name": row['full_name'],
                "role": row['role'],
                "hospital_name": row['hospital_name'],
                "auth_token": token
            },
            "db_engine": "SQLite (Fallback)"
        }
    except Exception as e:
        return {"success": False, "error": f"Authentication failed: {str(e)}"}

def get_user_by_token(token):
    """Retrieve user record from database by auth token."""
    if not token:
        return None
    mysql_conn = get_mysql_connection()
    if mysql_conn:
        try:
            with mysql_conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, username, email, full_name, role, hospital_name, auth_token
                    FROM users WHERE auth_token = %s
                """, (token,))
                row = cursor.fetchone()
            mysql_conn.close()
            if row:
                return dict(row)
        except Exception as e:
            if mysql_conn:
                mysql_conn.close()

    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, username, email, full_name, role, hospital_name, auth_token
            FROM users WHERE auth_token = ?
        """, (token,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return dict(row)
    except Exception as e:
        pass
    return None

def save_prediction(case_id, username, predicted_class, confidence, probabilities, model_name="ConvNeXt-Tiny", model_version="v1.0", gradcam_focus=0.75):
    """Save prediction event into database."""
    probs_str = json.dumps(probabilities)
    mysql_conn = get_mysql_connection()
    if mysql_conn:
        try:
            with mysql_conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO prediction_history (case_id, username, predicted_class, confidence, probabilities_json, model_name, model_version, gradcam_focus)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (case_id, username, predicted_class, confidence, probs_str, model_name, model_version, gradcam_focus))
            mysql_conn.close()
            return True
        except Exception as e:
            print(f"[Database Error] Failed to save prediction in MySQL: {e}")

    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO prediction_history (case_id, username, predicted_class, confidence, probabilities_json, model_name, model_version, gradcam_focus)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (case_id, username, predicted_class, confidence, probs_str, model_name, model_version, gradcam_focus))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[Database Error] Failed to save prediction in SQLite: {e}")
        return False

def get_prediction_history(limit=50):
    """Retrieve history of predictions."""
    mysql_conn = get_mysql_connection()
    if mysql_conn:
        try:
            with mysql_conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, case_id, username, predicted_class, confidence, probabilities_json, model_name, model_version, gradcam_focus, verification_status, created_at
                    FROM prediction_history ORDER BY id DESC LIMIT %s
                """, (limit,))
                rows = cursor.fetchall()
            mysql_conn.close()
            for r in rows:
                r['probabilities'] = json.loads(r['probabilities_json'])
                if isinstance(r['created_at'], str):
                    r['date_time'] = r['created_at']
                else:
                    r['date_time'] = r['created_at'].strftime("%Y-%m-%d %H:%M:%S")
            return rows
        except Exception as e:
            print(f"[Database Error] MySQL get history error: {e}")

    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, case_id, username, predicted_class, confidence, probabilities_json, model_name, model_version, gradcam_focus, verification_status, created_at
            FROM prediction_history ORDER BY id DESC LIMIT ?
        """, (limit,))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        for r in rows:
            r['probabilities'] = json.loads(r['probabilities_json'])
            r['date_time'] = str(r['created_at'])
        return rows
    except Exception as e:
        print(f"[Database Error] SQLite get history error: {e}")
        return []

def update_verification_status(record_id, new_status):
    """Update verification status (Pending, Reviewed, Verified, Rejected)."""
    mysql_conn = get_mysql_connection()
    if mysql_conn:
        try:
            with mysql_conn.cursor() as cursor:
                cursor.execute("UPDATE prediction_history SET verification_status = %s WHERE id = %s", (new_status, record_id))
            mysql_conn.close()
            return True
        except Exception as e:
            print(f"[Database Error] MySQL status update failed: {e}")

    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE prediction_history SET verification_status = ? WHERE id = ?", (new_status, record_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[Database Error] SQLite status update failed: {e}")
        return False

def get_db_status():
    """Return database connection status and engine details."""
    mysql_conn = get_mysql_connection()
    if mysql_conn:
        mysql_conn.close()
        return {
            "engine": "MySQL",
            "connected": True,
            "host": MYSQL_HOST,
            "database": MYSQL_DB,
            "port": MYSQL_PORT,
            "message": "Connected to MySQL database server"
        }
    else:
        return {
            "engine": "SQLite (Fallback)",
            "connected": True,
            "database": SQLITE_DB_PATH,
            "message": "MySQL server not running locally; using SQLite fallback store."
        }
