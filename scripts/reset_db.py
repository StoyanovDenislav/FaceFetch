import pymysql
import os
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.environ.get('DB_USER', 'root')
DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = int(os.environ.get('DB_PORT', 3306))
SQL_FILE = 'face_db.sql'

def reset_database():
    print(f"WARNING: This will DELETE all data in the 'facefetch' database and recreate it from {SQL_FILE}.")
    print("Connecting to MySQL server...")
    
    try:
        # Connect without selecting a database first
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True
        )
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    try:
        with conn.cursor() as cursor:
            # Read SQL file
            print(f"Reading {SQL_FILE}...")
            with open(SQL_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            delimiter = ';'
            statement = ''
            
            print("Executing SQL statements...")
            for line in lines:
                # Skip comments and empty lines if they are standalone
                if not statement and (line.strip().startswith('--') or not line.strip()):
                    continue
                
                # Handle DELIMITER command
                if line.strip().upper().startswith('DELIMITER'):
                    delimiter = line.strip().split()[1]
                    continue
                
                statement += line
                
                # Check if statement ends with the current delimiter
                if statement.strip().endswith(delimiter):
                    # Remove delimiter from the end
                    sql = statement.strip()
                    if delimiter != ';':
                        sql = sql[:-len(delimiter)]
                    else:
                         sql = sql[:-1] # Remove semi-colon
                    
                    # Execute
                    if sql.strip():
                        try:
                            # print(f"Executing: {sql[:50]}...")
                            cursor.execute(sql)
                        except Exception as e:
                            print(f"Error executing statement:\n{sql[:100]}...\nError: {e}")
                            # Stop on error? For now, yes.
                            return
                    
                    statement = ''
                    
            print("✅ Database successfully reset and reinstantiated from schema.")

    except Exception as e:
        print(f"❌ Error during reset: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    confirm = input("Are you sure you want to wipe the database? (yes/no): ")
    if confirm.lower() == 'yes':
        reset_database()
    else:
        print("Operation cancelled.")
