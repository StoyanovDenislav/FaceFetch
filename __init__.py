"""
PostgreSQL Database Module for FaceFetch
Handles storage and retrieval of recognized faces and detection photos
"""
import psycopg2
from psycopg2 import Error
import os
from datetime import datetime
import base64
import io
from typing import Optional, List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages MySQL database connections and operations"""
    
    def __init__(self, host='localhost', user='postgres', password='', database='facefetch', port=5432):
        """
        Initialize database connection
        
        Args:
            host: PostgreSQL host
            user: PostgreSQL user
            password: PostgreSQL password
            database: Database name
            port: PostgreSQL port (default 5432)
        """
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.connection = None
        self.connect()
    
    def connect(self):
        """Establish PostgreSQL connection"""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port
            )
            self.connection.autocommit = True
            logger.info(f"Connected to PostgreSQL database: {self.database}")
        except Error as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Database connection closed")
    
    def execute_query(self, query, params=None):
        """Execute a query and return results"""
        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Get column names for dictionary-like results
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            cursor.close()
            return results
        except Error as e:
            logger.error(f"Query execution failed: {e}")
            return None
    
    def execute_update(self, query, params=None):
        """Execute an insert/update query"""
        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            self.connection.commit()
            cursor.close()
            return True
        except Error as e:
            logger.error(f"Update failed: {e}")
            self.connection.rollback()
            return False
    
    def get_last_insert_id(self):
        """Get the last inserted row ID (for PostgreSQL, use RETURNING clause instead)"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT LASTVAL() as id")
            result = cursor.fetchone()
            cursor.close()
            return result[0] if result else None
        except Error as e:
            logger.error(f"Failed to get last insert ID: {e}")
            return None
    
    # ========== PERSON MANAGEMENT ==========
    
    def add_person(self, name: str, description: str = '', phone: str = '') -> Optional[int]:
        """Add a new person to the database"""
        query = """
        INSERT INTO persons (name, description, phone, created_at)
        VALUES (%s, %s, %s, NOW())
        """
        if self.execute_update(query, (name, description, phone)):
            return self.get_last_insert_id()
        return None
    
    def get_person(self, person_id: int) -> Optional[Dict]:
        """Get person details by ID"""
        query = "SELECT * FROM persons WHERE person_id = %s"
        results = self.execute_query(query, (person_id,))
        return results[0] if results else None
    
    def get_person_by_name(self, name: str) -> Optional[Dict]:
        """Get person by name"""
        query = "SELECT * FROM persons WHERE name = %s"
        results = self.execute_query(query, (name,))
        return results[0] if results else None
    
    def get_all_persons(self) -> List[Dict]:
        """Get all persons"""
        query = "SELECT * FROM persons ORDER BY created_at DESC"
        return self.execute_query(query) or []
    
    def update_person(self, person_id: int, name: str = None, description: str = None, phone: str = None) -> bool:
        """Update person information"""
        updates = []
        params = []
        
        if name:
            updates.append("name = %s")
            params.append(name)
        if description is not None:
            updates.append("description = %s")
            params.append(description)
        if phone:
            updates.append("phone = %s")
            params.append(phone)
        
        if not updates:
            return False
        
        params.append(person_id)
        query = f"UPDATE persons SET {', '.join(updates)} WHERE person_id = %s"
        return self.execute_update(query, params)
    
    def delete_person(self, person_id: int) -> bool:
        """Delete a person and all associated photos"""
        # Delete associated photos
        self.execute_update("DELETE FROM detected_faces WHERE person_id = %s", (person_id,))
        # Delete person
        query = "DELETE FROM persons WHERE person_id = %s"
        return self.execute_update(query, (person_id,))
    
    # ========== FACE ENCODING STORAGE ==========
    
    def add_face_encoding(self, person_id: int, encoding: List[float], photo_path: str = '') -> Optional[int]:
        """Store face encoding (for training/recognition)"""
        # Convert encoding list to JSON string
        import json
        encoding_json = json.dumps(encoding)
        
        query = """
        INSERT INTO face_encodings (person_id, encoding, photo_path, created_at)
        VALUES (%s, %s, %s, NOW())
        """
        if self.execute_update(query, (person_id, encoding_json, photo_path)):
            return self.get_last_insert_id()
        return None
    
    def get_face_encodings(self, person_id: int) -> List[Dict]:
        """Get all face encodings for a person"""
        query = "SELECT * FROM face_encodings WHERE person_id = %s ORDER BY created_at DESC"
        return self.execute_query(query, (person_id,)) or []
    
    def delete_face_encoding(self, encoding_id: int) -> bool:
        """Delete a face encoding"""
        query = "DELETE FROM face_encodings WHERE encoding_id = %s"
        return self.execute_update(query, (encoding_id,))
    
    # ========== DETECTION STORAGE ==========
    
    def add_detection(self, person_id: Optional[int], face_id: str, confidence: float, 
                     is_live: bool, photo_blob: bytes, frame_number: int = 0) -> Optional[int]:
        """Store a face detection with photo"""
        query = """
        INSERT INTO detected_faces (person_id, face_id, confidence, is_live, photo_blob, frame_number, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, NOW())
        """
        params = (person_id, face_id, confidence, is_live, photo_blob, frame_number)
        if self.execute_update(query, params):
            return self.get_last_insert_id()
        return None
    
    def get_detections(self, limit: int = 100, offset: int = 0, person_id: Optional[int] = None) -> List[Dict]:
        """Get recent detections"""
        if person_id:
            query = """
            SELECT * FROM detected_faces 
            WHERE person_id = %s
            ORDER BY timestamp DESC
            LIMIT %s OFFSET %s
            """
            return self.execute_query(query, (person_id, limit, offset)) or []
        else:
            query = """
            SELECT * FROM detected_faces 
            ORDER BY timestamp DESC
            LIMIT %s OFFSET %s
            """
            return self.execute_query(query, (limit, offset)) or []
    
    def get_detection_by_id(self, detection_id: int) -> Optional[Dict]:
        """Get detection details including photo"""
        query = "SELECT * FROM detected_faces WHERE detection_id = %s"
        results = self.execute_query(query, (detection_id,))
        return results[0] if results else None
    
    def get_detections_by_face_id(self, face_id: str) -> List[Dict]:
        """Get all detections for a specific face"""
        query = "SELECT * FROM detected_faces WHERE face_id = %s ORDER BY timestamp DESC"
        return self.execute_query(query, (face_id,)) or []
    
    def delete_detection(self, detection_id: int) -> bool:
        """Delete a detection"""
        query = "DELETE FROM detected_faces WHERE detection_id = %s"
        return self.execute_update(query, (detection_id,))
    
    def cleanup_old_detections(self, days: int = 30) -> int:
        """Delete detections older than specified days"""
        query = """
        DELETE FROM detected_faces 
        WHERE timestamp < DATE_SUB(NOW(), INTERVAL %s DAY)
        """
        cursor = self.connection.cursor()
        cursor.execute(query, (days,))
        deleted = cursor.rowcount
        self.connection.commit()
        cursor.close()
        return deleted
    
    # ========== STATISTICS ==========
    
    def get_person_statistics(self, person_id: int) -> Dict:
        """Get detection statistics for a person"""
        query = """
        SELECT 
            COUNT(*) as total_detections,
            SUM(CASE WHEN is_live = 1 THEN 1 ELSE 0 END) as live_detections,
            AVG(confidence) as avg_confidence,
            MAX(confidence) as max_confidence,
            MIN(timestamp) as first_detection,
            MAX(timestamp) as last_detection
        FROM detected_faces
        WHERE person_id = %s
        """
        results = self.execute_query(query, (person_id,))
        return results[0] if results else {}
    
    def get_all_statistics(self) -> Dict:
        """Get overall system statistics"""
        query = """
        SELECT 
            COUNT(DISTINCT p.person_id) as total_persons,
            COUNT(DISTINCT d.detection_id) as total_detections,
            COUNT(DISTINCT d.face_id) as unique_faces,
            SUM(CASE WHEN d.is_live = 1 THEN 1 ELSE 0 END) as live_detections,
            COUNT(DISTINCT DATE(d.timestamp)) as detection_days
        FROM persons p
        LEFT JOIN detected_faces d ON p.person_id = d.person_id
        """
        results = self.execute_query(query)
        return results[0] if results else {}
    
    def get_detections_by_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        """Get detections within a date range"""
        query = """
        SELECT * FROM detected_faces
        WHERE DATE(timestamp) BETWEEN %s AND %s
        ORDER BY timestamp DESC
        """
        return self.execute_query(query, (start_date, end_date)) or []
    
    # ========== UTILITY ==========
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True
        except Error:
            return False


# Convenience functions
def get_db_manager(host='localhost', user='postgres', password='', database='facefetch', port=5432) -> DatabaseManager:
    """Factory function to get database manager"""
    return DatabaseManager(host, user, password, database, port)