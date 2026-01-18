-- FaceFetch Database Schema
-- MySQL database schema for facial recognition system
-- Created: 2026-01-16
 
-- ============================================
-- DATABASE CREATION
-- ============================================
DROP DATABASE IF EXISTS facefetch;
CREATE DATABASE facefetch;
USE facefetch;
 
-- ============================================
-- USERS TABLE (Authentication)
-- ============================================
CREATE TABLE users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  email VARCHAR(255) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  first_name VARCHAR(100) NOT NULL,
  last_name VARCHAR(100) NOT NULL,
  company VARCHAR(255),
  role ENUM('operator', 'viewer') NOT NULL DEFAULT 'viewer',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
 
-- ============================================
-- KNOWN FACES TABLE (Biometrics)
-- ============================================
CREATE TABLE known_faces (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(120) NOT NULL,
  active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE INDEX idx_known_faces_name ON known_faces(name);
CREATE INDEX idx_known_faces_active ON known_faces(active);

-- ============================================
-- REFRESH TOKENS TABLE
-- ============================================
CREATE TABLE refresh_tokens (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  token_hash VARCHAR(255) NOT NULL,
  expires_at TIMESTAMP NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_refresh_tokens_users FOREIGN KEY (user_id)
    REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE INDEX idx_refresh_token_hash ON refresh_tokens(token_hash);

 
-- ============================================
-- FACE PROFILES TABLE
-- ============================================
CREATE TABLE face_profiles (
  id INT AUTO_INCREMENT PRIMARY KEY,
  known_face_id INT NOT NULL,
  label VARCHAR(120),
  image_path VARCHAR(255),
  face_encoding LONGBLOB NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT fk_face_profiles_known_faces FOREIGN KEY (known_face_id)
    REFERENCES known_faces(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
 
CREATE INDEX ix_face_profiles_known_face_id ON face_profiles(known_face_id);
CREATE INDEX ix_face_profiles_label ON face_profiles(label);
 
-- ============================================
-- SESSIONS TABLE
-- ============================================
CREATE TABLE sessions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(120) NOT NULL,
  started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  ended_at TIMESTAMP NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
 
CREATE INDEX ix_sessions_started_at ON sessions(started_at);
CREATE INDEX ix_sessions_ended_at ON sessions(ended_at);
 
-- ============================================
-- DETECTION EVENTS TABLE
-- ============================================
CREATE TABLE detection_events (
  id INT AUTO_INCREMENT PRIMARY KEY,
  session_id INT NOT NULL,
  known_face_id INT,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  result VARCHAR(20) NOT NULL,
  confidence DECIMAL(5,4) CHECK (confidence >= 0 AND confidence <= 1),
  face_location_top INT,
  face_location_right INT,
  face_location_bottom INT,
  face_location_left INT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_detection_events_sessions FOREIGN KEY (session_id)
    REFERENCES sessions(id) ON DELETE CASCADE,
  CONSTRAINT fk_detection_events_known_faces FOREIGN KEY (known_face_id)
    REFERENCES known_faces(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
 
CREATE INDEX ix_detection_events_session_id ON detection_events(session_id);
CREATE INDEX ix_detection_events_known_face_id ON detection_events(known_face_id);
CREATE INDEX ix_detection_events_timestamp ON detection_events(timestamp);
CREATE INDEX ix_detection_events_result ON detection_events(result);
 
-- ============================================
-- COMMANDS TABLE
-- ============================================
CREATE TABLE commands (
  id INT AUTO_INCREMENT PRIMARY KEY,
  session_id INT NOT NULL,
  issued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  command VARCHAR(255) NOT NULL,
  result VARCHAR(255),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_commands_sessions FOREIGN KEY (session_id)
    REFERENCES sessions(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
 
CREATE INDEX ix_commands_session_id ON commands(session_id);
CREATE INDEX ix_commands_issued_at ON commands(issued_at);
 
-- ============================================
-- ALERTS TABLE
-- ============================================
CREATE TABLE alerts (
  id INT AUTO_INCREMENT PRIMARY KEY,
  session_id INT,
  detection_event_id INT,
  alert_type VARCHAR(50) NOT NULL,
  message TEXT,
  severity VARCHAR(20),
  acknowledged BOOLEAN DEFAULT FALSE,
  acknowledged_at TIMESTAMP NULL,
  acknowledged_by VARCHAR(120),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT fk_alerts_sessions FOREIGN KEY (session_id)
    REFERENCES sessions(id) ON DELETE SET NULL,
  CONSTRAINT fk_alerts_detection_events FOREIGN KEY (detection_event_id)
    REFERENCES detection_events(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
 
CREATE INDEX ix_alerts_alert_type ON alerts(alert_type);
CREATE INDEX ix_alerts_severity ON alerts(severity);
CREATE INDEX ix_alerts_acknowledged ON alerts(acknowledged);
CREATE INDEX ix_alerts_created_at ON alerts(created_at);
 
-- ============================================
-- STATISTICS TABLE
-- ============================================
CREATE TABLE statistics (
  id INT AUTO_INCREMENT PRIMARY KEY,
  date DATE NOT NULL UNIQUE,
  total_detections INT DEFAULT 0,
  recognized_faces INT DEFAULT 0,
  unknown_faces INT DEFAULT 0,
  spoof_attempts INT DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
 
CREATE INDEX ix_statistics_date ON statistics(date);
 
-- ============================================
-- COMPOSITE INDEXES FOR PERFORMANCE
-- ============================================
-- ============================================
-- COMPOSITE INDEXES FOR PERFORMANCE
-- ============================================
CREATE INDEX ix_detection_known_face_time ON detection_events(known_face_id, timestamp);
CREATE INDEX ix_detection_result_time ON detection_events(result, timestamp);

-- ============================================
-- VIEWS
-- ============================================

CREATE OR REPLACE VIEW recent_detections AS
SELECT
  de.id,
  s.id as session_id,
  kf.name as face_name,
  de.result,
  de.confidence,
  de.timestamp,
  CONCAT(de.face_location_left, ',', de.face_location_top, ',', de.face_location_right, ',', de.face_location_bottom) as face_location
FROM detection_events de
LEFT JOIN sessions s ON de.session_id = s.id
LEFT JOIN known_faces kf ON de.known_face_id = kf.id
ORDER BY de.timestamp DESC
LIMIT 100;

CREATE OR REPLACE VIEW today_statistics AS
SELECT
  DATE(de.timestamp) as date,
  COUNT(*) as total_detections,
  SUM(CASE WHEN de.result = 'recognized' THEN 1 ELSE 0 END) as recognized_faces,
  SUM(CASE WHEN de.result = 'unknown' THEN 1 ELSE 0 END) as unknown_faces,
  SUM(CASE WHEN de.result = 'spoof' THEN 1 ELSE 0 END) as spoof_attempts
FROM detection_events de
WHERE DATE(de.timestamp) = CURDATE()
GROUP BY DATE(de.timestamp);

-- ============================================
-- STORED PROCEDURES / FUNCTIONS
-- ============================================

DELIMITER $$

CREATE PROCEDURE get_face_detection_summary(IN p_known_face_id INT)
BEGIN
  SELECT
    kf.id as known_face_id,
    kf.name as face_name,
    COUNT(de.id) as total_detections,
    SUM(CASE WHEN de.result = 'recognized' THEN 1 ELSE 0 END) as recognized_count,
    SUM(CASE WHEN de.result = 'unknown' THEN 1 ELSE 0 END) as unknown_count,
    SUM(CASE WHEN de.result = 'spoof' THEN 1 ELSE 0 END) as spoof_count,
    MAX(de.timestamp) as last_detection,
    AVG(de.confidence) as avg_confidence
  FROM known_faces kf
  LEFT JOIN detection_events de ON kf.id = de.known_face_id
  WHERE kf.id = p_known_face_id
  GROUP BY kf.id, kf.name;
END$$
 
CREATE PROCEDURE archive_old_detections(IN days_to_keep INT)
BEGIN
  DELETE FROM detection_events
  WHERE timestamp < DATE_SUB(NOW(), INTERVAL days_to_keep DAY);
 
  SELECT ROW_COUNT() as deleted_count;
END$$
 
DELIMITER ;
 
-- ============================================
-- SAMPLE DATA (Optional - for testing)
-- ============================================
-- INSERT INTO users (name, active) VALUES
-- ('John Doe', TRUE),
-- ('Jane Smith', TRUE),
-- ('Bob Johnson', FALSE);
 
-- ============================================
-- SCHEMA SUMMARY
-- ============================================
-- Tables:
--   - users: Known users/faces (1 record per person)
--   - face_profiles: Face encodings (multiple per user)
--   - sessions: Detection sessions
--   - detection_events: Face detection results
--   - commands: Issued commands
--   - alerts: Security alerts
--   - statistics: Daily aggregated stats
--
-- Views:
--   - recent_detections: Last 100 detections
--   - today_statistics: Today's stats
--
-- Procedures:
--   - get_user_detection_summary: Get user statistics (CALL get_user_detection_summary(1);)
--   - archive_old_detections: Clean old data (CALL archive_old_detections(90);)
--
-- ============================================
 