-- Exercise Competition Platform Database Schema

CREATE DATABASE IF NOT EXISTS exercise_competition;
USE exercise_competition;

-- Users Table
CREATE TABLE IF NOT EXISTS users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_username (username)
);

-- Exercises Table (Extensible for future exercises)
CREATE TABLE IF NOT EXISTS exercises (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) UNIQUE NOT NULL,  -- 'bicep_curl', 'plank', etc.
    display_name VARCHAR(100) NOT NULL,  -- 'Bicep Curl', 'Plank'
    description TEXT,
    metric_type ENUM('score', 'duration', 'reps') NOT NULL,  -- How to measure performance
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Exercise Sessions Table (Stores each workout session)
CREATE TABLE IF NOT EXISTS exercise_sessions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    exercise_id INT NOT NULL,
    score INT NOT NULL,  -- Quality score (0-100) or rep count
    duration_seconds FLOAT,  -- Session duration
    error_count INT DEFAULT 0,  -- Total errors detected
    video_filename VARCHAR(255),  -- Stored video path (optional)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (exercise_id) REFERENCES exercises(id) ON DELETE CASCADE,
    INDEX idx_user_exercise (user_id, exercise_id),
    INDEX idx_created_at (created_at),
    INDEX idx_score (score)
);

-- Challenges Table (Head-to-head competitions)
CREATE TABLE IF NOT EXISTS challenges (
    id INT PRIMARY KEY AUTO_INCREMENT,
    challenger_id INT NOT NULL,  -- User who created challenge
    opponent_id INT NOT NULL,  -- User being challenged
    exercise_id INT NOT NULL,
    status ENUM('pending', 'active', 'completed') DEFAULT 'pending',
    winner_id INT NULL,  -- Set when challenge completed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    FOREIGN KEY (challenger_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (opponent_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (exercise_id) REFERENCES exercises(id) ON DELETE CASCADE,
    FOREIGN KEY (winner_id) REFERENCES users(id) ON DELETE SET NULL,
    INDEX idx_status (status),
    INDEX idx_users (challenger_id, opponent_id)
);

-- Challenge Sessions Table (Links sessions to challenges)
CREATE TABLE IF NOT EXISTS challenge_sessions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    challenge_id INT NOT NULL,
    exercise_session_id INT NOT NULL,  -- Session submitted for this challenge
    FOREIGN KEY (challenge_id) REFERENCES challenges(id) ON DELETE CASCADE,
    FOREIGN KEY (exercise_session_id) REFERENCES exercise_sessions(id) ON DELETE CASCADE,
    UNIQUE KEY unique_challenge_session (challenge_id, exercise_session_id)
);

-- Insert initial exercises
INSERT INTO exercises (name, display_name, description, metric_type) VALUES
('bicep_curl', 'Bicep Curl', 'Bicep curl exercise with form detection', 'score'),
('plank', 'Plank', 'Plank exercise with posture detection', 'score')
ON DUPLICATE KEY UPDATE display_name=display_name;

