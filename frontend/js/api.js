// API Client for Exercise Competition Platform
const API_BASE = 'http://127.0.0.1:8080/api';

async function apiRequest(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    console.log('Making API request to:', url);
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    const config = { ...defaultOptions, ...options };
    
    try {
        const response = await fetch(url, config);
        console.log('API response status:', response.status);
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('API response data:', data);
        return data;
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

// Users API
async function createUser(username) {
    return apiRequest('/users', {
        method: 'POST',
        body: JSON.stringify({ username }),
    });
}

async function getUser(userId) {
    return apiRequest(`/users/${userId}`);
}

async function listUsers(skip = 0, limit = 100) {
    return apiRequest(`/users?skip=${skip}&limit=${limit}`);
}

// Exercises API
async function listExercises() {
    return apiRequest('/exercises');
}

async function getExercise(exerciseId) {
    return apiRequest(`/exercises/${exerciseId}`);
}

async function getLeaderboard(exerciseId, period = 'all_time', limit = 10) {
    return apiRequest(`/exercises/${exerciseId}/leaderboard?period=${period}&limit=${limit}`);
}

// Sessions API
async function uploadExerciseVideo(userId, exerciseId, videoFile) {
    const formData = new FormData();
    formData.append('video', videoFile);
    formData.append('user_id', userId);
    formData.append('exercise_id', exerciseId);
    
    const url = `${API_BASE}/sessions/upload`;
    
    try {
        const response = await fetch(url, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        throw error;
    }
}

async function getSession(sessionId) {
    return apiRequest(`/sessions/${sessionId}`);
}

async function getUserSessions(userId, exerciseId = null, skip = 0, limit = 100) {
    let endpoint = `/sessions/user/${userId}?skip=${skip}&limit=${limit}`;
    if (exerciseId) {
        endpoint += `&exercise_id=${exerciseId}`;
    }
    return apiRequest(endpoint);
}

// Challenges API
async function createChallengeAPI(challengerId, opponentId, exerciseId) {
    return apiRequest('/challenges', {
        method: 'POST',
        body: JSON.stringify({
            challenger_id: challengerId,
            opponent_id: opponentId,
            exercise_id: exerciseId,
        }),
    });
}

async function getChallenge(challengeId) {
    return apiRequest(`/challenges/${challengeId}`);
}

async function submitChallengeSession(challengeId, exerciseSessionId) {
    return apiRequest(`/challenges/${challengeId}/submit`, {
        method: 'POST',
        body: JSON.stringify({
            exercise_session_id: exerciseSessionId,
        }),
    });
}

async function acceptChallengeAPI(challengeId) {
    return apiRequest(`/challenges/${challengeId}/accept`, {
        method: 'POST',
    });
}

async function rejectChallengeAPI(challengeId) {
    return apiRequest(`/challenges/${challengeId}/reject`, {
        method: 'POST',
    });
}

async function getUserChallenges(userId, status = null) {
    let endpoint = `/challenges/user/${userId}`;
    if (status) {
        endpoint += `?status=${status}`;
    }
    return apiRequest(endpoint);
}

