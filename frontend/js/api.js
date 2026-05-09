// ── api.js ───────────────────────────────────────────────────
const API_BASE = 'http://43.204.163.235:8080/api';

async function apiRequest(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const defaultOptions = { headers: { 'Content-Type': 'application/json' } };
  const config = { ...defaultOptions, ...options };
  try {
    const response = await fetch(url, config);
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('API request failed:', error);
    throw error;
  }
}

// Users
async function createUser(username) {
  return apiRequest('/users', { method: 'POST', body: JSON.stringify({ username }) });
}
async function getUser(userId)              { return apiRequest(`/users/${userId}`); }
async function listUsers(skip=0, limit=100) { return apiRequest(`/users?skip=${skip}&limit=${limit}`); }

// Exercises
async function listExercises()              { return apiRequest('/exercises'); }
async function getExercise(exerciseId)      { return apiRequest(`/exercises/${exerciseId}`); }
async function getLeaderboard(exerciseId, period='all_time', limit=10) {
  return apiRequest(`/exercises/${exerciseId}/leaderboard?period=${period}&limit=${limit}`);
}

// Sessions
async function uploadExerciseVideo(userId, exerciseId, videoFile) {
  const formData = new FormData();
  formData.append('video', videoFile);
  formData.append('user_id', userId);
  formData.append('exercise_id', exerciseId);
  const response = await fetch(`${API_BASE}/sessions/upload`, { method: 'POST', body: formData });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || `HTTP error! status: ${response.status}`);
  }
  return await response.json();
}
async function getSession(sessionId) { return apiRequest(`/sessions/${sessionId}`); }
async function getUserSessions(userId, exerciseId=null, skip=0, limit=100) {
  let endpoint = `/sessions/user/${userId}?skip=${skip}&limit=${limit}`;
  if (exerciseId) endpoint += `&exercise_id=${exerciseId}`;
  return apiRequest(endpoint);
}

// Challenges
async function createChallengeAPI(challengerId, opponentId, exerciseId) {
  return apiRequest('/challenges', {
    method: 'POST',
    body: JSON.stringify({ challenger_id: challengerId, opponent_id: opponentId, exercise_id: exerciseId }),
  });
}
async function getChallenge(challengeId)    { return apiRequest(`/challenges/${challengeId}`); }
async function submitChallengeSession(challengeId, exerciseSessionId) {
  return apiRequest(`/challenges/${challengeId}/submit`, {
    method: 'POST',
    body: JSON.stringify({ exercise_session_id: exerciseSessionId }),
  });
}
async function acceptChallengeAPI(challengeId) {
  return apiRequest(`/challenges/${challengeId}/accept`, { method: 'POST' });
}
async function rejectChallengeAPI(challengeId) {
  return apiRequest(`/challenges/${challengeId}/reject`, { method: 'POST' });
}
async function getUserChallenges(userId, status=null) {
  let endpoint = `/challenges/user/${userId}`;
  if (status) endpoint += `?status=${status}`;
  return apiRequest(endpoint);
}
