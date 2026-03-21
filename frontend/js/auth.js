// ── auth.js ─────────────────────────────────────────────────
function checkAuth() {
  const userId   = localStorage.getItem('userId');
  const username = localStorage.getItem('username');
  if (!userId || !username) {
    const path = window.location.pathname;
    if (!path.endsWith('index.html') && !path.endsWith('/')) {
      window.location.href = 'index.html';
    }
  }
}

function logout() {
  localStorage.removeItem('userId');
  localStorage.removeItem('username');
  window.location.href = 'index.html';
}

// Render nav user info once DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  const username = localStorage.getItem('username') || '';
  const initial  = username.charAt(0).toUpperCase();

  const avatarEl = document.getElementById('nav-avatar');
  const nameEl   = document.getElementById('nav-username');

  if (avatarEl) avatarEl.textContent = initial || '?';
  if (nameEl)   nameEl.textContent   = username;

  // Highlight active link
  const links = document.querySelectorAll('.nav-links a');
  links.forEach(a => {
    if (window.location.pathname.endsWith(a.getAttribute('href'))) {
      a.classList.add('active');
    }
  });
});