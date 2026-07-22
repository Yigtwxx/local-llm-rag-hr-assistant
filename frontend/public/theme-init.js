// Applies the stored theme before first paint. Without this the page paints
// light, then React swaps to dark a frame later. Must stay in step with
// `useTheme` in src/App.tsx — same key, same fallback order.
//
// It lives here rather than inline in index.html so the CSP can stay on
// `script-src 'self'`: a hash whitelist would silently stop matching the first
// time this file is edited, and the flash would come back unnoticed.
(function () {
  var stored;
  try {
    stored = localStorage.getItem('novatek-theme');
  } catch (error) {
    stored = null;
  }
  var dark =
    stored === 'dark' ||
    (stored !== 'light' &&
      window.matchMedia('(prefers-color-scheme: dark)').matches);
  if (dark) document.documentElement.classList.add('dark');
})();
