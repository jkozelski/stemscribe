// StemScriber Service Worker — PWA support
const CACHE_NAME = 'stemscriber-v20';

// Static assets to cache for offline shell
const SHELL_ASSETS = [
  '/',
  '/index.html',
  '/css/variables.css',
  '/css/layout.css',
  '/css/upload.css',
  '/css/processing.css',
  '/css/results.css',
  '/css/settings.css',
  '/css/responsive.css',
  '/css/features.css',
  '/css/karaoke.css',
  '/js/config.js',
  '/js/utils.js',
  '/js/mixer.js',
  '/js/progress.js',
  '/js/results.js',
  '/js/trackinfo.js',
  '/js/archive.js',
  '/js/upload.js',
  '/js/panels.js',
  '/js/waveform.js',
  '/js/speed.js',
  '/js/karaoke.js',
  '/js/theme.js',
  '/js/app.js',
];

// Install: cache shell assets
self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open(CACHE_NAME).then(function(cache) {
      return cache.addAll(SHELL_ASSETS);
    })
  );
  self.skipWaiting();
});

// Activate: clean old caches
self.addEventListener('activate', function(event) {
  event.waitUntil(
    caches.keys().then(function(keys) {
      return Promise.all(
        keys.filter(function(k) { return k !== CACHE_NAME; })
            .map(function(k) { return caches.delete(k); })
      );
    })
  );
  self.clients.claim();
});

// Fetch: network-first for API, cache-first for static assets
self.addEventListener('fetch', function(event) {
  var url = new URL(event.request.url);

  // API calls and audio: always go to network
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/uploads/') || url.pathname.startsWith('/outputs/')) {
    return;
  }

  // HTML pages: network-first so updates are always picked up immediately
  if (url.pathname.endsWith('.html') || url.pathname === '/' || url.pathname === '/app') {
    event.respondWith(
      fetch(event.request).catch(function() {
        return caches.match(event.request);
      })
    );
    return;
  }

  // Static assets: cache-first
  event.respondWith(
    caches.match(event.request).then(function(cached) {
      if (cached) return cached;
      return fetch(event.request).then(function(response) {
        // Cache successful responses
        if (response.ok) {
          var clone = response.clone();
          caches.open(CACHE_NAME).then(function(cache) {
            cache.put(event.request, clone);
          });
        }
        return response;
      });
    })
  );
});
