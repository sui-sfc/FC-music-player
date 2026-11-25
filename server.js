// 依存なしの簡易静的サーバー:  node server.js  → http://localhost:8080
//  - 大きなファイル (rmvpe.onnx 362MB) はストリーミングで返す
//  - マルチスレッド WASM (onnxruntime-web) に必要な cross-origin isolation ヘッダーを付ける
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 8080;
const ROOT = __dirname;
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.md': 'text/plain; charset=utf-8',
  '.wasm': 'application/wasm', '.onnx': 'application/octet-stream',
};

http.createServer((req, res) => {
  let p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  if (p.endsWith('/')) p += 'index.html';
  const file = path.join(ROOT, path.normalize(p));
  if (!file.startsWith(ROOT)) { res.writeHead(403); return res.end(); }
  fs.stat(file, (err, st) => {
    if (err || !st.isFile()) { res.writeHead(404); return res.end('not found'); }
    res.writeHead(200, {
      'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream',
      'Content-Length': st.size,
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless',
      'Cache-Control': file.endsWith('.onnx') || file.endsWith('.wasm') ? 'max-age=86400' : 'no-cache',
    });
    fs.createReadStream(file).pipe(res);
  });
}).listen(PORT, () => console.log(`FC Music Player: http://localhost:${PORT}`));
