"""Serves this folder on http://localhost:8000 (the microphone needs localhost or https)."""
import http.server
import socketserver
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm", ".js": "text/javascript"}


with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
    print(f"Open http://localhost:{PORT}  (Ctrl+C to stop)")
    httpd.serve_forever()
