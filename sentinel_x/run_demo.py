#!/usr/bin/env python3
"""
Run the Sentinel-X Demo UI.

This script starts the FastAPI backend server which provides:
- REST API for demo control and patient data
- WebSocket for real-time updates
- Frontend serving (when built)

Usage:
    python run_demo.py [--port PORT] [--host HOST]

Frontend:
    cd frontend && npm install && npm run dev
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Run the Sentinel-X Demo UI server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    import uvicorn
    from api.main import app

    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║                    Sentinel-X Demo UI                      ║
    ╠═══════════════════════════════════════════════════════════╣
    ║  API Server:  http://{args.host}:{args.port}                        ║
    ║  API Docs:    http://{args.host}:{args.port}/docs                   ║
    ║                                                           ║
    ║  Frontend:    cd frontend && npm install && npm run dev   ║
    ║               Then open http://localhost:5173             ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
