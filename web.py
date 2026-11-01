"""Run the local voice dashboard: python web.py [--port 8765]."""

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError:
        parser.exit(
            1,
            "Install the web dependencies: python -m pip install -r requirements-web.txt\n",
        )
    uvicorn.run("stts.web:app", host="127.0.0.1", port=args.port, workers=1)
