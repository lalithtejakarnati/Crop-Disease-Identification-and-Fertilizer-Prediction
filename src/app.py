"""
Launcher for the Crop AI Assistant Flask application.
"""

from src.app.app import app


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True,
    )
