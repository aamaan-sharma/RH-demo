"""Application entry point using Flask application factory."""
from app import create_app
from flask_socketio import SocketIO
from app.config.settings import settings

app = create_app()
socketio = app.config['socketio']

if __name__ == "__main__":
    port = settings.PORT
    debug = settings.FLASK_DEBUG
    
    print(f"Starting Flask application on port {port} (debug={debug})")
    socketio.run(app, host="0.0.0.0", port=port, debug=debug, allow_unsafe_werkzeug=True)
