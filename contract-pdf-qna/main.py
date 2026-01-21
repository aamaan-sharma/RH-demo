import os

from app import create_app, socketio


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = str(os.getenv("FLASK_DEBUG", "0")).lower() in ("1", "true", "yes")
    socketio.run(
        app,
        host="0.0.0.0",
        port=port,
        debug=debug,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
