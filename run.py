import threading
import webview
from app import app


def start_flask():
    app.run(port=5000, debug=False, use_reloader=False)


if __name__ == '__main__':
    # Flask startuje w tle
    t = threading.Thread(target=start_flask)
    t.daemon = True
    t.start()

    # Otwiera natywne okno desktopowe
    webview.create_window(
        title='Animal Classifier',
        url='http://localhost:5000',
        width=620,
        height=720,
        resizable=False
    )
    webview.start()
