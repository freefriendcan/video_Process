import time

from flask import Flask, Response
from flask_cors import CORS

from config import PipelineConfig


class MjpegServer:
    """Flask-based MJPEG streaming server."""

    def __init__(self, cfg: PipelineConfig):
        self._cfg = cfg
        self._latest_frame = None
        self.app = Flask(__name__)
        CORS(self.app)
        self._register_routes()

    def _register_routes(self):
        @self.app.route('/')
        def index():
            return '''
            <html>
              <head><title>Mac Camera Stream</title></head>
              <body style="background-color:#121212; color:white; text-align:center; font-family:sans-serif;">
                <h2>Live Camera Feed (Face, Gesture & Fall Detection)</h2>
                <img src="/video_feed" style="border:2px solid #333; border-radius:10px; box-shadow: 0px 0px 20px #000;" />
              </body>
            </html>
            '''

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self._generate_frames(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

    def update_frame(self, jpeg_bytes):
        self._latest_frame = jpeg_bytes

    def _generate_frames(self):
        while True:
            if self._latest_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + self._latest_frame + b'\r\n')
            time.sleep(0.03)

    def run(self):
        print(f"Multi-Tracking Vision (Tasks API) Active: http://{self._cfg.server_host}:{self._cfg.server_port}/")
        self.app.run(host=self._cfg.server_host, port=self._cfg.server_port)
