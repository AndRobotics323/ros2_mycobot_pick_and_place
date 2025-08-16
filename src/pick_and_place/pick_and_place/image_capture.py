import cv2
import threading
import time
from flask import Flask, Response


device_num = 0
# device_num = '/dev/jetcocam0'


class CameraManager:
    def __init__(self, device=device_num, enable_streaming=True, flask_port=5000):
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라를 열 수 없습니다: {device}")

        self.latest_frame = None
        self.encoded_frame = None
        self.running = True
        self.lock = threading.Lock()
        
        # 프레임 캡처 스레드 시작
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()
        
        # Flask 스트리밍 설정 (선택적)
        if enable_streaming:
            self.app = Flask(__name__)
            self._setup_routes()
            self.flask_thread = threading.Thread(
                target=lambda: self.app.run(host='0.0.0.0', port=flask_port, threaded=True, debug=False, use_reloader=False), 
                daemon=True
            )
            self.flask_thread.start()
            print(f"카메라 스트리밍 시작: http://0.0.0.0:{flask_port}/stream")

    def _update_frame(self):
        """카메라로부터 프레임을 계속 읽는 백그라운드 루프"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # JPEG 인코딩 (스트리밍용)
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                with self.lock:
                    self.latest_frame = frame.copy()
                    self.encoded_frame = buffer.tobytes()
            time.sleep(0.03)  # 30 FPS

    def get_frame(self):
        """가장 최근 프레임을 반환 (ROS2에서 사용)"""
        with self.lock:
            if self.latest_frame is None:
                raise RuntimeError("아직 프레임이 준비되지 않았습니다.")
            return self.latest_frame.copy()

    def _setup_routes(self):
        """Flask 라우트 설정"""
        @self.app.route('/stream')
        def video_feed():
            return Response(self._generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/')
        def index():
            return '''
            <html><head><title>Camera Stream</title></head>
            <body style="text-align:center; font-family:Arial;">
            <h1>Robot Camera Stream</h1>
            <img src="/stream" style="max-width:90%; border:2px solid #ddd;">
            </body></html>
            '''

    def _generate_frames(self):
        """Flask 스트리밍용 프레임 생성"""
        while self.running:
            with self.lock:
                if self.encoded_frame is None:
                    time.sleep(0.1)
                    continue
                frame_data = self.encoded_frame
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

    def release(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1)
        self.cap.release()