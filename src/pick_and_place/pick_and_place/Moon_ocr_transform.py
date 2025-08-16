import cv2
import numpy as np
import easyocr
from PIL import ImageFont, ImageDraw, Image
import sys

# === 1. 카메라 내부 파라미터 (K, dist) ===
camera_matrix = np.array([
    [679.37865654, 0., 317.98359742],
    [0., 681.58665788, 212.92172904],
    [0., 0., 1.]], dtype=np.float32)

dist_coeffs = np.array([0.12239146, -0.96600967, -0.00930853,
                        0.00392854, 1.63260631], dtype=np.float64)

# === 2. 라벨 실제 크기 (미터 단위) ===
label_width = 0.04
label_height = 0.04

# === 3. OCR Reader 초기화 ===
reader = easyocr.Reader(['en', 'ko'], gpu=False)

# === 한글 출력 함수 ===
def draw_text_korean(img, text, position, font_size=20, font_color=(0,255,0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", font_size)
    draw.text(position, text, font=font, fill=font_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# === Pose 계산 함수 ===
def estimate_label_pose(bbox, K, dist):
    img_pts = np.array(bbox, dtype=np.float32)
    obj_pts = np.array([
        [-label_width/2, -label_height/2, 0],
        [ label_width/2, -label_height/2, 0],
        [ label_width/2,  label_height/2, 0],
        [-label_width/2,  label_height/2, 0]
    ], dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not success:
        return None, None

    camera_coords = (tvec.flatten() * 1000).tolist()  # mm 단위 변환
    rvec_deg = (np.rad2deg(rvec.flatten())).tolist()
    return camera_coords, rvec_deg

# === 5. 카메라 열기 ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[X] 카메라를 열 수 없습니다.")
    sys.exit()

print("카메라 실행 중: 'c' → 캡처 후 OCR 실행, 'q' → 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera Preview", frame)
    key = cv2.waitKey(1) & 0xFF

    # 캡처 & OCR 실행
    if key == ord('c'):
        print("[OCR 실행 중...]")
        sys.stdout.flush()

        results = reader.readtext(frame, detail=1)
        found = False

        for bbox, text, conf in results:
            if conf < 0.6:
                continue

            # numpy 타입일 경우 문자열 변환
            if not isinstance(text, str):
                text = str(text)

            print(f"[인식] {text} (신뢰도: {conf:.2f})")
            sys.stdout.flush()

            coords, rvec_deg = estimate_label_pose(bbox, camera_matrix, dist_coeffs)
 # Pose 계산 후 터미널 출력
            if coords is not None:
                found = True
                roll, pitch, yaw = rvec_deg
                print(f"    좌표: X={coords[0]:.1f}mm, Y={coords[1]:.1f}mm, Z={coords[2]:.1f}mm")
                print(f"    방향: Roll={roll:.1f}°, Pitch={pitch:.1f}°, Yaw={yaw:.1f}°")
                sys.stdout.flush()

                # 시각화
                pts = [tuple(map(int, p)) for p in bbox]
                cv2.polylines(frame, [np.array(pts)], True, (0,255,0), 2)

                info = f"X:{coords[0]:.1f}mm, Y:{coords[1]:.1f}mm, Z:{coords[2]:.1f}mm"
                info2 = f"R:{roll:.1f}° P:{pitch:.1f}° Y:{yaw:.1f}°"
                frame = draw_text_korean(frame, text, pts[0], font_size=25, font_color=(0,255,0))
                frame = draw_text_korean(frame, info, (pts[0][0], pts[0][1]+30), font_size=20, font_color=(0,255,255))
                frame = draw_text_korean(frame, info2, (pts[0][0], pts[0][1]+55), font_size=20, font_color=(255,255,0))

        if not found:
            print("[경고] 타겟을 찾지 못했습니다.")
            sys.stdout.flush()
            frame = draw_text_korean(frame, "Target Not Found", (20, 40), font_size=25, font_color=(255,0,0))

        cv2.imshow("OCR Pose Estimation", frame)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

