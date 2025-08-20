"""
ocr을 통해 camera_coords와 rvec_deg, 및 신발 정보를 구하는 모듈
"""
import sys
import cv2
import numpy as np

# from pick_and_place.http_request import ask_django_ocr  # detect() 내부에서 _detect_april_tag 호출
from http_request import ask_django_ocr  # detect() 내부에서 _detect_april_tag 호출

django_url = 'http://192.168.0.189:8000/gwanje/ocr_from_flask_stream/'
# django_url = 'http://192.168.5.17:8000/gwanje/ocr_from_flask_stream/'
# django_url = 'https://robocallee.ngrok.io/gwanje/ocr_from_flask_stream/'
 

# === 카메라 내부 파라미터 설정 ===
camera_matrix = np.array([[1018.8890899848071, 0., 372.64373648977255],
                          [0., 1016.7247236426332, 229.30521863962326],
                          [0., 0., 1.]], dtype=np.float32)
dist_coeffs = np.array([-0.4664, 2.0392, 0.00035, -0.00077, -16.977], dtype=np.float64)
tag_size = 0.02  # 단위: meter


# === Pose 계산 함수 ===
def estimate_label_pose(bbox, K, dist):
    img_pts = np.array(bbox, dtype=np.float32)
    obj_pts = np.array([
        [-tag_size/2, -tag_size/2, 0],
        [ tag_size/2, -tag_size/2, 0],
        [ tag_size/2,  tag_size/2, 0],
        [-tag_size/2,  tag_size/2, 0]
    ], dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not success:
        return None, None

    # camera_coords = (tvec.flatten() * 1000 ).tolist()  # mm 단위 변환
    camera_coords = (tvec.flatten() * 1000 ).tolist()  # mm 단위 변환

    rvec_deg = (np.rad2deg(rvec.flatten())).tolist()
    return camera_coords, rvec_deg



# camera_coords, rvec_deg, cur_model, cur_color, cur_size = detect_target_ocr(frame) # 타겟 id 설정 3 >> id
def _detect_ocr(frame, camera_matrix):
  
    # tmp_dict = { 'model': '아직', 'color' : 'yet', 'size': -1, 'coords': [0,0,0,0]  }
    tmp_dict = ask_django_ocr(django_url, 'get_shoe_info')

    four_points = tmp_dict['coords']

    camera_coords, rvec_deg = estimate_label_pose(four_points, camera_matrix, dist_coeffs)


# Pose 계산 후 터미널 출력
    if camera_coords is not None:
        camera_coords = [x * 10 for x in camera_coords]


        found = True
        roll, pitch, yaw = rvec_deg
        print(f"    좌표: X={camera_coords[0]:.1f}mm, Y={camera_coords[1]:.1f}mm, Z={camera_coords[2]:.1f}mm")
        # print(f"    방향: Roll={roll:.1f}°, Pitch={pitch:.1f}°, Yaw={yaw:.1f}°")
        

    return camera_coords, rvec_deg, tmp_dict['model'], tmp_dict['color'], tmp_dict['size']





def detect_target_ocr(frame):
    """
    외부에서 호출하는 함수: frame을 받아 AprilTag pose를 추출

    Args:
        frame (np.ndarray): BGR 이미지

    Returns:                                       
        camera_coords (list): [x, y, z] in mm
        rvec_deg (list): [rx, ry, rz] in degrees
    """
    return _detect_ocr(frame, camera_matrix )  # 추후 YOLO 등으로 교체 가능