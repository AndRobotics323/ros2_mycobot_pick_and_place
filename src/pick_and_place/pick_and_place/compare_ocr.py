
import time
import requests
import cv2
import numpy as np
import sys

import base64
import argparse

# from pick_and_place.image_detection import detect_target  # detect() 내부에서 _detect_april_tag 호출
# from pick_and_place.image_detection_ocr import detect_target_ocr

from pymycobot.mycobot280 import MyCobot280

from image_detection import detect_target  # detect() 내부에서 _detect_april_tag 호출
from image_detection_ocr import detect_target_ocr

from pick_and_place.base_coordinate_transform import transform_target_pose_camera_to_base


streaming_flask_url = "http://192.168.0.168:5000/stream" # arm2 주소임


def compare_two(mode):

    frame = None
 
    if mode == 'ocr':

        camera_coords, rvec_deg, cur_model, cur_color, cur_size = detect_target_ocr(frame) # 타겟 id 설정 3 >> id

        print('ocr result')
        print(camera_coords)
        # print(rvec_deg)

        print(cur_model)
        print(cur_color)
        print(cur_size)

        return camera_coords, rvec_deg


    elif mode == 'april':

        stream = requests.get(streaming_flask_url , stream=True)
        byte_data = b''

        for chunk in stream.iter_content(chunk_size=1024):
            byte_data += chunk
            start = byte_data.find(b'\xff\xd8')
            end = byte_data.find(b'\xff\xd9')
            if start != -1 and end != -1:
                jpg = byte_data[start:end+2]
                npimg = np.frombuffer(jpg, np.uint8)
                frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                break


        camera_coords, rvec_deg, tag_id = detect_target(frame, target_id=id) # 타겟 id 설정 3 >> id

        print('april result')
        print(camera_coords)
        # print(rvec_deg)

        return camera_coords, rvec_deg
    


if __name__ == '__main__':

    # compare_two('ocr')
    # compare_two('april')


    # # 핑키 바라보는 위치로 이동
    mc = MyCobot280( '/dev/ttyJETCOBOT' , 1000000 )


    # print("\n[1]: 핑키 방향으로 이동 중...")
    mc.send_angles([-14.23, 44.56, -23.55, -49.65, -0.61, 35.59], 25)
    mc.set_gripper_value(100, 50)  # 그리퍼 열기
    time.sleep(3)

    # print("그리퍼를 완전히 엽니다.")
    mc.set_gripper_value(100, 50)
    mc.send_angles([119.0, -12.04, -32.34, -36.12, -2.1, 69.78], 20) # 2번 버퍼 보는 초기 자세


    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["ocr", "april"], help="Choose which pipeline to run")
    args = parser.parse_args()


    if args.mode == "ocr":
        camera_coords, rvec_deg = compare_two('ocr')
        print("\n[2] :brain: OCR 인식 중...")

    elif args.mode == "april":
        camera_coords, rvec_deg = compare_two('april')
        print("\n[2] :brain: AprilTag 인식 중...")



    sys.exit(0)  # 0은 정상 종료, 다른 숫자는 에러 코드


    if camera_coords is not None and rvec_deg is not None:

        print("\n=== OCR 좌표 정보 ===")
        print(f"카메라 기준 좌표: {camera_coords}")
        print(f"회전 벡터 (도): {rvec_deg}")

        print("\n=== Base 좌표계로 변환 중... ===")
        try:
            base_coords = transform_target_pose_camera_to_base(
                camera_coords, rvec_deg, mc.get_radians()
            )

            # roll, pitch, yaw 고정
            base_coords[3], base_coords[4], base_coords[5] = -92.77, 39.31, -87.4
            print(f"베이스 좌표 [x, y, z, roll, pitch, yaw]: {base_coords}")

            # 핑크로 가는 경유지로 이동 (1차)
            print("\n[3]: 경유지(1차) 이동 중...")
            mc.send_angles([99.75, 121.55, -114.08, -1.05, -101.95, 42.18], 25)
            time.sleep(3)

            # offset값 설정
            base_coords[0] -= 57
            base_coords[1] += 0
            base_coords[2] -= 10
            
            # base 기준 좌표로 이동 후 물건 잡기
            mc.send_coords(base_coords, 20, 1)
            time.sleep(3)
            mc.set_gripper_value(0, 50)  # 그리퍼 닫기
            time.sleep(1)

            # 핑키에서 후진하는 경유지
            print("\n[4]: 경유지(후진) 이동")
            mc.send_angles([64.59, 118.74, -111.88, -11.07, -60.38, 42.53], 30)
            time.sleep(3)

            # collection으로 이동
            print(f"\n[5]: 버퍼로 이동 중...")
            mc.send_angles([76.2, -45.7, -39.72, 5.8, 7.2, 26.89], 30) 
            time.sleep(3)
            
            # 놓기
            print("\n[6]: 그리퍼 열기")
            mc.set_gripper_value(100, 50)
            time.sleep(1)

            # 초기 위치(핑키 바라보는 방향) 복귀
            print("\n[7]: 초기 위치 복귀")
            mc.send_angles([-14.67, 91.58, -87.62, -37.79, -6.67, 44.2], 25)
            time.sleep(2)
            
            mc.send_angles([0, 0, 0, 0, 0, 40], 25)
            print("\n[8]: 작업 완료")

        except Exception as e:
            print(f"좌표 변환 또는 로봇 이동 중 오류 발생: {e}")




    else:
        print("OCR 좌표를 가져올 수 없습니다.")
    





