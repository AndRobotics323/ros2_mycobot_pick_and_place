
import time
import requests
import cv2
import numpy as np

import base64
import argparse

# from pick_and_place.image_detection import detect_target  # detect() 내부에서 _detect_april_tag 호출
# from pick_and_place.image_detection_ocr import detect_target_ocr


from image_detection import detect_target  # detect() 내부에서 _detect_april_tag 호출
from image_detection_ocr import detect_target_ocr


# streaming_flask_url = "http://192.168.0.168:5000/stream" # arm2 주소임
streaming_flask_url = "http://192.168.35.138:5000/stream"


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


    elif mode == 'april':

        # streaming_flask_url = "http://192.168.0.168:5000/stream" # arm2 주소임
        streaming_flask_url = "http://192.168.35.138:5000/stream"

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


if __name__ == '__main__':
    # compare_two('ocr')
    # compare_two('april')

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["ocr", "april"], help="Choose which pipeline to run")
    args = parser.parse_args()

    if args.mode == "ocr":
        compare_two('ocr')
    elif args.mode == "april":
        compare_two('april')