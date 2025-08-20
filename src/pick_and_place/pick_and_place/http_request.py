
import requests
import base64
import cv2
import numpy as np
import re
from collections import Counter


# CJ 192.168.0.189

django_url = 'http://192.168.0.189:8000/gwanje/ocr_from_flask_stream/'
# django_url = 'http://192.168.5.17:8000/gwanje/ocr_from_flask_stream/'
# django_url = 'https://robocallee.jp.ngrok.io/gwanje/ocr_from_flask_stream/'


defined_models =['나이키', '아디다스', '뉴발란스', '반스', '컨버스','푸마']
defined_colors =['white', 'black', 'red', 'blue', 'green', 'yellow', 'gray', 'brown', 'pink', 'purple']
defined_sizes = ['230', '235', '240', '245', '250', '255', '260', '265', '270', '275', '280', '285', '290']


def classify_color(bgr):
    color_ranges = {
        'red':    [(0, 70, 50), (10, 255, 255)],
        'yellow': [(20, 70, 50), (35, 255, 255)],
        'green':  [(36, 70, 50), (85, 255, 255)],
        'blue':   [(86, 70, 50), (125, 255, 255)],
        'purple': [(126, 70, 50), (160, 255, 255)],
        'black':  [(0, 0, 0), (180, 255, 50)],
        'white':  [(0, 0, 200), (180, 30, 255)],
        'gray':   [(0, 0, 51), (180, 30, 199)]
    }
    hsv = cv2.cvtColor(np.uint8( [ [ bgr ] ] ), cv2.COLOR_BGR2HSV)[0][0]
    for color, (lower, upper) in color_ranges.items():
        if all(lower[i] <= hsv[i] <= upper[i] for i in range(3) ):
            return color
    return 'unknown'


def get_dominant_color(img, pts):
    # pts: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] (시계방향)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    masked = cv2.bitwise_and(img, img, mask=mask)
    pixels = masked[mask == 255].reshape(-1, 3)
    pixels = [tuple(p) for p in pixels if np.any(p)]  # 0,0,0 제외

    if not pixels:
        return None

    most_common = Counter(pixels).most_common(1)[0][0]
    return most_common  # (B, G, R)

    # mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # # 좌표들을 numpy array로 강제 변환 (x,y 정수쌍)
    # pts = np.array([tuple(map(int, p)) for p in pts], dtype=np.int32)
    # pts = pts.reshape((-1, 1, 2))  # (N,1,2) 형태로 변환

    # cv2.fillPoly(mask, [pts], 255)

    # mean = cv2.mean(img, mask=mask)[:3]
    # return tuple(map(int, mean))




def ask_django_ocr(url , mode):
    # data = {'name': 'CJ'}
    response = requests.post(url)

    print("Status code:", response.status_code)
    # print("Response text:", response.text)

    data = response.json()


    if len(data) == 0:
        print("탐지된 텍스트 없음! ")
        return None
    
    word_coords = data['results']
    

    # 예시 데이터 구조
    # print( word_coords )

    # [    {        'text': 'Hello',        'coords': [(100, 200), (150, 200), (150, 230), (100, 230)]    },
    #     {        'text': '안녕',        'coords': [(300, 400), (350, 400), (350, 430), (300, 430)]    }]

    if mode == 'get_coords':
        return word_coords


    elif mode == 'get_shoe_info':
        tmp_dict = { 'model': '아직', 'color' : 'yet', 'size': -1, 'coords': [0.0, 0.0, 0.0, 0.0]  }
        
        box_l_top, box_r_top, box_r_bottom,  box_l_bottom = None, None, None, None
        color_l_top, color_r_top, color_r_bottom,  color_l_bottom = None, None, None, None
        center_coord = None

        print('# detected words: ' + str( len(word_coords) ) )
        for item in word_coords:
            text = item['text']

            if re.fullmatch(r'[가-힣]+', text) and text in defined_models:
                tmp_dict['model'] = text
                
                box_l_top = item['coords'][0]
                box_r_top = item['coords'][1]

                color_r_top = item['coords'][2]

                # print(text)


            # elif re.fullmatch(r'[A-Za-z]+', text) and text in defined_colors:
                # tmp_json['color'] = text 
            
            elif re.fullmatch(r'[0-9]+', text) and text in defined_sizes:
                tmp_dict['size'] = int(text)

                box_l_bottom = item['coords'][3]

                color_l_top = item['coords'][1]
                color_l_bottom = item['coords'][2]

                
                # print(text)



        # 정사각형 중점
        center_coord = (    (box_l_bottom[0] + box_r_top[0]) / 2.0,  (box_l_bottom[1] + box_r_top[1]) / 2.0  )

        # 오른쪽 아래
        color_r_bottom = ( ( 2.0 * center_coord[0] - box_l_top[0] )  , ( 2.0 * center_coord[1] - box_l_top[1] ) ) 

        box_r_bottom = color_r_bottom

        tmp_dict['coords'] = [box_l_top, box_r_top, box_r_bottom, box_l_bottom]

        #이제 색깔 구하기

        img_b64 = data['image']
        img_bytes = base64.b64decode(img_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        dominant_color = get_dominant_color(img, [color_l_top, color_r_top, color_r_bottom, color_l_bottom] )
        tmp_dict['color'] = classify_color( dominant_color )

        return tmp_dict


    else : # mode == 'show_image'
        # Base64 디코딩
        img_b64 = data['image']
        img_bytes = base64.b64decode(img_b64)

        # NumPy 배열로 변환
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # 화면에 표시
        cv2.imshow('Received Image', img)

        while True:
            key = cv2.waitKey(1) & 0xFF  # 1ms 대기 후 키 입력 확인
            
            # 예: 'q' 키를 누르면 종료
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        return None



if __name__ == '__main__':
    
    # ask_django_ocr(django_url, 'get_coords')
    ask_django_ocr(django_url, 'get_shoe_info')
    # ask_django_ocr(django_url, 'show_image')


