import rclpy
from rclpy.node import Node
import time
import numpy as np
from pymycobot.mycobot280 import MyCobot280
from pymycobot.genre import Angle, Coord

from pick_and_place.base_coordinate_transform import transform_target_pose_camera_to_base
from pick_and_place.image_capture import CameraManager  # CameraManager 클래스 가져오기
from pick_and_place.image_detection import detect_target  # detect() 내부에서 _detect_april_tag 호출
from custom_messeage.srv import RobotArmRequest  # srv 경로에 따라 조정 필요

## Note: pick, place 각각 따로 모듈화하기(분리시켜 놓기)


class Robot2ControlNode(Node):
    def __init__(self):
        super().__init__('robot2_control_node')
        self.srv = self.create_service(
            RobotArmRequest,
            'arm2_service',
            self.arm2_control_callback
        )
        self.get_logger().info("arm2_control 서비스 서버가 시작되었습니다.")
        
        self.mc = MyCobot280('/dev/ttyJETCOBOT', 1000000)
        self.mc.thread_lock = True
        print("로봇이 연결되었습니다.")

        # 카메라 초기화 (예외 처리 추가)
        try:
            # self.camera = CameraManager()  # 여러 디바이스 자동 시도

            self.camera = CameraManager(enable_streaming=True, flask_port=5000)  # 스트리밍 활성화
            self.get_logger().info("카메라가 성공적으로 초기화되었습니다.")
        except RuntimeError as e:
            self.get_logger().error(f"카메라 초기화 실패: {e}")
            self.camera = None  # 카메라 없이도 동작하도록
 
    def arm2_control_callback(self, request, response):
        pinky_id = request.amr_id
        action = request.action.lower()
        shelf_num = request.shelf_num

        self.get_logger().info(f'[서비스 요청] action: {action}, shelf_num: {shelf_num}, pinky_num: {pinky_id}')

        if action == 'buffer_to_pinky':
            success, msg = self.handle_buffer_to_pinky(pinky_id)
        elif action == 'pinky_to_buffer':
            success, msg = self.handle_pinky_to_buffer(pinky_id)
        else:
            success, msg = False, f"지원하지 않는 action: {action}"

        ## Note: OCR 인식 후 얻은 데이터 저장
        response.robot_id = 2
        response.amr_id = pinky_id
        response.action = msg
        response.model = 'None'
        response.size = -1
        response.color = 'None'
        response.success = success
        return response


##=========================================================================================
    '''
    RobotArm2가 buffer에서 pinky로 물건을 옮기는 함수
    '''
    def handle_buffer_to_pinky(self, pinky_id):
        # TODO: 실제 로봇 로직 작성
        print(f"{pinky_id} 버퍼를 위한 초기자세로 이동..")
        if pinky_id == 1:
            self.mc.send_angles([137.1, -9.84, -31.28, -30.84, -3.69, 91.14], 20) # 1번 버퍼 보는 초기 자세
        elif pinky_id == 2:
            self.mc.send_angles([119.0, -12.04, -32.34, -36.12, -2.1, 69.78], 20) # 2번 버퍼 보는 초기 자세
        elif pinky_id == 3:
            self.mc.send_angles([86.39, -7.2, -32.34, -39.99, -0.7, 42.8], 20) # 3번 버퍼 보는 초기 자세
        else:
            print("정의되지 않은 핑키 번호")

        print("그리퍼를 완전히 엽니다.")
        self.mc.set_gripper_value(100, 50)
        
        # cur_joints_rad = self.mc.get_radians()
        # print("라디안:", cur_joints_rad)

        # 프레임 가져오고, 프레임에서 에이프릴테그 감지
        time.sleep(5)
        frame = self.camera.get_frame()
        camera_coords, rvec_deg, tag_id = detect_target(frame, target_id=pinky_id) # 타겟 id = pinky_id

        if camera_coords is not None and rvec_deg is not None:
            print("\n=== April Tag 좌표 정보 ===")
            print(f"카메라 기준 좌표: {camera_coords}")
            print(f"회전 벡터 (도): {rvec_deg}")

            print("\n=== Base 좌표계로 변환 중... ===")
            try:
                base_coords = transform_target_pose_camera_to_base(
                    camera_coords, rvec_deg, self.mc.get_radians()
                )

                # roll, pitch, yaw 고정
                base_coords[3], base_coords[4], base_coords[5] = -177.0, 2.0, -52.0
                print(f"베이스 좌표 [x, y, z, roll, pitch, yaw]: {base_coords}")

                approach_coords = base_coords.copy()
                # 버퍼마다 각각의 offset값 설정
                if pinky_id == 1:
                    approach_coords[0] -= 10
                    approach_coords[1] -= 7
                    approach_coords[2] += 70
                elif pinky_id == 2:
                    approach_coords[0] -= 5
                    approach_coords[1] -= 10
                    approach_coords[2] += 70
                elif pinky_id == 3:
                    approach_coords[0] -= 10
                    approach_coords[1] -= 10
                    approach_coords[2] += 70
                else:
                    print("제대로된 핑키 번호 입력하시오.")

                self.mc.send_coords(approach_coords, 20, 1)
                print("로봇 이동(2단계) 명령을 전송했습니다.")
                print("이동 완료까지 대기 중...")
                time.sleep(3)

                current_coords = self.mc.get_coords()

                # IK의 해를 못찾아서 로봇이 이동하지 못할 경우 예외 처리
                # |가야할 좌표값 - 이동 후 좌표값| 
                ## Note: 되는데 안된다고 판단할 위험 있음 (오차 범위 재설정 필요)
                if np.linalg.norm(np.array(approach_coords) - np.array(current_coords)) > 30.0:  # mm 단위 기준
                    print("이동할 수 없는 좌표값입니다.")
                    return False, '갈 수 없는 베이스 좌표로 인한 실패'

                print(f"이동 후 현재 좌표: {current_coords}")

                # 물체 잡기
                time.sleep(3)
                self.mc.set_gripper_value(0, 50)  # 그리퍼 닫기

            except Exception as e:
                print(f"좌표 변환 또는 로봇 이동 중 오류 발생: {e}")
                return False, "좌표 변환 또는 로봇 이동 중 오류 발생"

        else:
            print("April Tag 좌표를 가져올 수 없습니다.")
            return False, '에이프릴 테그 인식 실패' # "버퍼에서 핑키로 이동 실패"
        
        #ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

        # 수거존 근처로 이동 (AprilTag 인식 전에, 수동 조인트값 위치)
        print("\n=== 핑키로 place를 위한 이동 ===")
        
        # 핑키 바라보는 각도
        self.mc.send_angles([-14.67, 91.58, -87.62, -37.79, -6.67, 44.2], 20)
        time.sleep(5)

        # 수거존 위치 인식
        print("\n=== 핑키 AprilTag 인식 ===")
        frame = self.camera.get_frame()
        camera_coords2, rvec_deg2, tag_id2 = detect_target(frame, target_id=pinky_id) # 타겟 id 설정

        if camera_coords2 is not None and rvec_deg2 is not None:
            try:
                base_coords2 = transform_target_pose_camera_to_base(
                    camera_coords2, rvec_deg2, self.mc.get_radians()
                )

                # 그리퍼 방향은 그대로 유지 (roll, pitch, yaw 고정)
                base_coords2[3], base_coords2[4], base_coords2[5] = -136.02, 32.56, -131.44

                # x,y,z 보정
                base_coords2[0] -= 80
                base_coords2[1] += 0
                base_coords2[2] += 80
                print(f"수거존으로 이동할 좌표: {base_coords2}")

                #경유지
                self.mc.send_angles([-10.81, 62.4, -118.74, 8.43, -3.69, 44.38], 20)

                #도착지
                time.sleep(2)
                self.mc.send_coords(base_coords2, 20, 1)
                print("수거존 위치로 이동 완료")

                # 물건 내려놓기
                time.sleep(3)
                self.mc.set_gripper_value(100, 50)  # 그리퍼 열기
                print("그리퍼를 열었습니다. 물건을 내려놓았습니다.")
                
            except Exception as e:
                print(f"수거존 이동 중 오류 발생: {e}")
        else:
            print("수거존 AprilTag 인식 실패")
            return False, '에이프릴 테그 인식 실패' # "버퍼에서 핑키로 이동 실패

        # 초기 위치로 이동
        self.mc.send_angles([-10.81, 62.4, -118.74, 8.43, -3.69, 44.38], 20)
        time.sleep(2)
        self.mc.send_angles([0, 0, 0, 0, 0, 40], 20)

        self.get_logger().info(f"버퍼 → 핑키{pinky_id}")
        return True, 'arm2_buffer_to_pinky' # "버퍼에서 핑키로 이동 완료"
    
##=========================================================================================
    
    '''
    RobotArm2가 pinky에서 buffer로 물건을 옮기는 함수
    '''
    def handle_pinky_to_buffer(self, pinky_id):
        # TODO: 실제 로봇 로직 작성
        
        # 핑키 바라보는 위치로 이동
        print("\n[1]: 핑키 방향으로 이동 중...")
        self.mc.send_angles([-14.67, 91.58, -87.62, -37.79, -6.67, 44.2], 20)
        self.mc.set_gripper_value(100, 50)  # 그리퍼 열기
        time.sleep(5)

        print("그리퍼를 완전히 엽니다.")
        self.mc.set_gripper_value(100, 50)
        
        # cur_joints_rad = self.mc.get_radians()
        # print("라디안:", cur_joints_rad)

        # 프레임 가져오고, 프레임에서 에이프릴테그 감지
        print("\n[2] :brain: AprilTag 인식 중...")
        frame = self.camera.get_frame()
        camera_coords, rvec_deg, tag_id = detect_target(frame, target_id=3) # 타겟 id 설정

        if camera_coords is not None and rvec_deg is not None:
            print("\n=== April Tag 좌표 정보 ===")
            print(f"카메라 기준 좌표: {camera_coords}")
            print(f"회전 벡터 (도): {rvec_deg}")

            print("\n=== Base 좌표계로 변환 중... ===")
            try:
                base_coords = transform_target_pose_camera_to_base(
                    camera_coords, rvec_deg, self.mc.get_radians()
                )

                # roll, pitch, yaw 고정
                base_coords[3], base_coords[4], base_coords[5] = -150.0, 25.0, -138.0
                print(f"베이스 좌표 [x, y, z, roll, pitch, yaw]: {base_coords}")

                # 핑크로 가는 경유지로 이동 (1차)
                print("\n[3]: 경유지(1차) 이동 중...")
                self.mc.send_angles([-10.81, 62.4, -118.74, 8.43, -3.69, 44.38], 20)
                time.sleep(3)

                # offset값 설정
                base_coords[0] -= 50
                base_coords[1] += 10
                base_coords[2] += 70
                
                # base 기준 좌표로 이동 후 물건 잡기
                self.mc.send_coords(base_coords, 20, 1)
                time.sleep(3)
                self.mc.set_gripper_value(0, 50)  # 그리퍼 닫기
                time.sleep(1)

                # 핑키에서 후진하는 경유지
                print("\n[4]: 경유지(후진) 이동")
                self.mc.send_angles([-10.81, 62.4, -118.74, 8.43, -3.69, 44.38], 20)
                time.sleep(3)

                # 콜렉션 이동하기 전에 경유지
                self.mc.send_angles([76.64, -21.0, -16.78, -40.42, 0.87, 29.35], 20)
                time.sleep(3)
                
                # collection으로 이동
                print(f"\n[5]: 버퍼로 이동 중...")
                self.mc.send_angles([78.04, -53.43, -19.07, -11.33, 1.05, 33.66], 20)
                time.sleep(3)

                # 놓기
                print("\n[6]: 그리퍼 열기")
                self.mc.set_gripper_value(100, 50)
                time.sleep(1)

                # 초기위치로 가기 전의 경유지 후진 
                print("\n[7]: 후진(상승) 동작")
                self.mc.send_angles([76.64, -21.0, -16.78, -40.42, 0.87, 29.35], 20)
                time.sleep(3)

                # 초기 위치(핑키 바라보는 방향) 복귀
                print("\n[8]: 초기 위치 복귀")
                self.mc.send_angles([-14.67, 91.58, -87.62, -37.79, -6.67, 44.2], 20)
                time.sleep(2)
                self.mc.send_angles([0, 0, 0, 0, 0, 40], 20)
                print("\n[9]: 작업 완료")

            except Exception as e:
                print(f"좌표 변환 또는 로봇 이동 중 오류 발생: {e}")

        else:
            print("April Tag 좌표를 가져올 수 없습니다.")
            return False, '에이프릴 테그 인식 실패' # "버퍼에서 핑키로 이동 실패"
        
        self.get_logger().info(f"핑키{pinky_id} → 버퍼")
        return True, 'arm2_pinky_to_buffer' # "핑키에서 버퍼로 이동 완료"
##=========================================================================================


def destroy_node(self):
    """노드 종료 시 리소스 정리"""
    if self.camera:
        self.camera.release()  # 카메라 해제
    super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = Robot2ControlNode()
    try:
        rclpy.spin(node)
    finally:
        node.camera.release()  # 안전하게 종료 시 카메라 해제
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()