import cv2
import datetime
import os
from ultralytics import YOLO

# 모델 로드
model = YOLO('../yolo11n.pt')  # 경로는 너의 pt 파일 위치에 맞게 조정

# 위험구역 설정 (좌상단, 우하단 좌표)
danger_zone = ((100, 100), (400, 400))

# 프레임 저장 디렉토리 생성
SAVE_DIR = 'saved_frames'
os.makedirs(SAVE_DIR, exist_ok=True)

# 침입 여부 판단 함수
def is_intruding(xyxy, danger_zone):
    x1, y1, x2, y2 = xyxy
    dz_x1, dz_y1 = danger_zone[0]
    dz_x2, dz_y2 = danger_zone[1]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return dz_x1 < center_x < dz_x2 and dz_y1 < center_y < dz_y2

# 영상 불러오기
cap = cv2.VideoCapture('../ee.mp4')  # mp4 파일 경로 확인
if not cap.isOpened():
    print("영상 열기 실패!")
    exit()

# 영상 저장 설정
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_intrusion.mp4', fourcc, fps, (width, height))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame)[0]  # 첫 번째 예측 결과
    boxes = results.boxes

    # 위험구역 표시
    cv2.rectangle(frame, danger_zone[0], danger_zone[1], (0, 0, 255), 2)

    if boxes is not None:
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            cls = int(box.cls[0].cpu().item())

            if cls != 0:  # person 클래스만 필터링
                continue

            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if is_intruding((x1, y1, x2, y2), danger_zone):
                print(f"[{frame_count}] 침입 감지됨!")
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = os.path.join(SAVE_DIR, f"intrusion_{timestamp}.jpg")
                cv2.putText(frame, "INTRUSION!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imwrite(filename, frame)

    out.write(frame)

# 종료 처리
cap.release()
out.release()
print("분석 완료: output_intrusion.mp4 저장됨")