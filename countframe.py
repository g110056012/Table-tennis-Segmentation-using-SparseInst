import cv2

inputFile = "./table-tennis/ball_data/M-5.MOV"
cap = cv2.VideoCapture(inputFile, 0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frames_per_second = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('./table-tennis/ball_data/M-5_frame_count.mp4', fourcc, frames_per_second, (1920,  1080))

frame_counter = 0
while cap.isOpened():
    frame_counter += 1
    ret, frame = cap.read()     # read current frame
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv2.namedWindow('countframe', 0)
    cv2.putText(frame, "Frame: " + str(frame_counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5, cv2.LINE_4)
    cv2.imshow('countframe', frame)

    if cv2.waitKey(1) == 27:
        break  # esc to quit)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()