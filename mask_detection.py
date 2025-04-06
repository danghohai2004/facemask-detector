import cv2
import math
import tkinter as tk
import time
from tkinter import filedialog
from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

classNames = ['Incorrect_Mask', 'With_Mask', 'Without_Mask']
box_colors = {
    "Incorrect_Mask": (0, 0, 255),
    "With_Mask": (0, 255, 0),
    "Without_Mask": (255, 0, 0)
}

def start_detection(model_path, model_name):
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    model = YOLO(model_path).to(device)
    prev_time = time.time()
    fps_list = []
    last_fps_update = time.time()
    display_fps = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        counts = {"Incorrect_Mask": 0, "With_Mask": 0, "Without_Mask": 0}

        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]

                if cls == 0:
                    counts["Incorrect_Mask"] += 1
                elif cls == 1:
                    counts["With_Mask"] += 1
                elif cls == 2:
                    counts["Without_Mask"] += 1

                color = box_colors[class_name]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                org = [x1, y1 - 10]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.6
                thickness = 2
                cv2.putText(img, f"{class_name} ({confidence})", org, font, fontScale, color, thickness)

        cv2.putText(img, f"Incorrect_Mask: {counts['Incorrect_Mask']}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    box_colors["Incorrect_Mask"], 2)
        cv2.putText(img, f"With_Mask: {counts['With_Mask']}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    box_colors["With_Mask"], 2)
        cv2.putText(img, f"Without_Mask: {counts['Without_Mask']}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    box_colors["Without_Mask"], 2)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        fps_list.append(fps)
        prev_time = curr_time

        if curr_time - last_fps_update >= 1.0:
            display_fps = int(sum(fps_list) / len(fps_list)) if fps_list else 0
            fps_list.clear()
            last_fps_update = curr_time

        cv2.putText(img, f"FPS: {display_fps}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 255), 2)

        cv2.imshow(f'Webcam - {model_name}', img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def browse_file(entry):
    file_path = filedialog.askopenfilename(filetypes=[("PyTorch Model Files", "*.pt"), ("All Files", "*.*")])
    if file_path:
        entry.delete(0, tk.END)
        entry.insert(0, file_path)

def on_ok_button(entry):
    model_path = entry.get()
    if model_path:
        model_name = model_path.split("/")[-1].split(".")[0]
        start_detection(model_path, model_name)

root = tk.Tk()
root.title("YOLO Mask Detection")
root.geometry("500x200")
root.config(bg="#2e3d49")

label = tk.Label(root, text="Select YOLO Model File", font=("Helvetica", 14), bg="#2e3d49", fg="white")
label.pack(pady=20)

frame = tk.Frame(root, bg="#2e3d49")
frame.pack(pady=10)

entry = tk.Entry(frame, width=40, font=("Helvetica", 12))
entry.grid(row=0, column=0, padx=10)

browse_button = tk.Button(frame, text="Browse", font=("Helvetica", 12), width=10, height=1,
                          bg="#4CAF50", fg="white", relief="solid", command=lambda: browse_file(entry))
browse_button.grid(row=0, column=1)

ok_button = tk.Button(root, text="OK", font=("Helvqqetica", 12), width=20, height=2,
                      bg="#4CAF50", fg="white", relief="solid", command=lambda: on_ok_button(entry))
ok_button.pack(pady=20)

root.mainloop()
