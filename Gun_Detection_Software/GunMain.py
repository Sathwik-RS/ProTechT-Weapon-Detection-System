import cv2
from ultralytics import YOLO
import datetime
import smtplib
import ssl
from email.message import EmailMessage
import os

# ==== EMAIL SETUP ====
sender_email = "rssathwik5@gmail.com"        # 🔁 your Gmail here
app_password = ""           # 🔁 your app password here
subject = "🚨 Gun Detection Alert"

admin_email = input("Enter admin email to notify: ").strip()

def send_email_with_image(recipient_email, image_path):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg.set_content("Gun has been detected. See attached image.")

    with open(image_path, 'rb') as f:
        file_data = f.read()
        file_name = os.path.basename(image_path)
        msg.add_attachment(file_data, maintype='image', subtype='jpeg', filename=file_name)

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        print(f"✅ Alert email sent to {recipient_email}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# ==== LOAD MODEL ====
model = YOLO(r"C:\Users\rssat\OneDrive\Desktop\Gun_Detection_Software\gun-detection\best3.pt")
class_names = model.names
gun_class_ids = [i for i, name in class_names.items() if "gun" in name.lower()]

if not gun_class_ids:
    raise ValueError("❌ Your model does not include 'gun' in its class names!")

cap = cv2.VideoCapture(0)
frame_count = 0
tracked_gun_ids = set()
gun_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue

    results = model.track(frame, conf=0.5, iou=0.4, persist=True)

    if results[0].boxes is None:
        continue

    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    ids = results[0].boxes.id
    ids = ids.cpu().numpy() if ids is not None else [None] * len(boxes)

    for box, class_id, conf, obj_id in zip(boxes, class_ids, confs, ids):
        class_id = int(class_id)
        if class_id not in gun_class_ids:
            continue

        if obj_id is not None:
            tracked_gun_ids.add(int(obj_id))

            if not gun_detected:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_filename = f"gun_detected_{timestamp}.jpg"
                cv2.imwrite(screenshot_filename, frame)
                print(f"📸 Screenshot saved: {screenshot_filename}")
                print("🚨 Gun Detected! Sending email alert...")
                
                send_email_with_image(admin_email, screenshot_filename)
                
                gun_detected = True
                break

    if gun_detected:
        break

    for box, class_id, obj_id in zip(boxes, class_ids, ids):
        if obj_id is None or int(obj_id) not in tracked_gun_ids:
            continue

        x1, y1, x2, y2 = map(int, box)
        gun_id = int(obj_id)
        label = f"Gun #{gun_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Gun Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

