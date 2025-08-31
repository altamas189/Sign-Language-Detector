import os
import cv2

data_dir = './data'
os.makedirs(data_dir, exist_ok=True)#make dict if its already not  available 


class_name = input("Enter NEW class name : ").strip()

if class_name == "":
    print(" Invalid class name.")
    exit()

class_dir = os.path.join(data_dir, class_name)


if os.path.exists(class_dir):
    print(f" Class '{class_name}' already exists. Please choose a new name.")
    exit()

os.makedirs(class_dir)
print(f"\nNew class '{class_name}' created at: {class_dir}")

# Set target image count
data_size = 50
cap = cv2.VideoCapture(0)


print(f'\n Ready to collect data for "{class_name}"...')
print(" Press 'Q' to start capturing.")
while True:
    ret, frame = cap.read()
    cv2.putText(frame, f'Class: {class_name} - Press "Q" to start :)', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Begin image capture
print(f"\n Collecting {data_size} images for '{class_name}'...")
counter = 0
while counter < data_size:
    ret, frame = cap.read()
    img_path = os.path.join(class_dir, f'{counter}.jpg')
    cv2.imwrite(img_path, frame)
    counter += 1
    cv2.putText(frame, f'Collecting {counter}/{data_size}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) #frame constrains 
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('e'):  
        break

print(" Data collection completed!")
cap.release()
cv2.destroyAllWindows()
