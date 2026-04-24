import cv2
import sys
from tkinter import *

class App:
    def __init__(self, src):
        self.root = Tk()
        self.root.title("Control Panel")
        self.root.geometry("200x150")
        
        Button(self.root, text="Clear (C)", command=self.clear, 
              width=20, height=2).pack(pady=5)
        
        Button(self.root, text="Quit (Q)", command=self.exit, 
              width=20, height=2).pack(pady=5)
        
        self.cap = cv2.VideoCapture(0 if src == "camera" else src)
        if not self.cap.isOpened():
            print("Error")
            sys.exit(1)
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.delay = int(1000 / self.fps)
        
        self.rectangles = []
        self.running = True
        
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Video", self.mouse_callback)
        
        self.video_loop()
        self.root.mainloop()
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            size = 50
            self.rectangles.append((x - size, y - size, x + size, y + size))
    
    def clear(self):
        self.rectangles.clear()
    
    def exit(self):
        #print("Exiting...")
        self.running = False
        self.root.quit()
        self.cap.release()
        cv2.destroyAllWindows()
        sys.exit()

    def video_loop(self):
        while self.running:
            if cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1:
                self.running = False
                break

            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            for (x1, y1, x2, y2) in self.rectangles:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 5)
            
            cv2.imshow("Video", frame)
            key = cv2.waitKey(self.delay) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                self.running = False
            elif key == ord('c') or key == ord('C'):
                self.rectangles.clear()
            
            self.root.update()

        self.exit()

if len(sys.argv) < 2:
    print("Use: python CV_HW_1_Volkova.py camera or python CV_HW_1_Volkova.py video.mp4")
else:
    App(sys.argv[1])