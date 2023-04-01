import cv2
from PIL import Image

from scanning import scan_process


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def take_screen(self):
        print("-- taking screen --")
        _, image = self.video.read()
        img = Image.fromarray(image)
        img = img.convert('L')
        img.save("screen.jpg")
        classified_zahlen = scan_process("screen.jpg")
        if len(classified_zahlen) != 0:
            return classified_zahlen
        else:
            return None

    def get_frame(self):
        success, image = self.video.read()
        image = cv2.flip(image, 1)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

