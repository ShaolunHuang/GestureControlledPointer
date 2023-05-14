import collections

import cv2
import mediapipe as mp
import time
import math
from enum import Enum


class HandType(Enum):
    LEFT = "Left"
    RIGHT = "Right"


class VirtualInputOnMP:
    def __init__(self, mode=False, max_hands=2, hand_type=HandType.RIGHT):
        self.hand_list = collections.defaultdict(list)
        self.results = None
        self.mode = mode
        self.max_hands = max_hands
        self.hand_type = hand_type

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands)
        self.mp_draw_utilities = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmark in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw_utilities.draw_landmarks(img, hand_landmark,
                                                          self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, draw=True):
        h, w, c = img.shape
        self.hand_list = collections.defaultdict(list)
        if self.results.multi_hand_landmarks:
            for hand_class, hand_landmarks in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                x_list = []
                y_list = []
                curr_lmList = []
                curr_hand = {}
                for hand_id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    x_list.append(cx)
                    y_list.append(cy)
                    curr_lmList.append([hand_id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                xmin, xmax = min(x_list), max(x_list)
                ymin, ymax = min(y_list), max(y_list)
                bbox = xmin, ymin, xmax, ymax
                if draw:
                    cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                                  (0, 255, 0), 2)
                curr_hand["lmList"] = curr_lmList
                curr_hand["bbox"] = bbox
                self.hand_list[hand_class.classification[0].label].append(curr_hand)

        return self.hand_list, img

    def fingers_up(self):
        left_fingers = []
        right_fingers = []
        if len(self.hand_list["Left"]) != 0:
            left_lmList = self.hand_list["Left"][0]["lmList"]
            if left_lmList[self.tipIds[0]][1] > left_lmList[self.tipIds[0] - 1][1]:
                left_fingers.append(1)
            else:
                left_fingers.append(0)
            for id in range(1, 5):
                if left_lmList[self.tipIds[id]][2] < left_lmList[self.tipIds[id] - 2][2]:
                    left_fingers.append(1)
                else:
                    left_fingers.append(0)
        if len(self.hand_list["Right"]) != 0:
            right_lmList = self.hand_list["Right"][0]["lmList"]
            if right_lmList[self.tipIds[0]][1] < right_lmList[self.tipIds[0] - 1][1]:
                right_fingers.append(1)
            else:
                right_fingers.append(0)
            for id in range(1, 5):
                if right_lmList[self.tipIds[id]][2] < right_lmList[self.tipIds[id] - 2][2]:
                    right_fingers.append(1)
                else:
                    right_fingers.append(0)

        return left_fingers, right_fingers

    def find_distance(self, p1, p2, img, draw=True, r=15, t=3):
        left_length = -1
        right_length = -1
        cx_left, cy_left = -1, -1
        cx_right, cy_right = -1, -1

        def get_length(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if draw:
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
                cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)
            return length, cx, cy

        if len(self.hand_list["Left"]) != 0:
            left_lmList = self.hand_list["Left"][0]["lmList"]
            left_length, cx_left, cy_left = get_length(left_lmList[p1][1:], left_lmList[p2][1:])
        if len(self.hand_list["Right"]) != 0:
            right_lmList = self.hand_list["Right"][0]["lmList"]
            right_length, cx_right, cy_right = get_length(right_lmList[p1][1:], right_lmList[p2][1:])
        return left_length, right_length, img, [(cx_left, cy_left), (cx_right, cy_right)]


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = VirtualInputOnMP()
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.find_hands(img)
        lmList, img = detector.find_position(img)
        if len(lmList) != 0:
            left_fingers, right_fingers = detector.fingers_up()
            left_length, right_length, img, _ = detector.find_distance(8, 12, img)
        if len(lmList[HandType.RIGHT.value]) != 0:
            print("right venv position: ", lmList[HandType.RIGHT.value][0]["lmList"][4])
            print("right venv fingers: ", right_fingers)
            print("right venv length: ", right_length)
        if len(lmList[HandType.LEFT.value]) != 0:
            print("left venv position: ", lmList[HandType.LEFT.value][0]["lmList"][4])
            print("left venv fingers: ", left_fingers)
            print("left venv length: ", left_length)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
