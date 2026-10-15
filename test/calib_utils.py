"""ฟังก์ชันที่ใช้ร่วมกันระหว่าง calibrate และงานหยิบวาง (ไม่แก้ไฟล์เดิมในโฟลเดอร์แม่ แค่ import มาใช้)

    pixel_to_robot(H, u, v)            พิกเซลในภาพ -> (x, y) มม. ของแขน
    image_angle_to_r(H, u, v, angle)   มุมหมุนจาก algo (องศาในภาพ) -> มุมที่ต้องหมุนแกน r ของแขน
    Robot                              คุยกับ MG400 ผ่าน LAN (อ่านตำแหน่ง / สั่งขยับ / DO)
"""
import json
import os
import re
import socket
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)      # ให้ import algo.py / camera_check.py จากโฟลเดอร์แม่ได้

from camera_check import open_camera, process   # noqa: E402,F401  (ใช้ซ้ำ ไม่แก้)
from algo import solve                           # noqa: E402,F401

CALIB_PATH = os.path.join(HERE, "calib.json")
CALIB_IMAGE = os.path.join(HERE, "calib_frame.png")

ROBOT_IP = "192.168.1.6"
DASHBOARD_PORT = 29999
MOVE_PORT = 30003
SAFE_MARGIN = 10.0       # มม. ที่ยอมให้ออกนอกกรอบจุด calibrate ได้


# =============== ไฟล์ค่า calibrate ===============
def save_calib(data, path=CALIB_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_calib(path=CALIB_PATH):
    if not os.path.exists(path):
        raise SystemExit(f"ยังไม่มีไฟล์ {path} ให้รัน calibrate ก่อน")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["H"] = np.array(data["H"], dtype=np.float64)
    return data


# =============== แปลงพิกัด ===============
def pixel_to_robot(H, u, v):
    """พิกเซล (u, v) -> (x, y) มม. ของแขน"""
    p = cv2.perspectiveTransform(np.array([[[u, v]]], np.float64), H)[0, 0]
    return float(p[0]), float(p[1])


def robot_to_pixel(H, x, y):
    """(x, y) มม. -> พิกเซล ใช้วาดกรอบพื้นที่ทำงานบนภาพ"""
    p = cv2.perspectiveTransform(np.array([[[x, y]]], np.float64), np.linalg.inv(H))[0, 0]
    return float(p[0]), float(p[1])


def image_angle_to_r(H, u, v, angle_deg, length=30.0):
    """แปลงมุมหมุนในภาพ (จาก algo.solve, แกน y ชี้ลง) เป็นมุมหมุนในพิกัดแขน (องศา, ช่วง -180..180)

    วิธีคิด: วาดลูกศรสั้นๆ จากจุด (u, v) สองอัน คือลูกศรมุม 0 กับลูกศรที่หมุนไปตาม angle เหมือนที่ algo หมุนบล็อก
    แล้วแปลงปลายลูกศรทั้งสองไปเป็นพิกัดแขนด้วย H จากนั้นวัดว่าในพิกัดแขนลูกศรหมุนไปกี่องศา
    ถ้าภาพกลับด้านหรือแกนสลับ H จะจัดการเครื่องหมายให้เอง ไม่ต้องเดา
    """
    a = np.deg2rad(angle_deg)
    c = np.array(pixel_to_robot(H, u, v))
    p0 = np.array(pixel_to_robot(H, u + length, v)) - c
    p1 = np.array(pixel_to_robot(H, u + length * np.cos(a), v + length * np.sin(a))) - c
    d = np.degrees(np.arctan2(p1[1], p1[0]) - np.arctan2(p0[1], p0[0]))
    return float((d + 180) % 360 - 180)


def in_workspace(calib, x, y, margin=SAFE_MARGIN):
    (x0, x1), (y0, y1) = calib["x_range"], calib["y_range"]
    return x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin


# =============== แขน MG400 ===============
class Robot:
    """เชื่อมต่อ MG400 2 พอร์ต: dashboard (ตั้งค่า/อ่านค่า) และ move (สั่งขยับ)
    ต่างจาก dobot_api.py ตรงที่รอคำตอบจนครบ (ลงท้ายด้วย ;) และหยุดทันทีถ้าแขนตอบ error
    """

    def __init__(self, ip=ROBOT_IP):
        print(f"กำลังเชื่อมต่อแขน {ip} ...")
        self.dash = self._open(ip, DASHBOARD_PORT, 5)
        self.move = self._open(ip, MOVE_PORT, 60)    # Sync() อาจรอนาน

    @staticmethod
    def _open(ip, port, timeout):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((ip, port))
        return s

    @staticmethod
    def _txrx(sock, cmd):
        sock.sendall(cmd.encode())
        buf = b""
        while not buf.rstrip().endswith(b";"):
            chunk = sock.recv(1024)
            if not chunk:
                raise ConnectionError("แขนปิดการเชื่อมต่อ")
            buf += chunk
        resp = buf.decode().strip()
        print(f">> {cmd}   << {resp}")
        if int(resp.split(",")[0]) != 0:
            raise RuntimeError(f"แขนตอบ error: {resp}")
        return resp

    def d(self, cmd):
        return self._txrx(self.dash, cmd)

    def m(self, cmd):
        return self._txrx(self.move, cmd)

    def pose(self):
        """คืน (x, y, z, r) ตำแหน่งปลายแขนตอนนี้"""
        vals = re.search(r"\{(.*?)\}", self.d("GetPose()")).group(1).split(",")
        return tuple(float(v) for v in vals[:4])

    def enable(self, speed=20):
        self.d("ClearError()")
        self.d("EnableRobot()")
        for _ in range(30):                          # รอจนแขนพร้อม (RobotMode = 5)
            if "{5}" in self.d("RobotMode()"):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("แขนไม่เข้าสถานะพร้อมทำงาน")
        self.d(f"SpeedFactor({speed})")

    def movl(self, x, y, z, r):
        """ขยับเป็นเส้นตรงแล้วรอจนถึง"""
        self.m(f"MovL({x:.1f},{y:.1f},{z:.1f},{r:.1f})")
        self.m("Sync()")

    def close(self):
        for s in (self.dash, self.move):
            try:
                s.close()
            except Exception:
                pass
