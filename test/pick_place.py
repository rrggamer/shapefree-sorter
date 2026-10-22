"""หยิบบล็อกทีละชิ้นไปวางลงช่อง: กล้อง -> algo.solve() -> แปลงเป็น มม. ด้วย calib.json -> แขนหยิบ หมุน วาง

*** ร่างแรก ยังไม่เคยลองกับแขนจริง ***
ต้อง calibrate ให้ได้ test/calib.json ก่อน และกรอกค่าในส่วน "ค่าที่ต้องวัดเอง" ข้างล่างให้ครบ

วิธีใช้ (รันจากโฟลเดอร์โปรเจกต์)
    py -3.12 test/pick_place.py                    # โหมดซ้อม: แขนไม่ขยับ แค่วาดแผนบนภาพ + พิมพ์พิกัด
    py -3.12 test/pick_place.py --run              # ทำจริง: แขนหยิบวาง
    py -3.12 test/pick_place.py --run --no-grip    # แขนเคลื่อนตามแผนจริง แต่ไม่สั่งหัวดูด (ทดสอบท่าทาง)
    py -3.12 test/pick_place.py --thresh 78 --blur 4    # ค่าเดียวกับที่ปรับใน camera_check_h.py

ไม่ใช้เส้นแบ่งแล้ว: วางบล็อกกับกระดานตรงไหนก็ได้ในกรอบพื้นที่ทำงาน (test/work_area.json ตั้งด้วยปุ่ม a ใน camera_check_h.py)
    กระดาน = ก้อนสีเข้มที่ใหญ่ที่สุด, รู = ส่วนสีอ่อนข้างในกระดาน, บล็อก = ก้อนสีเข้มอื่นๆ
    วางบล็อกห่างจากกระดานนิดหน่อย ถ้าแตะกันจะถูกนับรวมเป็นกระดาน

ปุ่ม (กดในหน้าต่างภาพ)
    m  ถ่ายภาพแล้วคิดแผน (จับคู่ + มุม) วาดบนภาพ
    g  เริ่มหยิบวาง (โหมด --run เท่านั้น) ถ่ายภาพใหม่ + คิดแผนใหม่ก่อนหยิบทุกชิ้น
       บล็อกที่ถูกเลื่อนหรือหยิบพลาดจะถูกหาใหม่เอง  (--no-grip ทำตามแผนจากปุ่ม m รอบเดียว)
    c  ล้างแผน
    q  ออก
"""
import argparse
import time

import cv2
import numpy as np

from calib_utils import (Robot, ROBOT_IP, image_angle_to_r, in_workspace, load_calib, open_camera,
                         pixel_to_robot, solve)
from camera_check import binarize, draw_objects, put_status
from camera_check_h import area_mask, draw_matches_h, load_area

# ===== ค่าที่ต้องวัดเอง (ใช้ py -3.12 test/calibrate.py pose) =====
PICK_Z = -143.0            # z (มม.) ตอนหัวจับแตะผิวบนบล็อก
PLACE_Z = -132.2           # z (มม.) ตอนปล่อยบล็อกลงช่อง (ผิวบนกระดาน + เผื่อเล็กน้อย)
SAFE_Z = 6.1            # z (มม.) ที่เคลื่อนข้ามไปมาแล้วไม่ชนอะไร
CLEAR_POS = (316.5, -168.6, 24.6, -86.3)  # (x, y, z, r) จุดที่แขนไปรอ ไม่บังกล้องตอนถ่ายภาพ
GRIP_DO = 1              # ช่อง DO ที่สั่งหัวดูดลม (ต้องลองว่าช่องไหน)
GRIP_ON = 1              # ค่าที่ทำให้ "จับ" (บางหัวจับกลับกัน ใช้ 0)
GRIP_WAIT = 0.5          # วินาทีที่รอให้หัวจับหนีบ/ปล่อยเสร็จ
R_SIGN = 1               # ถ้าทดสอบแล้วแขนหมุนกลับทาง ให้เปลี่ยนเป็น -1
PICK_R = -70.0           # มุม r ตอนลงหยิบ (ตอนวางจะเป็น PICK_R + มุมที่คำนวณได้)
SPEED = 20               # % ความเร็ว เริ่มช้าๆ ก่อน
USE_GRIP = True          # --no-grip ตั้งเป็น False
COLLISION_LEVEL = 3      # ความไวตรวจชน 0 = ปิด, 1-5 (เลขมาก = ไวมาก ชนเบาๆ ก็หยุด)
MOVE_TIMEOUT = 20.0      # วินาที ถ้าแขนยังไปไม่ถึงจุดหมายถือว่ามีปัญหา หยุดทำงาน
RETRY_STEP = 1.0         # มม. ถ้าหยิบไม่ติด รอบถัดไปลง PICK_Z ต่ำกว่าเดิมเท่านี้
MAX_RETRY = 6            # แต่ละชิ้นหยิบพลาดได้กี่ครั้ง (ลงต่ำสุด MAX_RETRY x RETRY_STEP มม. จาก PICK_Z)
LEFT_MM = 15.0           # ถ้ายังเห็นบล็อกห่างจากจุดหยิบไม่เกินเท่านี้ (มม.) ถือว่ายังหยิบไม่ไป


class Collision(Exception):
    """แขนหยุดเพราะชน หรือขึ้น error ระหว่างเคลื่อนที่"""


def move(robot, x, y, z, r):
    """ขยับเป็นเส้นตรง แล้วคอยเช็กทุก 0.1 วินาทีว่าชน/error ไหม (แทน Sync ที่อาจค้างตอนแขนหยุดเพราะชน)"""
    robot.m(f"MovL({x:.1f},{y:.1f},{z:.1f},{r:.1f})")
    t0 = time.time()
    while time.time() - t0 < MOVE_TIMEOUT:
        time.sleep(0.1)
        if "{9}" in robot.d("RobotMode()"):                  # 9 = error (รวมถึงชน)
            raise Collision(f"แขนหยุดระหว่างไป ({x:.1f}, {y:.1f}, {z:.1f}) ErrorID: {robot.d('GetErrorID()')}")
        px, py, pz, pr = robot.pose()
        if max(abs(px - x), abs(py - y), abs(pz - z)) < 0.5 and abs((pr - r + 180) % 360 - 180) < 0.5:
            return
    raise Collision(f"ไปไม่ถึง ({x:.1f}, {y:.1f}, {z:.1f}) ภายใน {MOVE_TIMEOUT:.0f} วินาที")


def find_objects(frame, thresh, blur, mask=None):
    """แยกบล็อกกับรูโดยไม่ใช้เส้นแบ่ง คืน (ภาพขาวดำบล็อก, ภาพขาวดำรู, เส้นขอบกระดานหรือ None) ขนาดเท่าภาพเต็ม
    กระดาน = ก้อนสีเข้มที่ใหญ่ที่สุด  รู = ส่วนสีอ่อนที่อยู่ข้างในกระดาน  บล็อก = ก้อนสีเข้มที่เหลือ"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if blur > 0:
        gray = cv2.GaussianBlur(gray, (2 * blur + 1, 2 * blur + 1), 0)
    if mask is not None:
        gray = np.where(mask > 0, gray, 255).astype(np.uint8)      # นอกกรอบ = สีขาวเหมือนกระดาษ
    dark = binarize(gray, thresh, invert=True)
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= 300]
    bw_b = np.zeros_like(dark)
    bw_h = np.zeros_like(dark)
    if not cnts:
        return bw_b, bw_h, None
    board = max(cnts, key=cv2.contourArea)
    for c in cnts:
        if c is not board:
            cv2.drawContours(bw_b, [c], -1, 255, -1)
    # ขอบกระดานตรงรูริมๆ บางมาก อาจขาดจนรูหลุดออกนอกเส้นขอบ ใช้สี่เหลี่ยม (หมุนได้) ที่ครอบกระดานแทน
    board = cv2.boxPoints(cv2.minAreaRect(board)).astype(np.int32).reshape(-1, 1, 2)
    inside = np.zeros_like(dark)
    cv2.drawContours(inside, [board], -1, 255, -1)
    inside = cv2.erode(inside, np.ones((7, 7), np.uint8))           # ตัดขอบกระดานออก กันขอบกลายเป็นรู
    light = binarize(gray, thresh, invert=False)
    bw_h = cv2.bitwise_and(light, inside)
    return bw_b, bw_h, board


def make_plan(result, dy_b, dy_h, H, calib):
    """แปลงผลจาก solve() (พิกเซล) เป็นแผนของแขน (มม. + มุม r) และเรียงลำดับหยิบ"""
    plan = []
    for r in result:
        bu, bv = r["block"][0], r["block"][1] + dy_b      # พิกัด y นับจากภาพย่อย ต้องบวกกลับเป็นภาพเต็ม
        hu, hv = r["hole"][0], r["hole"][1] + dy_h
        bx, by = pixel_to_robot(H, bu, bv)
        hx, hy = pixel_to_robot(H, hu, hv)
        dr = R_SIGN * image_angle_to_r(H, bu, bv, r["angle"])
        step = {"pick": (bx, by), "place": (hx, hy), "dr": dr, "angle_img": r["angle"],
                "iou": r["iou"], "fits": r["fits"], "skip": None}
        if not r["fits"]:
            step["skip"] = "ใส่ไม่ได้ (IoU ต่ำ)"
        elif not in_workspace(calib, bx, by) or not in_workspace(calib, hx, hy):
            step["skip"] = "อยู่นอกพื้นที่ที่ calibrate"
        plan.append(step)
    return order_plan(plan)


def order_plan(plan, start=(0.0, 0.0)):
    """เรียงลำดับหยิบแบบง่าย: เลือกบล็อกที่ใกล้ปลายแขนที่สุดก่อน (จุดตั้งต้นคือจุดวางชิ้นก่อนหน้า)
    TODO (คน B): ลองวิธีอื่นที่ทำให้ระยะรวมสั้นกว่านี้ แล้ววิเคราะห์ Big-O
    """
    todo = [p for p in plan if p["skip"] is None]
    done = []
    pos = np.array(CLEAR_POS[:2] if CLEAR_POS else start)
    while todo:
        k = int(np.argmin([np.hypot(*(np.array(p["pick"]) - pos)) for p in todo]))
        step = todo.pop(k)
        done.append(step)
        pos = np.array(step["place"])
    return done + [p for p in plan if p["skip"] is not None]


def print_plan(plan):
    print("\n===== แผน =====")
    for i, p in enumerate(plan, 1):
        head = f"{i}. หยิบ X={p['pick'][0]:.1f} Y={p['pick'][1]:.1f} -> วาง X={p['place'][0]:.1f} " \
               f"Y={p['place'][1]:.1f}  หมุน r {p['dr']:+.1f}° (ในภาพ {p['angle_img']:+.1f}°)  IoU={p['iou']:.2f}"
        print(head + (f"  [ข้าม: {p['skip']}]" if p["skip"] else ""))


def check_config(calib=None):
    missing = [n for n, v in (("PICK_Z", PICK_Z), ("PLACE_Z", PLACE_Z), ("SAFE_Z", SAFE_Z),
                              ("CLEAR_POS", CLEAR_POS)) if v is None]
    if missing:
        raise SystemExit("ยังไม่ได้กรอกค่า: " + ", ".join(missing) + " (แก้ที่หัวไฟล์ test/pick_place.py)")
    if not (PICK_Z < SAFE_Z and PLACE_Z < SAFE_Z):
        raise SystemExit("SAFE_Z ต้องสูงกว่า PICK_Z และ PLACE_Z")
    if calib is not None:
        low = [n for n, v in (("PICK_Z", PICK_Z), ("PLACE_Z", PLACE_Z)) if v <= calib["z_table"]]
        if low:
            print("คำเตือน: " + ", ".join(low) + f" ต่ำกว่าหรือเท่ากับ z ตอน calibrate ({calib['z_table']:.1f}) "
                  "ถ้ามั่นใจว่าค่าถูกก็ทำต่อได้ ระวังชน")


def grip(robot, on):
    if not USE_GRIP:
        print(f"(--no-grip: ข้ามการ{'ดูด' if on else 'ปล่อย'})")
        return
    robot.d(f"DO({GRIP_DO},{GRIP_ON if on else 1 - GRIP_ON})")
    time.sleep(GRIP_WAIT)


def execute(robot, plan, pick_z=None):
    """ทำตามแผนทีละคู่: เหนือบล็อก -> ลง -> จับ -> ขึ้น -> เหนือช่อง + หมุน -> ลง -> ปล่อย -> ขึ้น"""
    pick_z = PICK_Z if pick_z is None else pick_z
    x, y, z, r = robot.pose()
    move(robot, x, y, max(z, SAFE_Z), r)                      # ยกขึ้นก่อนเสมอ กันชน
    for i, p in enumerate([p for p in plan if p["skip"] is None], 1):
        (bx, by), (hx, hy) = p["pick"], p["place"]
        place_r = PICK_R + p["dr"]
        print(f"\n--- ชิ้นที่ {i}: หยิบ ({bx:.1f}, {by:.1f}) -> วาง ({hx:.1f}, {hy:.1f}) r={place_r:+.1f} ---")
        grip(robot, False)
        move(robot, bx, by, SAFE_Z, PICK_R)
        move(robot, bx, by, pick_z, PICK_R)
        grip(robot, True)
        move(robot, bx, by, SAFE_Z, PICK_R)
        move(robot, hx, hy, SAFE_Z, place_r)                  # หมุนระหว่างเคลื่อนที่ไปช่อง
        move(robot, hx, hy, PLACE_Z, place_r)
        grip(robot, False)
        move(robot, hx, hy, SAFE_Z, place_r)
    move(robot, *CLEAR_POS)                                   # ถอยออกให้กล้องเห็นผล
    print("\nทำครบทุกชิ้นในแผนแล้ว")


def near(pt, pts, mm=LEFT_MM):
    """มีจุดใน pts ที่ห่างจาก pt ไม่เกิน mm ไหม"""
    return any(np.hypot(x - pt[0], y - pt[1]) < mm for x, y in pts)


def remove_used_holes(bw_h, dy_h, used, H):
    """ลบรูที่วางบล็อกไปแล้วออกจากภาพขาวดำของฝั่งรู จะได้ไม่เอาไปจับคู่ซ้ำ"""
    bw_h = bw_h.copy()
    cnts, _ = cv2.findContours(bw_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        if near(pixel_to_robot(H, m["m10"] / m["m00"], m["m01"] / m["m00"] + dy_h), used):
            cv2.drawContours(bw_h, [c], -1, 0, -1)
    return bw_h


def run_closed_loop(robot, scan, H, calib):
    """ถ่ายภาพใหม่ก่อนหยิบทุกชิ้น: ถ่าย -> คิดแผนใหม่ -> หยิบวาง 1 ชิ้น -> ถอยหลบกล้อง -> วนจนหมด
    บล็อกที่ถูกเลื่อน หรือหยิบพลาด จะถูกเห็นในภาพรอบถัดไปเอง  รูที่วางไปแล้วจะถูกจำไว้ไม่ใช้ซ้ำ"""
    used = []                        # ตำแหน่ง (มม.) ของรูที่วางบล็อกไปแล้ว
    last = None                      # ขั้นที่เพิ่งทำ ใช้เช็กว่าหยิบติดไหม
    fails = 0                        # ชิ้นที่กำลังหยิบ พลาดไปแล้วกี่ครั้ง (เริ่มนับใหม่ทุกชิ้น)
    while True:
        bw_b, bw_h, dy_b, dy_h, block_pts = scan()
        if last is not None:
            if near(last["pick"], block_pts):
                fails += 1
                if fails > MAX_RETRY:
                    print(f"ชิ้นนี้หยิบพลาดครบ {MAX_RETRY} ครั้งแล้ว หยุดก่อน ลองเช็ก PICK_Z / หัวดูด")
                    return
                print(f"ยังเห็นบล็อกที่จุดหยิบเดิม = หยิบไม่ติด ลองชิ้นเดิมใหม่ด้วย "
                      f"PICK_Z = {PICK_Z - fails * RETRY_STEP:.1f} (ครั้งที่ {fails}/{MAX_RETRY})")
            else:
                fails = 0
                used.append(last["place"])
        if not block_pts:
            print(f"\nไม่เหลือบล็อกแล้ว วางไป {len(used)} ชิ้น")
            return
        bw_h = remove_used_holes(bw_h, dy_h, used, H)
        result, _ = solve(bw_b, bw_h)
        steps = [p for p in make_plan(result, dy_b, dy_h, H, calib) if p["skip"] is None]
        if not steps:
            print(f"\nเหลือบล็อก {len(block_pts)} ชิ้น แต่ไม่มีรูที่ใส่ได้ หยุด")
            return
        step = steps[0]
        if fails:                                            # หยิบพลาด: ลองชิ้นเดิม (ที่อยู่ใกล้จุดเดิม) ต่ำลงกว่าเดิม
            same = [p for p in steps if near(last["pick"], [p["pick"]])]
            if same:
                step = same[0]
            else:                                            # หาชิ้นเดิมไม่เจอในแผน = ไปชิ้นใหม่ เริ่ม Z เดิม
                fails = 0
        execute(robot, [step], PICK_Z - fails * RETRY_STEP)  # ชิ้นใหม่เริ่มที่ PICK_Z เสมอ
        last = step


def draw_plan(vis, plan, H):
    """เขียนลำดับและพิกัด มม. ไว้ข้างบล็อกแต่ละชิ้น"""
    Hinv = np.linalg.inv(H)
    for i, p in enumerate(plan, 1):
        u, v = cv2.perspectiveTransform(np.array([[p["pick"]]], np.float64), Hinv)[0, 0]
        color = (0, 0, 255) if p["skip"] else (255, 255, 255)
        text = f"#{i} X{p['pick'][0]:.0f} Y{p['pick'][1]:.0f} r{p['dr']:+.0f}" if not p["skip"] else "SKIP"
        cv2.putText(vis, text, (int(u) - 60, int(v) + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(vis, text, (int(u) - 60, int(v) + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="ให้แขนขยับจริง (ไม่ใส่ = ซ้อม)")
    ap.add_argument("--ip", default=ROBOT_IP)
    ap.add_argument("--thresh", type=int, default=78)
    ap.add_argument("--blur", type=int, default=4)
    ap.add_argument("--exposure", type=float)
    ap.add_argument("--no-grip", action="store_true", help="ไม่สั่งหัวดูด แขนเคลื่อนอย่างเดียว")
    args = ap.parse_args()
    global USE_GRIP
    USE_GRIP = not args.no_grip

    calib = load_calib()
    H = calib["H"]
    robot = None
    if args.run:
        check_config(calib)
        robot = Robot(args.ip)
        robot.enable(SPEED)
        robot.d(f"SetCollisionLevel({COLLISION_LEVEL})")       # เปิดตรวจชน
        move(robot, *CLEAR_POS)                              # หลบกล้องก่อนถ่ายภาพ
    else:
        print("โหมดซ้อม: แขนจะไม่ขยับ (ใส่ --run เพื่อทำจริง)")

    area = load_area()
    if area is None:
        print("ยังไม่มี test/work_area.json จะดูทั้งภาพ (ตั้งกรอบด้วยปุ่ม a ใน camera_check_h.py)")
    mask = area_mask((calib["height"], calib["width"]), area) if area else None

    cap = open_camera(calib["cam"], calib["width"], calib["height"], args.exposure)
    cv2.namedWindow("pick_place", cv2.WINDOW_NORMAL)

    def scan():
        """ถ่ายภาพใหม่ คืน (ภาพบล็อก, ภาพรู, dy บล็อก, dy รู, ตำแหน่ง มม. ของบล็อกที่เห็น)"""
        for _ in range(5):                                   # ทิ้งภาพเก่าที่ค้างในกล้อง
            cap.read()
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("อ่านภาพจากกล้องไม่ได้")
        vis = frame.copy()                                   # อัปเดตหน้าต่างระหว่างทำงาน จะได้ไม่ดูเหมือนค้าง
        put_status(vis, [("RUNNING - new photo before every pick", (0, 0, 255))])
        cv2.imshow("pick_place", vis)
        cv2.waitKey(1)
        bw_b, bw_h, _ = find_objects(frame, args.thresh, args.blur, mask)
        dy_b = dy_h = 0                                      # ไม่มีเส้นแบ่งแล้ว ภาพบล็อก/รูเป็นภาพเต็ม
        cnts, _ = cv2.findContours(bw_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = []
        for c in cnts:
            if cv2.contourArea(c) < 300:                     # เกณฑ์เดียวกับ draw_objects
                continue
            m = cv2.moments(c)
            pts.append(pixel_to_robot(H, m["m10"] / m["m00"], m["m01"] / m["m00"] + dy_b))
        return bw_b, bw_h, dy_b, dy_h, pts

    result, plan = None, None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            bw_b, bw_h, board = find_objects(frame, args.thresh, args.blur, mask)
            dy_b = dy_h = 0
            vis = frame.copy()
            if board is not None:
                cv2.drawContours(vis, [board], -1, (255, 255, 0), 2)         # กรอบกระดาน (ฟ้า)
            if area:
                cv2.polylines(vis, [np.array(area, np.int32)], True, (255, 0, 255), 2)
            nb, _ = draw_objects(vis[dy_b:dy_b + bw_b.shape[0]], bw_b, 0, (0, 255, 0), "B")
            nh, _ = draw_objects(vis[dy_h:dy_h + bw_h.shape[0]], bw_h, 0, (255, 120, 0), "H")
            if result is not None:
                draw_matches_h(vis, result, dy_b, dy_h)
                draw_plan(vis, plan, H)
            status = [(f"blocks: {nb}  holes: {nh}", (0, 255, 0) if nb == nh and nb else (0, 0, 255)),
                      ("RUN" if robot else "DRY RUN (arm will not move)", (0, 0, 255) if robot else (0, 255, 255)),
                      ("[m] plan  [g] go  [c] clear  [q] quit", (255, 255, 255))]
            put_status(vis, status)
            cv2.imshow("pick_place", vis)

            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("c"):
                result = plan = None
            if k == ord("m"):
                if nb == 0 or nh == 0:
                    print("ยังไม่เห็นบล็อกหรือช่อง ปรับ --thresh")
                    continue
                result, _ = solve(bw_b, bw_h)
                plan = make_plan(result, dy_b, dy_h, H, calib)
                print_plan(plan)
            if k == ord("g"):
                if plan is None:
                    print("กด m คิดแผนก่อน")
                elif robot is None:
                    print("โหมดซ้อม: ไม่ขยับแขน (รันใหม่ด้วย --run)")
                else:
                    try:
                        if USE_GRIP:
                            run_closed_loop(robot, scan, H, calib)
                        else:                                # --no-grip บล็อกไม่หายไป ทำตามแผนรอบเดียวพอ
                            execute(robot, plan)
                    except Collision as e:
                        print(f"\n!!! หยุดแล้ว: {e}")
                        print("ดูว่าชนอะไร ย้ายของออก แล้วรันใหม่ (สคริปต์จะ ClearError ให้ตอนเริ่ม)")
                        try:
                            grip(robot, False)                   # ปิดหัวดูด
                        except RuntimeError:
                            pass
                        break
                    result = plan = None
                    for _ in range(5):                       # ทิ้งภาพเก่าที่ค้างในกล้อง
                        cap.read()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if robot:
            robot.close()


if __name__ == "__main__":
    main()
