"""หยิบบล็อกทีละชิ้นไปวางลงช่อง: กล้อง -> algo.solve() -> แปลงเป็น มม. ด้วย calib.json -> แขนหยิบ หมุน วาง

*** ร่างแรก ยังไม่เคยลองกับแขนจริง ***
ต้อง calibrate ให้ได้ test/calib.json ก่อน และกรอกค่าในส่วน "ค่าที่ต้องวัดเอง" ข้างล่างให้ครบ

วิธีใช้ (รันจากโฟลเดอร์โปรเจกต์)
    py -3.12 test/pick_place.py                    # โหมดซ้อม: แขนไม่ขยับ แค่วาดแผนบนภาพ + พิมพ์พิกัด
    py -3.12 test/pick_place.py --run              # ทำจริง: แขนหยิบวาง
    py -3.12 test/pick_place.py --run --no-grip    # แขนเคลื่อนตามแผนจริง แต่ไม่สั่งหัวดูด (ทดสอบท่าทาง)
    py -3.12 test/pick_place.py --thresh 142 --blur 4 --split 38    # ค่าเดียวกับที่ปรับใน camera_check_h.py

ภาพแบ่งบน/ล่างแบบ camera_check_h.py (บล็อกบน กระดานล่าง ถ้ากลับกันใส่ --blocks-bottom)
และใช้กรอบพื้นที่ทำงานจาก test/work_area.json (ตั้งด้วยปุ่ม a ใน camera_check_h.py)

ปุ่ม (กดในหน้าต่างภาพ)
    m  ถ่ายภาพแล้วคิดแผน (จับคู่ + มุม) วาดบนภาพ
    g  ทำตามแผน (โหมด --run เท่านั้น)
    c  ล้างแผน
    q  ออก
"""
import argparse
import time

import cv2
import numpy as np

from calib_utils import (Robot, ROBOT_IP, image_angle_to_r, in_workspace, load_calib, open_camera,
                         pixel_to_robot, solve)
from camera_check import draw_objects, put_status
from camera_check_h import area_mask, draw_matches_h, load_area, process_h

# ===== ค่าที่ต้องวัดเอง (ใช้ py -3.12 test/calibrate.py pose) =====
PICK_Z = -140.0            # z (มม.) ตอนหัวจับแตะผิวบนบล็อก
PLACE_Z = -119.2           # z (มม.) ตอนปล่อยบล็อกลงช่อง (ผิวบนกระดาน + เผื่อเล็กน้อย)
SAFE_Z = 6.1            # z (มม.) ที่เคลื่อนข้ามไปมาแล้วไม่ชนอะไร
CLEAR_POS = (316.5, -168.6, 24.6, -86.3)  # (x, y, z, r) จุดที่แขนไปรอ ไม่บังกล้องตอนถ่ายภาพ
GRIP_DO = 1              # ช่อง DO ที่สั่งหัวดูดลม (ต้องลองว่าช่องไหน)
GRIP_ON = 1              # ค่าที่ทำให้ "จับ" (บางหัวจับกลับกัน ใช้ 0)
GRIP_WAIT = 0.5          # วินาทีที่รอให้หัวจับหนีบ/ปล่อยเสร็จ
R_SIGN = 1               # ถ้าทดสอบแล้วแขนหมุนกลับทาง ให้เปลี่ยนเป็น -1
PICK_R = -70.0           # มุม r ตอนลงหยิบ (ตอนวางจะเป็น PICK_R + มุมที่คำนวณได้)
SPEED = 20               # % ความเร็ว เริ่มช้าๆ ก่อน
USE_GRIP = True          # --no-grip ตั้งเป็น False


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


def execute(robot, plan):
    """ทำตามแผนทีละคู่: เหนือบล็อก -> ลง -> จับ -> ขึ้น -> เหนือช่อง + หมุน -> ลง -> ปล่อย -> ขึ้น"""
    x, y, z, r = robot.pose()
    robot.movl(x, y, max(z, SAFE_Z), r)                      # ยกขึ้นก่อนเสมอ กันชน
    for i, p in enumerate([p for p in plan if p["skip"] is None], 1):
        (bx, by), (hx, hy) = p["pick"], p["place"]
        place_r = PICK_R + p["dr"]
        print(f"\n--- ชิ้นที่ {i}: หยิบ ({bx:.1f}, {by:.1f}) -> วาง ({hx:.1f}, {hy:.1f}) r={place_r:+.1f} ---")
        grip(robot, False)
        robot.movl(bx, by, SAFE_Z, PICK_R)
        robot.movl(bx, by, PICK_Z, PICK_R)
        grip(robot, True)
        robot.movl(bx, by, SAFE_Z, PICK_R)
        robot.movl(hx, hy, SAFE_Z, place_r)                  # หมุนระหว่างเคลื่อนที่ไปช่อง
        robot.movl(hx, hy, PLACE_Z, place_r)
        grip(robot, False)
        robot.movl(hx, hy, SAFE_Z, place_r)
    robot.movl(*CLEAR_POS)                                   # ถอยออกให้กล้องเห็นผล
    print("\nวางครบแล้ว")


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
    ap.add_argument("--thresh", type=int, default=142)
    ap.add_argument("--blur", type=int, default=4)
    ap.add_argument("--split", type=int, default=38, help="เส้นแบ่งบล็อก/กระดาน (%% ของความสูงภาพ นับจากบน)")
    ap.add_argument("--blocks-bottom", action="store_true", help="บล็อกอยู่ล่าง กระดานอยู่บน")
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
        robot.movl(*CLEAR_POS)                               # หลบกล้องก่อนถ่ายภาพ
    else:
        print("โหมดซ้อม: แขนจะไม่ขยับ (ใส่ --run เพื่อทำจริง)")

    area = load_area()
    if area is None:
        print("ยังไม่มี test/work_area.json จะดูทั้งภาพ (ตั้งกรอบด้วยปุ่ม a ใน camera_check_h.py)")
    mask = area_mask((calib["height"], calib["width"]), area) if area else None

    cap = open_camera(calib["cam"], calib["width"], calib["height"], args.exposure)
    cv2.namedWindow("pick_place", cv2.WINDOW_NORMAL)
    result, plan = None, None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            bw_b, bw_h, sy, dy_b, dy_h, _, _ = process_h(frame, args.thresh, args.blur, args.split,
                                                          not args.blocks_bottom, True, 1, 0, mask)
            vis = frame.copy()
            cv2.line(vis, (0, sy), (vis.shape[1], sy), (255, 255, 0), 1)
            if area:
                cv2.polylines(vis, [np.array(area, np.int32)], True, (255, 0, 255), 2)
            nb, _ = draw_objects(vis[dy_b:dy_b + bw_b.shape[0]], bw_b, 0, (0, 255, 0), "B")
            nh, _ = draw_objects(vis[dy_h:dy_h + bw_h.shape[0]], bw_h, 0, (255, 120, 0), "H")
            if result is not None:
                draw_matches_h(vis, result, dy_b, dy_h)
                draw_plan(vis, plan, H)
            put_status(vis, [(f"blocks: {nb}  holes: {nh}", (0, 255, 0) if nb == nh and nb else (0, 0, 255)),
                             ("RUN" if robot else "DRY RUN (arm will not move)", (0, 0, 255) if robot else (0, 255, 255)),
                             ("[m] plan  [g] go  [c] clear  [q] quit", (255, 255, 255))])
            cv2.imshow("pick_place", vis)

            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("c"):
                result = plan = None
            if k == ord("m"):
                if nb == 0 or nh == 0:
                    print("ยังไม่เห็นบล็อกหรือช่อง ปรับ --thresh / --split")
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
                    execute(robot, plan)
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
