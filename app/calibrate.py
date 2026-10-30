"""Calibrate กล้องกับแขน MG400: หาสูตรแปลง "พิกเซลในภาพ -> มม. ของแขน" แล้วเซฟไว้ใน app/calib.json

วิธีใช้ (รันจากโฟลเดอร์โปรเจกต์)
    py -3.12 app/calibrate.py pose               # อ่าน x, y, z, r ตอนนี้ (กด Enter อ่านซ้ำ, q ออก)
    py -3.12 app/calibrate.py calibrate          # คลิกจุดในภาพ -> jog แขนไปแตะ -> เซฟ calib.json
    py -3.12 app/calibrate.py check              # คลิกในภาพแล้วดูว่าได้กี่ มม. (แขนไม่ขยับ)
    py -3.12 app/calibrate.py check --move       # เหมือนข้างบน แต่กด g ให้แขนไปลอยเหนือจุดที่คลิก

ขั้นตอน calibrate
    1. ติดสติกเกอร์/จุดเล็กๆ บนโต๊ะ 5-6 จุด กระจายทั่วพื้นที่ทำงาน (4 มุม + กลาง) ทั้งฝั่งบล็อกและฝั่งกระดาน
    2. รันโหมด calibrate กด space หยุดภาพ แล้วคลิกจุดตามลำดับ (u = ย้อนจุดล่าสุด) กด s ไปต่อ
    3. ลากแขน/jog ให้ปลายหัวจับแตะจุดที่ 1 แล้วกด Enter ทำจนครบทุกจุด (สคริปต์แค่อ่านค่า ไม่สั่งขยับ)
    4. ดูค่า error (มม.) ถ้าเกิน 3 มม. ควรทำใหม่

ห้ามขยับกล้อง แขน หรือเปลี่ยนความละเอียดภาพหลัง calibrate ถ้าขยับต้องทำใหม่
"""
import argparse
import time

import cv2
import numpy as np

from calib_utils import (CALIB_IMAGE, CALIB_PATH, Robot, ROBOT_IP, image_angle_to_r, in_workspace,
                         load_calib, open_camera, pixel_to_robot, save_calib)

WIN = "calibrate"
MIN_POINTS = 4
WARN_ERROR_MM = 3.0
HOVER_ABOVE = 50.0       # มม. เหนือโต๊ะตอนโหมด check --move


def draw_points(img, pts, color=(0, 0, 255)):
    for i, (u, v) in enumerate(pts, 1):
        cv2.circle(img, (int(u), int(v)), 6, color, -1)
        cv2.putText(img, str(i), (int(u) + 8, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(img, str(i), (int(u) + 8, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def put_text(img, text, y=30, color=(0, 255, 255)):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


# =============== โหมด pose ===============
def mode_pose(args):
    """อ่านตำแหน่งปัจจุบัน ใช้วัดความสูง Z ตอนหัวจับแตะผิวบนบล็อก / ผิวกระดาน"""
    robot = Robot(args.ip)
    try:
        while True:
            x, y, z, r = robot.pose()
            print(f"x={x:.1f}  y={y:.1f}  z={z:.1f}  r={r:.1f}")
            if input("Enter = อ่านอีกครั้ง, q = ออก: ").strip().lower() == "q":
                break
    finally:
        robot.close()


# =============== โหมด calibrate ===============
def pick_points(cap):
    """แสดงภาพสด กด space หยุดภาพ แล้วคลิกจุด คืน (ภาพที่หยุด, รายการจุดพิกเซล) หรือ None ถ้ากด q"""
    clicks = []
    state = {"frozen": None}

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and state["frozen"] is not None:
            clicks.append((x, y))

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)
    print("กด space หยุดภาพ -> คลิกจุดตามลำดับ -> s ไปต่อ  (u ย้อนจุด, space อีกครั้ง = กลับไปภาพสด, q ออก)")
    while True:
        if state["frozen"] is None:
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("อ่านภาพจากกล้องไม่ได้")
            view = frame.copy()
            put_text(view, "LIVE - press SPACE to freeze")
        else:
            view = state["frozen"].copy()
            draw_points(view, clicks)
            put_text(view, f"FROZEN - click points ({len(clicks)})  [u] undo  [s] next  [space] live")
        cv2.imshow(WIN, view)

        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            return None
        if k == ord(" "):
            state["frozen"] = None if state["frozen"] is not None else frame.copy()
            clicks.clear()
        if k == ord("u") and clicks:
            clicks.pop()
        if k == ord("s"):
            if state["frozen"] is None:
                print("กด space หยุดภาพก่อน")
            elif len(clicks) < MIN_POINTS:
                print(f"ต้องมีอย่างน้อย {MIN_POINTS} จุด (ตอนนี้ {len(clicks)})")
            else:
                return state["frozen"], list(clicks)


def mode_calibrate(args):
    cap = open_camera(args.cam, args.width, args.height, args.exposure)
    try:
        picked = pick_points(cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    if picked is None:
        print("ยกเลิก")
        return
    frame, clicks = picked

    # อ่านตำแหน่งแขนทีละจุด (ไม่สั่งขยับ คนลาก/jog เอง)
    robot = Robot(args.ip)
    robot_pts = []
    try:
        for i, (u, v) in enumerate(clicks, 1):
            input(f"\nย้ายปลายหัวจับไปแตะจุดที่ {i} (พิกเซล {u}, {v}) แล้วกด Enter ...")
            x, y, z, _ = robot.pose()
            robot_pts.append((x, y, z))
            print(f"  จุด {i}: x={x:.1f}  y={y:.1f}  z={z:.1f}")
    finally:
        robot.close()

    px = np.array(clicks, np.float64)
    rb = np.array(robot_pts, np.float64)
    H, _ = cv2.findHomography(px, rb[:, :2], 0)     # 0 = ใช้ทุกจุดเฉลี่ยกัน (least squares)
    if H is None:
        raise SystemExit("คำนวณ homography ไม่ได้ (จุดอาจอยู่ในแนวเส้นตรงเดียวกัน) ลองใหม่")

    proj = cv2.perspectiveTransform(px.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - rb[:, :2], axis=1)
    print("\nความคลาดเคลื่อนแต่ละจุด (มม.): " + ", ".join(f"{e:.2f}" for e in err))
    print(f"เฉลี่ย {err.mean():.2f} มม.  สูงสุด {err.max():.2f} มม.")
    if err.max() > WARN_ERROR_MM:
        print(f"คำเตือน: เกิน {WARN_ERROR_MM} มม. ลองแตะให้ตรงกลางจุดกว่านี้ หรือเพิ่มจำนวนจุด แล้ว calibrate ใหม่")
    if len(clicks) == 4:
        print("หมายเหตุ: 4 จุดพอดีเป๊ะเสมอ (error = 0) จึงวัดความแม่นไม่ได้ แนะนำ 5 จุดขึ้นไป หรือเช็กด้วยโหมด check")

    mn, mx = rb[:, :2].min(axis=0), rb[:, :2].max(axis=0)
    save_calib({
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cam": args.cam, "width": frame.shape[1], "height": frame.shape[0],
        "pixel_pts": px.tolist(),
        "robot_pts": rb.tolist(),
        "H": H.tolist(),
        "z_table": float(rb[:, 2].mean()),
        "error_mm": {"mean": float(err.mean()), "max": float(err.max()), "each": err.tolist()},
        "x_range": [float(mn[0]), float(mx[0])],
        "y_range": [float(mn[1]), float(mx[1])],
    })
    vis = frame.copy()
    draw_points(vis, clicks)
    cv2.imwrite(CALIB_IMAGE, vis)
    print(f"\nบันทึกแล้ว: {CALIB_PATH}")
    print(f"ภาพตอน calibrate: {CALIB_IMAGE}")
    print(f"ความสูงโต๊ะ z = {rb[:, 2].mean():.1f} มม.")
    print(f"พื้นที่ที่ calibrate ครอบคลุม: X {mn[0]:.1f} ถึง {mx[0]:.1f}   Y {mn[1]:.1f} ถึง {mx[1]:.1f}")


# =============== โหมด check ===============
def mode_check(args):
    calib = load_calib()
    H = calib["H"]
    hull = cv2.convexHull(np.array(calib["pixel_pts"], np.float32)).astype(np.int32)
    z_hover = calib["z_table"] + HOVER_ABOVE

    robot = None
    if args.move:
        robot = Robot(args.ip)
        robot.enable(args.speed)

    cap = open_camera(calib["cam"] if args.cam is None else args.cam,
                      calib["width"], calib["height"], args.exposure)
    target = {"px": None}

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            target["px"] = (x, y)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)
    print("คลิกในภาพเพื่อดูพิกัดแขน" + ("  กด g ให้แขนไปลอยเหนือจุด" if robot else "") + "  q ออก")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != (calib["width"], calib["height"]):
                raise SystemExit(f"ความละเอียดกล้อง {frame.shape[1]}x{frame.shape[0]} ไม่ตรงกับตอน "
                                 f"calibrate {calib['width']}x{calib['height']}")
            cv2.polylines(frame, [hull], True, (0, 255, 0), 1)     # กรอบพื้นที่ที่ calibrate
            draw_points(frame, calib["pixel_pts"], (0, 200, 0))
            put_text(frame, f"error mean {calib['error_mm']['mean']:.2f} mm  max {calib['error_mm']['max']:.2f} mm")
            if target["px"]:
                u, v = target["px"]
                x, y = pixel_to_robot(H, u, v)
                ok_ws = in_workspace(calib, x, y)
                color = (0, 255, 255) if ok_ws else (0, 0, 255)
                cv2.drawMarker(frame, (u, v), color, cv2.MARKER_CROSS, 20, 2)
                put_text(frame, f"pixel ({u}, {v}) -> X={x:.1f} Y={y:.1f} mm"
                         + ("" if ok_ws else "  OUTSIDE"), 60, color)
                # ลูกศรมุม +30 ในภาพ เทียบกับมุม r ที่แปลงได้ ใช้เช็กทิศการหมุน
                put_text(frame, f"image +30 deg -> r {image_angle_to_r(H, u, v, 30):+.1f} deg", 90, color)
            cv2.imshow(WIN, frame)

            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("g") and robot and target["px"]:
                x, y = pixel_to_robot(H, *target["px"])
                if not in_workspace(calib, x, y):
                    print(f"X={x:.1f} Y={y:.1f} อยู่นอกพื้นที่ที่ calibrate ไม่ขยับ")
                    continue
                cx, cy, cz, cr = robot.pose()
                robot.movl(cx, cy, max(cz, z_hover), cr)              # ยกขึ้นก่อน กันชน
                robot.movl(x, y, z_hover, cr)
                print(f"ไปที่ X={x:.1f} Y={y:.1f} Z={z_hover:.1f} แล้ว ดูว่าปลายหัวจับตรงจุดที่คลิกไหม")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if robot:
            robot.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["pose", "calibrate", "check"])
    ap.add_argument("--ip", default=ROBOT_IP)
    ap.add_argument("--cam", type=int, default=None, help="หมายเลขกล้อง (calibrate ค่าเริ่มต้น 1, check ใช้ค่าในไฟล์)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--exposure", type=float, help="ล็อกค่า exposure (เช่น -6)")
    ap.add_argument("--move", action="store_true", help="โหมด check: ให้แขนขยับจริงเมื่อกด g")
    ap.add_argument("--speed", type=int, default=20, help="ความเร็วแขน %% (โหมด check --move)")
    args = ap.parse_args()

    if args.mode == "pose":
        mode_pose(args)
    elif args.mode == "calibrate":
        if args.cam is None:
            args.cam = 1                  # ค่าเดียวกับ camera_check.py
        mode_calibrate(args)
    else:
        mode_check(args)


if __name__ == "__main__":
    main()
