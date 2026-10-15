"""เหมือน camera_check.py แต่เส้นแบ่งเป็นแนวนอน (บน / ล่าง) แทนซ้าย / ขวา

วิธีใช้ (รันจากโฟลเดอร์โปรเจกต์)
    python test/camera_check_h.py                 # กล้องตัวที่ 1
    python test/camera_check_h.py --cam 0
    python test/camera_check_h.py --image a.png   # ทดสอบกับรูปนิ่ง

แถบเลื่อน
    split %       ตำแหน่งเส้นแบ่ง (% ของความสูงภาพ นับจากบน)
    blocks on top 1 = บล็อกอยู่บน กระดานอยู่ล่าง   0 = กลับกัน
    ที่เหลือเหมือน camera_check.py

ปุ่ม  m จับคู่   c ล้าง   s บันทึกภาพ   q / Esc ออก
     a  ตั้งกรอบพื้นที่ทำงาน: คลิก 4 มุมของกระดาษขาว (u = ย้อน) ครบ 4 จุดแล้วเซฟลง test/work_area.json เอง
        นอกกรอบจะถูกมองเป็นพื้นขาว จึงไม่นับของเกะกะรอบโต๊ะ   x  ลบกรอบ (ดูทั้งภาพ)
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)      # ใช้ฟังก์ชันจาก camera_check.py / algo.py (ไม่แก้ไฟล์เดิม)

from camera_check import (MIN_WIDTH_PX, background_is_light, binarize, draw_objects,  # noqa: E402
                          open_camera, put_status)
from algo import solve                                                                # noqa: E402

WIN = "camera_check_h"
WIN_BW = "binary (blocks | holes)"
AREA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work_area.json")


def load_area():
    """คืนรายการ 4 มุม [(x, y), ...] หรือ None ถ้ายังไม่เคยตั้ง"""
    if not os.path.exists(AREA_PATH):
        return None
    with open(AREA_PATH, encoding="utf-8") as f:
        return [tuple(p) for p in json.load(f)["corners"]]


def save_area(corners, width, height):
    with open(AREA_PATH, "w", encoding="utf-8") as f:
        json.dump({"corners": [list(map(int, p)) for p in corners], "width": width, "height": height}, f, indent=2)


def area_mask(shape, corners):
    """ภาพขาวดำ: 255 = ในกรอบ, 0 = นอกกรอบ"""
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.array(corners, np.int32)], 255)
    return mask


def process_h(frame, thresh, blur, split, blocks_top, auto, block_invert, hole_invert, mask=None):
    """แบ่งภาพบน/ล่าง คืน (ภาพบล็อกขาวดำ, ภาพช่องขาวดำ, y เส้นแบ่ง, dy บล็อก, dy ช่อง, ความสว่าง, (กลับสีบล็อก, กลับสีช่อง))
    dy = ระยะที่ต้องบวกกับพิกัด y ในภาพย่อยเพื่อให้เป็นพิกัดในภาพเต็ม
    mask = กรอบพื้นที่ทำงาน (จาก area_mask) นอกกรอบจะถูกระบายเป็นสีขาวเหมือนพื้นกระดาษ"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if blur > 0:
        gray = cv2.GaussianBlur(gray, (2 * blur + 1, 2 * blur + 1), 0)
    if mask is not None:
        gray = np.where(mask > 0, gray, 255).astype(np.uint8)
    sy = int(np.clip(split, 5, 95) / 100 * gray.shape[0])
    top, bottom = gray[:sy], gray[sy:]
    if blocks_top:
        g_block, g_hole, dy_b, dy_h = top, bottom, 0, sy
    else:
        g_block, g_hole, dy_b, dy_h = bottom, top, sy, 0
    if auto:
        block_invert = background_is_light(g_block, thresh)
        hole_invert = not background_is_light(g_hole, thresh)
    block_invert, hole_invert = bool(block_invert), bool(hole_invert)
    bw_block = binarize(g_block, thresh, invert=block_invert)
    bw_hole = binarize(g_hole, thresh, invert=hole_invert)
    return bw_block, bw_hole, sy, dy_b, dy_h, float(gray.mean()), (block_invert, hole_invert)


def draw_matches_h(vis, result, dy_b, dy_h):
    for r in result:
        bx, by = int(r["block"][0]), int(r["block"][1]) + dy_b
        hx, hy = int(r["hole"][0]), int(r["hole"][1]) + dy_h
        color = (0, 200, 255) if r["fits"] else (0, 0, 255)
        cv2.line(vis, (bx, by), (hx, hy), color, 2)
        cv2.putText(vis, f"rot {r['angle']:+.1f} IoU {r['iou']:.2f}", (bx - 60, by + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=1, help="หมายเลขกล้อง")
    ap.add_argument("--image", help="ใช้รูปนิ่งแทนกล้อง")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--exposure", type=float, help="ล็อกค่า exposure (เช่น -6)")
    args = ap.parse_args()

    cap = None
    still = None
    if args.image:
        still = cv2.imread(args.image)
        if still is None:
            raise SystemExit(f"เปิดรูป {args.image} ไม่ได้")
    else:
        cap = open_camera(args.cam, args.width, args.height, args.exposure)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_BW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("threshold", WIN, 142, 255, lambda v: None)
    cv2.createTrackbar("blur", WIN, 4, 10, lambda v: None)
    cv2.createTrackbar("split %", WIN, 50, 100, lambda v: None)
    cv2.createTrackbar("blocks on top", WIN, 1, 1, lambda v: None)
    cv2.createTrackbar("auto invert", WIN, 1, 1, lambda v: None)
    cv2.createTrackbar("invert blocks", WIN, 1, 1, lambda v: None)   # ใช้เมื่อ auto invert = 0
    cv2.createTrackbar("invert holes", WIN, 0, 1, lambda v: None)

    area = load_area()
    picking = None                      # ระหว่างตั้งกรอบ: รายการจุดที่คลิก
    mask = None

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and picking is not None and len(picking) < 4:
            picking.append((x, y))

    cv2.setMouseCallback(WIN, on_mouse)

    result = None
    bright_hist = []
    while True:
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                print("อ่านภาพจากกล้องไม่ได้")
                break
        else:
            frame = still.copy()

        thresh = cv2.getTrackbarPos("threshold", WIN)
        blur = cv2.getTrackbarPos("blur", WIN)
        split = cv2.getTrackbarPos("split %", WIN)
        blocks_top = cv2.getTrackbarPos("blocks on top", WIN)
        auto = cv2.getTrackbarPos("auto invert", WIN)
        inv_b = cv2.getTrackbarPos("invert blocks", WIN)
        inv_h = cv2.getTrackbarPos("invert holes", WIN)
        if area and (mask is None or mask.shape != frame.shape[:2]):
            mask = area_mask(frame.shape, area)
        bw_block, bw_hole, sy, dy_b, dy_h, bright, (inv_b, inv_h) = process_h(
            frame, thresh, blur, split, blocks_top, auto, inv_b, inv_h, mask if area else None)

        bright_hist = (bright_hist + [bright])[-60:]
        swing = max(bright_hist) - min(bright_hist)

        vis = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv2.line(vis, (0, sy), (vis.shape[1], sy), (255, 255, 0), 1)
        # วาดลงบนภาพย่อย (view ของ vis) จึงไม่ต้องเลื่อนพิกัด y เอง
        nb, small_b = draw_objects(vis[dy_b:dy_b + bw_block.shape[0]], bw_block, 0, (0, 255, 0), "B")
        nh, small_h = draw_objects(vis[dy_h:dy_h + bw_hole.shape[0]], bw_hole, 0, (255, 120, 0), "H")
        if result is not None:
            draw_matches_h(vis, result, dy_b, dy_h)
        if area and picking is None:
            cv2.polylines(vis, [np.array(area, np.int32)], True, (255, 0, 255), 2)

        ok_c, bad_c = (0, 255, 0), (0, 0, 255)
        put_status(vis, [
            (f"blocks: {nb}   holes: {nh}", ok_c if nb == nh and nb > 0 else bad_c),
            (f"too small (<{MIN_WIDTH_PX}px): {small_b + small_h}", ok_c if small_b + small_h == 0 else bad_c),
            (f"brightness {bright:.0f}  swing {swing:.1f}", ok_c if swing < 8 else bad_c),
            (f"invert blocks={int(inv_b)} holes={int(inv_h)} ({'auto' if auto else 'manual'})  "
             f"blocks {'TOP' if blocks_top else 'BOTTOM'}", (255, 255, 255)),
            ("[m] match  [c] clear  [s] save  [a] area  [x] no area  [q] quit", (255, 255, 255)),
        ])
        if picking is not None:
            for i, p in enumerate(picking, 1):
                cv2.circle(vis, p, 6, (255, 0, 255), -1)
                cv2.putText(vis, str(i), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            if len(picking) > 1:
                cv2.polylines(vis, [np.array(picking, np.int32)], False, (255, 0, 255), 2)
            put_status(vis[vis.shape[0] - 40:], [(f"AREA: click 4 corners of the paper ({len(picking)}/4)  "
                                                  "[u] undo  [a] cancel", (255, 0, 255))])

        cv2.imshow(WIN, vis)
        cv2.imshow(WIN_BW, np.vstack([bw_block, np.full((4, bw_block.shape[1]), 128, np.uint8), bw_hole]))

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            result = None
        if key == ord("a"):
            picking = [] if picking is None else None      # กด a อีกครั้ง = ยกเลิก
        if key == ord("u") and picking:
            picking.pop()
        if picking is not None and len(picking) == 4:
            area, mask, picking, result = list(picking), None, None, None
            save_area(area, frame.shape[1], frame.shape[0])
            print(f"บันทึกกรอบพื้นที่ทำงานแล้ว: {AREA_PATH}  มุม = {area}")
        if key == ord("x"):
            area, mask, result = None, None, None
            print("ไม่ใช้กรอบแล้ว (ไฟล์ work_area.json ยังอยู่ เปิดโปรแกรมใหม่จะโหลดกลับมา)")
        if key == ord("m"):
            if nb == 0 or nh == 0:
                print("ยังไม่เห็นบล็อกหรือช่อง ปรับ threshold / split ก่อน")
            else:
                t0 = time.time()
                result, _ = solve(bw_block, bw_hole)
                print(f"--- จับคู่ {len(result)} คู่ ใช้เวลา {time.time() - t0:.2f} วินาที ---")
                for r in result:
                    status = "ใส่ได้" if r["fits"] else "ใส่ไม่ได้"
                    print(f"บล็อก ({r['block'][0]:.0f}, {r['block'][1] + dy_b:.0f}) -> "
                          f"ช่อง ({r['hole'][0]:.0f}, {r['hole'][1] + dy_h:.0f})  "
                          f"หมุน {r['angle']:+.1f}°  IoU={r['iou']:.3f}  {status}")
        if key == ord("s"):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"frame_{stamp}.png", frame)
            cv2.imwrite(f"frame_{stamp}_blocks.png", bw_block)
            cv2.imwrite(f"frame_{stamp}_holes.png", bw_hole)
            print(f"บันทึก frame_{stamp}*.png แล้ว (threshold={thresh}, blur={blur}, split={split}%, "
                  f"blocks_top={blocks_top})")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
