"""เช็กกล้องก่อน calibrate: ดูภาพ threshold สด ปรับค่าด้วยแถบเลื่อน และดูว่า detect() เห็นอะไร

วิธีใช้
    py -3.12 camera_check.py                 # ใช้กล้องตัวที่ 0
    py -3.12 camera_check.py --cam 1         # ใช้กล้องตัวอื่น
    py -3.12 camera_check.py --image a.png   # ทดสอบกับรูปนิ่ง (ไม่ต้องมีกล้อง)

ปุ่ม
    m  ลองจับคู่บล็อกกับช่อง (เรียก solve จาก algo.py) แล้ววาดผลทับภาพ
    c  ล้างผลจับคู่
    s  บันทึกภาพดิบและภาพ threshold ไว้ทดสอบทีหลัง
    q / Esc  ออก

ฝั่งซ้ายของเส้นแบ่ง = พื้นที่บล็อก (บล็อกสีอ่อนบนพื้นเข้ม)
ฝั่งขวาของเส้นแบ่ง = กระดาน (ช่องเป็นรูมืด จึงกลับสีให้ช่องเป็นสีขาว)
"""
import argparse
import time
import cv2
import numpy as np

from algo import solve

WIN = "camera_check"
WIN_BW = "binary (left: blocks | right: holes)"
MIN_WIDTH_PX = 50        # บล็อกควรกว้างอย่างน้อยเท่านี้ในภาพ


def open_camera(index, width, height):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)   # DSHOW บน Windows ตั้งค่ากล้องได้มากกว่า
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"เปิดกล้อง {index} ไม่ได้")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # ปิด auto เพื่อไม่ให้ภาพสว่างขึ้นลงตอนแขนเข้ามาในภาพ (บางรุ่นไม่รองรับ จะไม่มีผล)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"กล้อง {index}: {w:.0f}x{h:.0f}  autofocus={cap.get(cv2.CAP_PROP_AUTOFOCUS):.0f}  "
          f"auto_exposure={cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):.2f}")
    return cap


def clear_border(bw):
    """ลบก้อนสีขาวที่แตะขอบภาพ (เช่น พื้นโต๊ะรอบกระดานที่กลายเป็นสีขาวหลังกลับสี)"""
    _, labels = cv2.connectedComponents(bw)
    edge = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    out = bw.copy()
    out[np.isin(labels, edge[edge > 0])] = 0
    return out


def binarize(gray, thresh, invert):
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, bw = cv2.threshold(gray, thresh, 255, mode)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))   # ลบจุดรบกวนเล็กๆ
    return clear_border(bw)


def process(frame, thresh, blur, split, hole_invert):
    """คืน (ภาพบล็อกขาวดำ, ภาพช่องขาวดำ, ตำแหน่งเส้นแบ่ง x, ความสว่างเฉลี่ย)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if blur > 0:
        gray = cv2.GaussianBlur(gray, (2 * blur + 1, 2 * blur + 1), 0)
    sx = int(np.clip(split, 5, 95) / 100 * gray.shape[1])
    bw_block = binarize(gray[:, :sx], thresh, invert=False)
    bw_hole = binarize(gray[:, sx:], thresh, invert=bool(hole_invert))
    return bw_block, bw_hole, sx, float(gray.mean())


def draw_objects(vis, bw, dx, color, label):
    """วาดเส้นขอบที่ detect() เห็น พร้อมขนาดในภาพ คืนจำนวนวัตถุและจำนวนที่เล็กเกินไป"""
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    n = small = 0
    for c in cnts:
        if cv2.contourArea(c) < 300:            # เกณฑ์เดียวกับ detect() ใน algo.py
            continue
        m = cv2.moments(c)
        cx, cy = int(m["m10"] / m["m00"]) + dx, int(m["m01"] / m["m00"])
        width = int(min(cv2.minAreaRect(c)[1]))   # ด้านแคบของกรอบที่หมุนตามรูป
        too_small = width < MIN_WIDTH_PX
        small += too_small
        cv2.drawContours(vis, [c + [dx, 0]], -1, color, 2)
        cv2.circle(vis, (cx, cy), 4, color, -1)
        cv2.putText(vis, f"{label}{n} {width}px", (cx - 40, cy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255) if too_small else color, 1)
        n += 1
    return n, small


def draw_matches(vis, result, sx):
    for r in result:
        bx, by = map(int, r["block"])
        hx, hy = int(r["hole"][0]) + sx, int(r["hole"][1])
        color = (0, 200, 255) if r["fits"] else (0, 0, 255)
        cv2.line(vis, (bx, by), (hx, hy), color, 2)
        cv2.putText(vis, f"rot {r['angle']:+.1f} IoU {r['iou']:.2f}", (bx - 60, by + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def put_status(vis, lines):
    y = 25
    for text, color in lines:
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        y += 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="หมายเลขกล้อง")
    ap.add_argument("--image", help="ใช้รูปนิ่งแทนกล้อง")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    cap = None
    still = None
    if args.image:
        still = cv2.imread(args.image)
        if still is None:
            raise SystemExit(f"เปิดรูป {args.image} ไม่ได้")
    else:
        cap = open_camera(args.cam, args.width, args.height)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_BW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("threshold", WIN, 127, 255, lambda v: None)
    cv2.createTrackbar("blur", WIN, 1, 10, lambda v: None)
    cv2.createTrackbar("split %", WIN, 50, 100, lambda v: None)
    cv2.createTrackbar("invert holes", WIN, 1, 1, lambda v: None)

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
        inv = cv2.getTrackbarPos("invert holes", WIN)
        bw_block, bw_hole, sx, bright = process(frame, thresh, blur, split, inv)

        # ความสว่างย้อนหลังประมาณ 2 วินาที ใช้ดูว่าแสงนิ่งไหม
        bright_hist = (bright_hist + [bright])[-60:]
        swing = max(bright_hist) - min(bright_hist)

        vis = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv2.line(vis, (sx, 0), (sx, vis.shape[0]), (255, 255, 0), 1)
        nb, small_b = draw_objects(vis, bw_block, 0, (0, 255, 0), "B")
        nh, small_h = draw_objects(vis, bw_hole, sx, (255, 120, 0), "H")
        if result is not None:
            draw_matches(vis, result, sx)

        ok_c, bad_c = (0, 255, 0), (0, 0, 255)
        put_status(vis, [
            (f"blocks: {nb}   holes: {nh}", ok_c if nb == nh and nb > 0 else bad_c),
            (f"too small (<{MIN_WIDTH_PX}px): {small_b + small_h}", ok_c if small_b + small_h == 0 else bad_c),
            (f"brightness {bright:.0f}  swing {swing:.1f}", ok_c if swing < 8 else bad_c),
            ("[m] match  [c] clear  [s] save  [q] quit", (255, 255, 255)),
        ])

        cv2.imshow(WIN, vis)
        cv2.imshow(WIN_BW, np.hstack([bw_block, np.full((bw_block.shape[0], 4), 128, np.uint8), bw_hole]))

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            result = None
        if key == ord("m"):
            if nb == 0 or nh == 0:
                print("ยังไม่เห็นบล็อกหรือช่อง ปรับ threshold / split ก่อน")
            else:
                t0 = time.time()
                result, _ = solve(bw_block, bw_hole)
                print(f"--- จับคู่ {len(result)} คู่ ใช้เวลา {time.time() - t0:.2f} วินาที ---")
                for r in result:
                    status = "ใส่ได้" if r["fits"] else "ใส่ไม่ได้"
                    print(f"บล็อก ({r['block'][0]:.0f}, {r['block'][1]:.0f}) -> "
                          f"ช่อง ({r['hole'][0] + sx:.0f}, {r['hole'][1]:.0f})  "
                          f"หมุน {r['angle']:+.1f}°  IoU={r['iou']:.3f}  {status}")
        if key == ord("s"):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"frame_{stamp}.png", frame)
            cv2.imwrite(f"frame_{stamp}_blocks.png", bw_block)
            cv2.imwrite(f"frame_{stamp}_holes.png", bw_hole)
            print(f"บันทึก frame_{stamp}*.png แล้ว (threshold={thresh}, blur={blur}, split={split}%)")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
