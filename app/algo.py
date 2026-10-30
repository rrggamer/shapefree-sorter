import time
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# ===== ปรับค่าการทำงานได้จากส่วนนี้ =====
CANVAS = 300             # ขนาดภาพที่ใช้เทียบรูป
NORM_AREA = 10000.0      # พื้นที่มาตรฐานของรูปก่อนนำมาเปรียบเทียบ
COARSE_STEP = 4.0        # ระยะค้นหามุมตอนจับคู่ (องศา; ค่ายิ่งน้อยยิ่งละเอียด)
ANGLE_STEP = 2.0         # ระยะค้นหามุมตอนคำนวณการหมุน (องศา)
BAND = 0.01              # ช่วงคะแนน IoU ที่ถือว่าใกล้คะแนนสูงสุด
MIN_FIT = 0.90           # คะแนน IoU ขั้นต่ำที่ถือว่าใส่ลงช่องได้

# ค่าตัวอย่างและไฟล์ผลลัพธ์
SCENE_SIZE = 800         # ขนาดภาพฉากจำลอง
BLOCK_SCALE = 60         # ขนาดบล็อกในฉากจำลอง
HOLE_SCALE = 1.04        # ขนาดช่องเทียบกับบล็อก
EVALUATION_SCENES = 20   # จำนวนฉากที่ใช้ประเมิน (ตามเกณฑ์สเตป 1 ใช้ 100)
DEMO_IMAGE_PATH = "demo_step1.png"

BASIC = ["circle", "square", "hexagon", "trapezoid"]   # 4 รูปที่ใช้จริง
UNSEEN = ["star", "heart"]                             # รูปที่อัลกอไม่เคยรู้จักมาก่อน


def _poly(n, r=1.0):
    return np.array([[r * np.cos(2 * np.pi * k / n), r * np.sin(2 * np.pi * k / n)] for k in range(n)])


def _star(n=5, r_out=1.0, r_in=0.45):
    a = np.pi * np.arange(2 * n) / n - np.pi / 2
    r = np.where(np.arange(2 * n) % 2 == 0, r_out, r_in)
    return np.c_[r * np.cos(a), r * np.sin(a)]


def _heart(n=120):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))   # แกน y ของภาพชี้ลง
    p = np.c_[x, y]
    return (p - p.mean(0)) / 16


CANON = {
    "circle": _poly(72, 1.0),
    "square": np.array([[-.85, -.85], [.85, -.85], [.85, .85], [-.85, .85]]),
    "hexagon": _poly(6, 1.0),
    "trapezoid": np.array([[-1.0, -.55], [1.0, -.55], [.55, .55], [-.55, .55]]),
    "star": _star(),
    "heart": _heart(),
}
# มุมสมมาตรของแต่ละรูป ใช้ตอนตรวจคำตอบเท่านั้น (อัลกอไม่ได้ใช้ค่านี้)
PERIOD = {"circle": 1, "square": 90, "hexagon": 60, "trapezoid": 360, "star": 72, "heart": 360}


def draw_shape(img, name, center, angle, scale):
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    p = CANON[name] * scale @ R.T + center
    cv2.fillPoly(img, [np.round(p * 16).astype(np.int32)], 255, shift=4)


def _cells(n, size):
    """แบ่งฉากเป็นตาราง แล้วคืนจุดกลางของแต่ละช่อง n ช่อง"""
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    xs = (np.arange(cols) + 0.5) * size / cols
    ys = (np.arange(rows) + 0.5) * size / rows
    return [(x, y) for y in ys for x in xs][:n]


def make_scene(rng, block_names=BASIC, hole_names=None, size=SCENE_SIZE,
               hole_scale=HOLE_SCALE, S=BLOCK_SCALE):
    """คืน (ภาพบล็อก, ภาพช่อง, เฉลยบล็อก, เฉลยช่อง) สลับตำแหน่งและมุมแบบสุ่ม"""
    hole_names = block_names if hole_names is None else hole_names
    block_img = np.zeros((size, size), np.uint8)
    hole_img = np.zeros((size, size), np.uint8)
    truth_b, truth_h = [], []
    for img, truth, scale, names in ((block_img, truth_b, S, block_names),
                                     (hole_img, truth_h, S * hole_scale, hole_names)):
        cells = _cells(len(names), size)
        order = rng.permutation(len(names))
        for cell, k in zip(cells, order):
            ctr = np.array(cell) + rng.uniform(-25, 25, 2)
            ang = rng.uniform(0, 360)
            draw_shape(img, names[k], ctr, ang, scale)
            truth.append((names[k], ctr, ang))
    return block_img, hole_img, truth_b, truth_h


# =============== อัลกอ ===============
def detect(img):
    """หาทุกวัตถุในภาพ -> list ของ {c: centroid(px), pts: เส้นขอบ normalize แล้ว}"""
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 300:
            continue
        m = cv2.moments(c)
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        pts = (c.reshape(-1, 2).astype(np.float64) - [cx, cy]) * np.sqrt(NORM_AREA / area)
        out.append({"c": (cx, cy), "pts": pts})
    return out


def render(pts, angle_deg):
    a = np.deg2rad(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    img = np.zeros((CANVAS, CANVAS), np.uint8)
    cv2.fillPoly(img, [np.round(pts @ R.T + CANVAS / 2).astype(np.int32)], 1)
    return img.astype(bool)


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


def best_rotation(bpts, hpts):
    """คืน (มุมที่ต้องหมุนบล็อก องศา, IoU) โดยเลือกมุมที่หมุนน้อยที่สุดในบรรดามุมที่พอดีเท่ากัน"""
    href = render(hpts, 0)
    angles = np.arange(0, 360, ANGLE_STEP)
    scores = np.array([iou(render(bpts, a), href) for a in angles])
    best = scores.max()
    near = scores >= best - BAND
    if near.mean() > 0.9:                      # ทุกมุมพอดี = วงกลม
        return 0.0, best
    cand = (angles[near] + 180) % 360 - 180    # แปลงเป็นช่วง -180..180
    a0 = cand[np.argmin(np.abs(cand))]         # เลือกมุมที่หมุนน้อยสุด
    fine = np.arange(a0 - 4, a0 + 4.01, 0.5)   # ละเอียดขึ้นรอบมุมนั้น
    fs = [iou(render(bpts, a), href) for a in fine]
    return float(fine[int(np.argmax(fs))]), float(max(fs))


def solve(block_img, hole_img):
    B, H = detect(block_img), detect(hole_img)
    hole_ref = [render(h["pts"], 0) for h in H]
    coarse = np.arange(0, 360, COARSE_STEP)
    S = np.zeros((len(B), len(H)))
    for i, b in enumerate(B):
        rends = [render(b["pts"], a) for a in coarse]
        for j in range(len(H)):
            S[i, j] = max(iou(r, hole_ref[j]) for r in rends)
    rows, cols = linear_sum_assignment(-S)     # Hungarian
    result = []
    for i, j in zip(rows, cols):
        ang, sc = best_rotation(B[i]["pts"], H[j]["pts"])
        result.append({"block": B[i]["c"], "hole": H[j]["c"], "angle": ang,
                       "iou": sc, "fits": sc >= MIN_FIT})
    return result, S


# =============== ทดสอบ ===============
def nearest(truth, c):
    return min(truth, key=lambda t: np.hypot(t[1][0] - c[0], t[1][1] - c[1]))


def evaluate(n_scenes=100, seed=0, names=BASIC):
    """ทดสอบจับคู่และมุมหมุน กับฉากสุ่มที่บล็อกและช่องมีรูปชุดเดียวกัน"""
    rng = np.random.default_rng(seed)
    ok = total = 0
    errs = []
    t0 = time.time()
    for _ in range(n_scenes):
        bi, hi, tb, th = make_scene(rng, names)
        res, S = solve(bi, hi)
        for r in res:
            nb, cb, ab = nearest(tb, r["block"])
            nh, ch, ah = nearest(th, r["hole"])
            total += 1
            if nb == nh:
                ok += 1
                if nb != "circle":
                    p = PERIOD[nb]
                    true = (ah - ab) % p
                    pred = r["angle"] % p
                    errs.append(abs((pred - true + p / 2) % p - p / 2))
    dt = (time.time() - t0) / n_scenes
    print(f"รูปที่ใช้: {', '.join(names)}")
    print(f"จับคู่ถูก: {ok}/{total} ({100 * ok / total:.1f}%)")
    if errs:
        print(f"ค่าคลาดเคลื่อนของมุม (ไม่รวมวงกลม): เฉลี่ย {np.mean(errs):.2f}°; "
              f"สูงสุด {np.max(errs):.2f}°; เกิน 2° จำนวน {sum(e > 2 for e in errs)} ครั้ง")
    print(f"เวลาเฉลี่ยต่อฉาก ({len(names)} ชิ้น): {dt:.2f} วินาที")


def evaluate_no_fit(n_scenes=100, seed=1, block_names=("circle", "square", "hexagon", "star"),
                    hole_names=BASIC):
    """ทดสอบกรณีมีบล็อกที่ไม่มีช่องพอดี: บล็อกนั้นต้องถูกแจ้งว่า "ใส่ไม่ได้" และชิ้นอื่นต้องยังใส่ได้"""
    rng = np.random.default_rng(seed)
    misfit = sorted(set(block_names) - set(hole_names))
    correct = total = 0
    ious = []
    for _ in range(n_scenes):
        bi, hi, tb, th = make_scene(rng, list(block_names), list(hole_names))
        res, _ = solve(bi, hi)
        for r in res:
            nb = nearest(tb, r["block"])[0]
            should_fit = nb not in misfit
            total += 1
            correct += r["fits"] == should_fit
            if not should_fit:
                ious.append(r["iou"])
    print(f"บล็อกที่ไม่มีช่องพอดี: {', '.join(misfit)}")
    print(f"ตัดสินใส่ได้/ใส่ไม่ได้ถูก: {correct}/{total} ({100 * correct / total:.1f}%)")
    print(f"IoU ของบล็อกที่ไม่มีช่อง: เฉลี่ย {np.mean(ious):.3f}, สูงสุด {np.max(ious):.3f} "
          f"(เกณฑ์ใส่ได้ = {MIN_FIT})")


def demo(path=DEMO_IMAGE_PATH, seed=3, names=BASIC):
    rng = np.random.default_rng(seed)
    bi, hi, tb, th = make_scene(rng, names)
    res, _ = solve(bi, hi)
    vis = cv2.cvtColor(np.hstack([bi, hi]), cv2.COLOR_GRAY2BGR)
    size = bi.shape[1]
    for k, r in enumerate(res):
        bx, by = map(int, r["block"]); hx, hy = map(int, r["hole"])
        cv2.line(vis, (bx, by), (hx + size, hy), (0, 200, 255), 2)
        cv2.putText(vis, f"rot {r['angle']:+.1f}", (bx - 50, by + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
        cv2.putText(vis, f"IoU {r['iou']:.2f}", (hx + size - 50, hy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 160, 0), 2)
    cv2.imwrite(path, cv2.resize(vis, None, fx=0.75, fy=0.75))
    for r in res:
        status = "ใส่ได้" if r["fits"] else "ใส่ไม่ได้"
        print(f"บล็อก ({r['block'][0]:.0f}, {r['block'][1]:.0f}) -> "
              f"ช่อง ({r['hole'][0]:.0f}, {r['hole'][1]:.0f})  "
              f"หมุน {r['angle']:+.1f}°  IoU={r['iou']:.3f}  {status}")


if __name__ == "__main__":
    print("=== ทดสอบ 1: รูปพื้นฐาน 4 รูป ===")
    evaluate(EVALUATION_SCENES)
    print("\n=== ทดสอบ 2: เพิ่มรูปที่ไม่เคยเห็น (ไม่ต้องสอน) ===")
    evaluate(EVALUATION_SCENES, names=BASIC + UNSEEN)
    print("\n=== ทดสอบ 3: บล็อกที่ไม่มีช่องพอดี ===")
    evaluate_no_fit(EVALUATION_SCENES)
    print("\n--- ตัวอย่าง 1 ฉาก ---")
    demo()
    demo("demo_unseen.png", names=BASIC + UNSEEN)
