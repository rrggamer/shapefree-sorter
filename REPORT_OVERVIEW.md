# ShapeFree Sorter: ข้อมูลภาพรวมสำหรับทำรายงาน

> **คำสั่งสำหรับ Claude:** ใช้ข้อมูลในไฟล์นี้เขียนรายงานโปรเจกต์วิชา Algorithm เป็นภาษาไทย
> โดยมีหัวข้อ (1) บทนำและเป้าหมาย (2) หน้าที่ของสมาชิกและของแต่ละส่วนในระบบ (3) รายละเอียดการทำงานและอัลกอริทึม
> (4) วิธีการใช้งาน (5) Pseudo code (6) การวิเคราะห์ความซับซ้อน (Big-O) (7) ผลการทดลอง (8) ปัญหาที่พบและสรุป
> ช่องที่เขียนว่า `[...]` ให้เว้นไว้ให้ผู้เขียนกรอกเอง ห้ามแต่งตัวเลขหรือข้อมูลเพิ่ม

---

## 1. บทนำและเป้าหมาย

**ปัญหา:** มีบล็อกหลายรูปทรงวางอยู่บนโต๊ะแบบสุ่มตำแหน่งและสุ่มมุม และมีกระดานที่มีช่องรูปทรงตรงกัน
ต้องให้แขนกลหยิบบล็อกแต่ละชิ้นไปใส่ช่องที่ถูกต้อง พร้อมหมุนให้ตรงมุม

**จุดเด่น (shape-agnostic):** ระบบไม่ได้ถูกสอนว่ามีรูปอะไรบ้าง (ไม่มีการเขียนว่า "นี่คือสี่เหลี่ยม" หรือ "นี่คือวงกลม")
ระบบเทียบรูปร่างของบล็อกกับช่องโดยตรงด้วยการวัดว่าซ้อนกันได้พอดีแค่ไหน
จึงใช้กับรูปที่ไม่เคยเห็นมาก่อนได้ทันที เช่น ดาว หรือหัวใจ

**อุปกรณ์**
- กล้อง Logitech c922 (1280x720) มองลงมาที่โต๊ะ
- แขนกล Dobot MG400 (4 แกน: x, y, z, r) ใช้หัวดูดลม เชื่อมต่อผ่าน LAN (TCP/IP)
- คอมพิวเตอร์ที่รัน Python 3.12 + OpenCV + NumPy + SciPy
- บล็อก 4 ชิ้น (วงกลม สี่เหลี่ยม หกเหลี่ยม คางหมู) และกระดาน 4 ช่อง วางบนกระดาษขาว

**ภาพรวมการทำงาน (pipeline)**
```
กล้องถ่ายภาพ
  → แยกบล็อก / ช่อง ออกจากพื้น (threshold + contour)
  → จับคู่บล็อกกับช่อง (IoU + Hungarian algorithm)
  → หามุมหมุนที่น้อยที่สุดของแต่ละคู่
  → แปลงพิกัดภาพ (พิกเซล) เป็นพิกัดแขน (มม.) ด้วย Homography
  → เรียงลำดับการหยิบ (Greedy nearest neighbor)
  → แขนหยิบ หมุน วาง ทีละชิ้น และถ่ายภาพใหม่ก่อนหยิบทุกชิ้น (closed loop)
```

---

## 2. หน้าที่

### 2.1 หน้าที่ของสมาชิก
| สมาชิก | หน้าที่ |
|---|---|
| [ชื่อสมาชิก A] | อัลกอริทึมจับคู่รูปทรง: IoU, การสร้างตารางคะแนน, Hungarian algorithm, การหามุมหมุนที่น้อยที่สุด (`app/algo.py`) และการทดสอบด้วยฉากจำลอง |
| [ชื่อสมาชิก B] | การแปลงพิกัดกล้อง → แขน (Homography / calibrate), การเรียงลำดับการหยิบ (Greedy), การควบคุมแขนและระบบหยิบวาง |
| [แก้ไขตามจริง] | [...] |

### 2.2 หน้าที่ของแต่ละไฟล์
| ไฟล์ | หน้าที่ | ฟังก์ชันสำคัญ |
|---|---|---|
| `app/algo.py` | อัลกอริทึมหลัก + ฉากจำลองไว้ทดสอบ | `detect`, `render`, `iou`, `solve`, `best_rotation`, `evaluate` |
| `app/camera_check.py` | ฟังก์ชันจัดการกล้องและภาพ | `open_camera`, `binarize`, `clear_border`, `draw_objects` |
| `app/camera_check_h.py` | ดูภาพสด ปรับค่า ตั้งกรอบพื้นที่ทำงาน | `load_area`, `save_area`, `area_mask` |
| `app/calib_utils.py` | แปลงพิกัด และสื่อสารกับแขน | `pixel_to_robot`, `image_angle_to_r`, `in_workspace`, คลาส `Robot` |
| `app/calibrate.py` | calibrate กล้องกับแขน (หา Homography H) | `mode_calibrate`, `mode_check`, `mode_pose` |
| `app/pick_place.py` | โปรแกรมหลัก: วางแผนและสั่งแขนหยิบวาง | `find_objects`, `make_plan`, `order_plan`, `execute`, `run_closed_loop`, `move` |
| `app/calib.json` | ผล calibrate: เมทริกซ์ H, ความสูงโต๊ะ, ขอบเขตพื้นที่ | - |
| `app/work_area.json` | 4 มุมของกระดาษขาว (กรอบพื้นที่ทำงาน) | - |

---

## 3. รายละเอียดการทำงาน

### 3.1 แยกบล็อกและช่องออกจากภาพ (`find_objects` ใน pick_place.py)
1. แปลงภาพเป็นขาวดำ (grayscale) และเบลอ (Gaussian blur) เพื่อลดสัญญาณรบกวน
2. ระบายสีขาวทับทุกอย่างที่อยู่นอกกรอบกระดาษ (work area mask) ของเกะกะรอบโต๊ะจะได้ไม่ถูกนับ
3. Threshold แยก "สีเข้ม" (บล็อกและกระดาน) ออกจากพื้นกระดาษสีขาว
4. หาเส้นขอบ (contour) ของทุกก้อนสีเข้ม แล้วตัดก้อนที่เล็กกว่า 300 พิกเซลทิ้ง
5. **ก้อนที่ใหญ่ที่สุด = กระดาน** ก้อนอื่น = บล็อก
6. ช่อง = ส่วนสีอ่อนที่อยู่ข้างในสี่เหลี่ยมที่ครอบกระดาน (หดขอบเข้ามาเล็กน้อย กันขอบกระดานถูกนับเป็นช่อง)

### 3.2 เตรียมรูปทรงให้เทียบกันได้ (`detect` ใน algo.py)
สำหรับแต่ละวัตถุ: หาจุดศูนย์กลาง (centroid) จาก image moments แล้ว
- ย้ายเส้นขอบให้จุดศูนย์กลางอยู่ที่ (0, 0) เพื่อให้ตำแหน่งไม่มีผล
- ย่อหรือขยายให้มีพื้นที่เท่ากันทุกชิ้น (10,000 ตร.พิกเซล) เพื่อให้ขนาดไม่มีผล
ช่องมักใหญ่กว่าบล็อกเล็กน้อย การทำให้ขนาดเท่ากันจึงทำให้เทียบแค่ "รูปร่าง" ได้

### 3.3 วัดความเหมือนด้วย IoU
- `render(pts, มุม)`: หมุนรูปไปตามมุมที่กำหนด แล้ววาดลงภาพขนาด 300x300
- `IoU = พื้นที่ส่วนที่ซ้อนกัน / พื้นที่รวมของทั้งสองรูป` ได้ค่า 0 ถึง 1 (1 = ซ้อนกันพอดีทุกพิกเซล)
- คะแนนของคู่ (บล็อก i, ช่อง j) = IoU สูงสุดเมื่อหมุนบล็อกทุก 4° (0°, 4°, …, 356° = 90 มุม)

### 3.4 จับคู่ด้วย Hungarian algorithm (`solve`)
- สร้างตารางคะแนน S ขนาด n×m (บล็อก × ช่อง)
- ใช้ `scipy.optimize.linear_sum_assignment` (Hungarian algorithm) หาการจับคู่ 1 ต่อ 1 ที่ทำให้ **ผลรวมคะแนนสูงสุด**
- ทำไมไม่ใช้ Greedy (จับคู่ที่คะแนนสูงสุดก่อน): Greedy อาจแย่งช่องที่ชิ้นอื่นต้องใช้มากกว่า เช่น สี่เหลี่ยมกับหกเหลี่ยมได้คะแนนใกล้กัน
  ส่วน Hungarian รับประกันว่าได้ผลรวมคะแนนดีที่สุดทั้งกระดาน
- ถ้า IoU ของคู่ที่ได้ต่ำกว่า 0.90 จะถือว่า "ใส่ไม่ได้" แล้วข้ามชิ้นนั้น

### 3.5 หามุมหมุนที่น้อยที่สุด (`best_rotation`)
รูปที่สมมาตรมีหลายมุมที่ใส่ได้ เช่น สี่เหลี่ยมจัตุรัสใส่ได้ทุก 90° ระบบจึงเลือกมุมที่หมุนน้อยที่สุดเพื่อให้แขนหมุนน้อยลง
1. ลองหมุนทุก 2° (180 มุม) แล้วเก็บมุมที่ได้คะแนนห่างจากคะแนนสูงสุดไม่เกิน 0.01
2. ถ้ามุมที่ได้คะแนนดีมีมากกว่า 90% ของทุกมุม แปลว่าเป็นวงกลม → ไม่ต้องหมุน (0°)
3. แปลงมุมให้อยู่ช่วง -180° ถึง 180° แล้วเลือกมุมที่มีค่าสัมบูรณ์น้อยที่สุด
4. ค้นหาละเอียดขึ้นทีละ 0.5° ในช่วง ±4° รอบมุมนั้น

ระบบไม่ต้องรู้ว่ารูปสมมาตรกี่มุม เพราะหาเองจากคะแนน IoU

### 3.6 แปลงพิกัดภาพเป็นพิกัดแขน (Homography / calibrate)
- ติดสติกเกอร์บนโต๊ะ N จุด (ใช้ 7 จุด) คลิกตำแหน่งในภาพ (u, v) และนำปลายแขนไปแตะเพื่ออ่านค่า (x, y)
- หาเมทริกซ์ H ขนาด 3×3 ด้วย `cv2.findHomography` (least squares ทุกจุด) ที่ทำให้
  `[x, y, 1]ᵀ ∝ H · [u, v, 1]ᵀ`
- Homography รองรับกรณีกล้องเอียงเล็กน้อย (perspective) ได้ ไม่ต้องให้กล้องตั้งฉากกับโต๊ะพอดี
- **แปลงมุม** (`image_angle_to_r`): ลากลูกศรสั้นจากจุดศูนย์กลาง 2 เส้น (มุม 0° และมุมที่ต้องหมุน) แปลงปลายลูกศรด้วย H
  แล้ววัดมุมระหว่างลูกศรทั้งสองในพิกัดแขน ทำให้ไม่ต้องเดาว่าแกนภาพกับแกนแขนกลับด้านกันหรือไม่

### 3.7 เรียงลำดับการหยิบ (`order_plan`) – Greedy nearest neighbor
- เริ่มจากตำแหน่งที่แขนรออยู่ (CLEAR_POS)
- เลือกบล็อกที่ใกล้ปลายแขนที่สุด → ตำแหน่งปลายแขนจะย้ายไปอยู่ที่ช่องที่เพิ่งวาง → เลือกบล็อกที่ใกล้ที่สุดต่อ
- ช่วยลดระยะทางที่แขนเคลื่อนที่ทั้งหมด (ไม่รับประกันว่าสั้นที่สุด แต่เร็วและพอสำหรับ 4 ชิ้น)

### 3.8 ท่าหยิบวางของแขน (`execute`)
ทุกชิ้นทำ 8 ท่า: ยกขึ้นเหนือบล็อก → ลงไปที่ PICK_Z → เปิดหัวดูด → ยกขึ้น SAFE_Z →
ไปเหนือช่องพร้อมหมุนแกน r → ลงไปที่ PLACE_Z → ปิดหัวดูด → ยกขึ้น
เมื่อทำเสร็จ แขนจะถอยไปที่ CLEAR_POS เพื่อไม่บังกล้อง

### 3.9 ระบบป้องกันและแก้ไขเอง (closed loop, `run_closed_loop`)
- **ถ่ายภาพใหม่ก่อนหยิบทุกชิ้น** ถ้ามีบล็อกถูกเลื่อน ระบบจะเห็นและวางแผนใหม่ให้เอง
- **จำช่องที่วางแล้ว** แล้วลบช่องนั้นออกจากภาพ กันการวางซ้ำช่องเดิม
- **ตรวจว่าหยิบติดไหม:** ถ้ายังเห็นบล็อกอยู่ห่างจากจุดเดิมไม่เกิน 15 มม. แสดงว่าดูดไม่ติด
  รอบถัดไประบบจะลงต่ำกว่าเดิมอีก 1 มม. ลองได้สูงสุด 6 ครั้ง
- **ตรวจการชน:** เปิด collision detection ของแขน (ระดับ 3) และเช็กสถานะแขนทุก 0.1 วินาทีระหว่างเคลื่อนที่
  ถ้าแขนเข้าโหมด error (RobotMode = 9) หรือไปไม่ถึงจุดหมายภายใน 20 วินาที ระบบจะหยุดและปิดหัวดูด
- ไม่ยอมสั่งแขนไปจุดที่อยู่นอกพื้นที่ที่ calibrate ไว้ (เผื่อขอบ 10 มม.)

---

## 4. วิธีการใช้งาน

**ติดตั้ง:** `py -3.12 -m pip install opencv-python numpy scipy`
**เชื่อมต่อ:** ตั้ง IP คอมเป็น 192.168.1.10 และ IP แขนเป็น 192.168.1.6 (เชื่อมด้วยสาย LAN)

| ขั้น | คำสั่ง | ทำอะไร |
|---|---|---|
| 0 | `py -3.12 app/algo.py` | ทดสอบอัลกอริทึมกับฉากจำลอง (ไม่ต้องใช้กล้องหรือแขน) |
| 1 | `py -3.12 app/camera_check_h.py` | ดูภาพกล้อง กด `a` แล้วคลิก 4 มุมกระดาษ เพื่อตั้งกรอบพื้นที่ทำงาน |
| 2 | `py -3.12 app/calibrate.py calibrate` | คลิกสติกเกอร์ในภาพ แล้วนำแขนไปแตะทีละจุด → ได้ `calib.json` |
| 2.1 | `py -3.12 app/calibrate.py check --move` | ตรวจผล calibrate: คลิกในภาพ กด `g` แขนจะไปลอยเหนือจุดที่คลิก |
| 2.2 | `py -3.12 app/calibrate.py pose` | อ่านตำแหน่งแขน ใช้วัดค่า PICK_Z / PLACE_Z |
| 3 | `py -3.12 app/pick_place.py` | ซ้อม: กด `m` ดูแผนบนภาพ แขนไม่ขยับ |
| 4 | `py -3.12 app/pick_place.py --run` | ทำจริง: กด `m` ดูแผน → กด `g` เริ่มหยิบวาง |

ปุ่มใน pick_place: `m` คิดแผน, `g` เริ่มหยิบวาง, `c` ล้างแผน, `q` ออก
ข้อควรระวัง: ห้ามขยับกล้องหรือฐานแขนหลัง calibrate (ถ้าขยับต้อง calibrate ใหม่)

---

## 5. Pseudo code

### 5.1 detect: หาวัตถุและทำให้เทียบกันได้
```
function DETECT(binaryImage):
    objects ← []
    for each contour c in FIND_CONTOURS(binaryImage):
        A ← AREA(c)
        if A < 300: continue                          // ตัดจุดรบกวน
        (cx, cy) ← CENTROID(c)                        // จาก image moments
        pts ← (POINTS(c) − (cx, cy)) × sqrt(10000 / A) // ย้ายไปจุดกำเนิด + ปรับให้พื้นที่เท่ากัน
        objects.append({center: (cx, cy), pts: pts})
    return objects
```

### 5.2 IoU
```
function IOU(maskA, maskB):
    inter ← COUNT(maskA AND maskB)
    union ← COUNT(maskA OR maskB)
    return inter / union  (ถ้า union = 0 คืน 0)

function RENDER(pts, θ):
    return ภาพ 300×300 ที่ระบายรูป ROTATE(pts, θ) ไว้ตรงกลาง
```

### 5.3 solve: จับคู่บล็อกกับช่อง
```
function SOLVE(blockImage, holeImage):
    B ← DETECT(blockImage);  H ← DETECT(holeImage)
    for j in 1..|H|: holeRef[j] ← RENDER(H[j].pts, 0)
    S ← matrix |B| × |H|
    for i in 1..|B|:
        rotated ← [RENDER(B[i].pts, θ) for θ in 0°, 4°, ..., 356°]
        for j in 1..|H|:
            S[i][j] ← max over r in rotated of IOU(r, holeRef[j])
    pairs ← HUNGARIAN(maximize S)                    // linear_sum_assignment(−S)
    result ← []
    for each (i, j) in pairs:
        (angle, score) ← BEST_ROTATION(B[i].pts, H[j].pts)
        result.append({block: B[i].center, hole: H[j].center,
                       angle: angle, iou: score, fits: score ≥ 0.90})
    return result
```

### 5.4 best_rotation: หามุมที่หมุนน้อยที่สุด
```
function BEST_ROTATION(bpts, hpts):
    ref ← RENDER(hpts, 0)
    angles ← [0°, 2°, ..., 358°]
    score[a] ← IOU(RENDER(bpts, a), ref) for each a
    best ← max(score)
    good ← {a : score[a] ≥ best − 0.01}
    if |good| > 0.9 × |angles|: return (0°, best)   // วงกลม: มุมไหนก็ได้
    good ← {WRAP_TO(−180°..180°, a) : a in good}
    a0 ← argmin over a in good of |a|                // หมุนน้อยที่สุด
    fine ← [a0 − 4°, a0 − 3.5°, ..., a0 + 4°]        // ค้นละเอียด 0.5°
    a* ← argmax over a in fine of IOU(RENDER(bpts, a), ref)
    return (a*, IOU(RENDER(bpts, a*), ref))
```

### 5.5 Calibrate (หา Homography)
```
procedure CALIBRATE():
    frame ← CAPTURE()
    pixelPts ← ผู้ใช้คลิกสติกเกอร์ N จุด (N ≥ 4) บน frame
    robotPts ← []
    for each p in pixelPts:
        รอผู้ใช้นำปลายแขนไปแตะสติกเกอร์แล้วกด Enter
        robotPts.append(ROBOT.GET_POSE().xyz)
    H ← FIND_HOMOGRAPHY(pixelPts, robotPts.xy)    // least squares
    error[k] ← ‖ APPLY(H, pixelPts[k]) − robotPts[k].xy ‖
    SAVE(H, z_table = mean(robotPts.z), ขอบเขต x/y, error)

function PIXEL_TO_ROBOT(H, u, v):
    [x', y', w] ← H · [u, v, 1]
    return (x'/w, y'/w)

function IMAGE_ANGLE_TO_R(H, u, v, θ, L = 30):
    c  ← PIXEL_TO_ROBOT(H, u, v)
    p0 ← PIXEL_TO_ROBOT(H, u + L, v) − c
    p1 ← PIXEL_TO_ROBOT(H, u + L·cosθ, v + L·sinθ) − c
    return WRAP_TO(−180°..180°, atan2(p1) − atan2(p0))
```

### 5.6 แยกบล็อก / ช่องจากภาพจริง
```
function FIND_OBJECTS(frame, thresh, blur, workMask):
    g ← GAUSSIAN_BLUR(GRAY(frame), blur)
    g[นอก workMask] ← 255                          // นอกกรอบ = กระดาษขาว
    dark ← THRESHOLD_INV(g, thresh)
    C ← {c in CONTOURS(dark) : AREA(c) ≥ 300}
    board ← argmax over c in C of AREA(c)
    blockImg ← ระบาย C − {board}
    inside ← ERODE(FILL(MIN_AREA_RECT(board)), 7×7)
    holeImg ← THRESHOLD(g, thresh) AND inside        // ส่วนสว่างในกระดาน = ช่อง
    return blockImg, holeImg
```

### 5.7 วางแผนและเรียงลำดับ (Greedy nearest neighbor)
```
function MAKE_PLAN(result, H):
    plan ← []
    for r in result:
        pick  ← PIXEL_TO_ROBOT(H, r.block)
        place ← PIXEL_TO_ROBOT(H, r.hole)
        dr    ← R_SIGN × IMAGE_ANGLE_TO_R(H, r.block, r.angle)
        skip  ← "ใส่ไม่ได้" if not r.fits
                 else "นอกพื้นที่" if pick หรือ place อยู่นอกพื้นที่ calibrate
                 else none
        plan.append({pick, place, dr, skip})
    return ORDER_PLAN(plan)

function ORDER_PLAN(plan):
    todo ← {p in plan : p.skip = none};  done ← []
    pos ← CLEAR_POS.xy
    while todo not empty:
        k ← argmin over p in todo of DIST(p.pick, pos)
        move k from todo to done
        pos ← k.place
    return done + {p in plan : p.skip ≠ none}
```

### 5.8 สั่งแขนหยิบวาง
```
procedure EXECUTE(robot, steps, pickZ):
    MOVE(robot, ตำแหน่งปัจจุบัน แต่ z = max(z, SAFE_Z))   // ยกขึ้นก่อนเสมอ
    for s in steps:
        rPlace ← PICK_R + s.dr
        GRIP(off)
        MOVE(s.pick,  SAFE_Z, PICK_R);  MOVE(s.pick,  pickZ,  PICK_R);  GRIP(on)
        MOVE(s.pick,  SAFE_Z, PICK_R)
        MOVE(s.place, SAFE_Z, rPlace);  MOVE(s.place, PLACE_Z, rPlace); GRIP(off)
        MOVE(s.place, SAFE_Z, rPlace)
    MOVE(CLEAR_POS)                                       // หลบกล้อง

procedure MOVE(robot, x, y, z, r):
    ส่งคำสั่ง MovL(x, y, z, r)
    repeat every 0.1 s until 20 s:
        if RobotMode = 9: raise COLLISION              // ชน หรือ error
        if ตำแหน่งปัจจุบันห่างเป้าหมาย < 0.5 มม. และ 0.5°: return
    raise COLLISION("ไปไม่ถึงในเวลาที่กำหนด")
```

### 5.9 Closed loop: ถ่ายใหม่ก่อนหยิบทุกชิ้น
```
procedure RUN_CLOSED_LOOP(robot):
    used ← [];  last ← none;  fails ← 0
    loop:
        (blockImg, holeImg, blockPts) ← SCAN()           // ถ่ายภาพใหม่
        if last ≠ none:
            if มีบล็อกใน blockPts ห่างจาก last.pick < 15 มม.:   // ดูดไม่ติด
                fails ← fails + 1
                if fails > MAX_RETRY: stop
            else:
                fails ← 0;  used.append(last.place)       // วางสำเร็จ
        if blockPts empty: stop                           // วางครบแล้ว
        holeImg ← ลบช่องที่อยู่ใกล้ตำแหน่งใน used ออก
        steps ← {p in MAKE_PLAN(SOLVE(blockImg, holeImg)) : p.skip = none}
        if steps empty: stop
        step ← steps[0]
        if fails > 0 and มีขั้นที่ pick ใกล้ last.pick: step ← ขั้นนั้น
        else: fails ← 0
        EXECUTE(robot, [step], PICK_Z − fails × RETRY_STEP)   // ดูดไม่ติด → ลงต่ำลง
        last ← step
```

---

## 6. การวิเคราะห์ความซับซ้อน

ให้ n = จำนวนบล็อก, m = จำนวนช่อง (ปกติ n = m), C = ขนาดภาพที่ใช้เทียบ (300 ดังนั้น C² = 90,000 พิกเซล),
K = จำนวนมุมตอนจับคู่ (360/4 = 90), A = จำนวนมุมตอนหามุมหมุน (360/2 = 180 บวกค้นละเอียดอีก 17 มุม), P = จำนวนพิกเซลในภาพกล้อง

| ขั้น | ความซับซ้อน | หมายเหตุ |
|---|---|---|
| Threshold + contour | O(P) | ทำครั้งเดียวต่อภาพ |
| สร้างตาราง S | O(n·K·C² + n·m·K·C²) = **O(n·m·K·C²)** | ส่วนที่ใช้เวลามากที่สุด |
| Hungarian | **O(n³)** | ดีกว่าการลองทุกแบบ ที่เป็น O(n!) |
| หามุมหมุน | O(n·A·C²) | ทำเฉพาะคู่ที่จับได้ n คู่ |
| Homography | O(N) ต่อการ calibrate | N = จำนวนจุด ทำครั้งเดียว |
| เรียงลำดับ Greedy | **O(n²)** | Brute force (ลองทุกลำดับ) = O(n!·n) |
| Closed loop | O(n · เวลา solve) | ถ่ายและคำนวณใหม่ n รอบ (บวกรอบที่ลองซ้ำ) |

สรุป: เวลาส่วนใหญ่ใช้ไปกับการคำนวณ IoU (O(n·m·K·C²)) ส่วนการจับคู่ (Hungarian) ใช้เวลาน้อยมากเมื่อเทียบกัน

---

## 7. ผลการทดลอง

### 7.1 ทดสอบอัลกอริทึมด้วยฉากจำลอง (`py -3.12 app/algo.py`, 20 ฉากสุ่มต่อการทดสอบ)
| การทดสอบ | ผล |
|---|---|
| 4 รูปพื้นฐาน (วงกลม สี่เหลี่ยม หกเหลี่ยม คางหมู) | จับคู่ถูก 80/80 (100%), มุมคลาดเคลื่อนเฉลี่ย 0.21° สูงสุด 0.56°, ใช้เวลาเฉลี่ย 0.76 วินาทีต่อฉาก |
| เพิ่มรูปที่ไม่เคยเห็น (ดาว หัวใจ) รวม 6 รูป | จับคู่ถูก 120/120 (100%), มุมคลาดเคลื่อนเฉลี่ย 0.21° สูงสุด 0.89°, ใช้เวลาเฉลี่ย 2.05 วินาทีต่อฉาก |
| มีบล็อกที่ไม่มีช่องพอดี (ดาว) | ตัดสินใส่ได้/ใส่ไม่ได้ถูก 80/80 (100%), IoU ของบล็อกที่ไม่มีช่องเฉลี่ย 0.668 (ต่ำกว่าเกณฑ์ 0.90 ชัดเจน) |

### 7.2 Calibrate
- ใช้ 7 จุด ความคลาดเคลื่อนเฉลี่ย 2.43 มม. สูงสุด 3.26 มม. (calibrate ล่าสุด 22 ต.ค. 2026)

### 7.3 ทดสอบกับแขนจริง
- หยิบและวางบล็อก 4 ชิ้นลงช่องได้สำเร็จ
- [ใส่จำนวนรอบที่ทดสอบ / อัตราสำเร็จ / เวลาที่ใช้ต่อรอบ ถ้ามี]
- ภาพประกอบ: `images/demo_step1.png`, `images/demo_unseen.png` (ฉากจำลอง), `images/shape_success.png` (ผลจริง)

---

## 8. ปัญหาที่พบและวิธีแก้

| ปัญหา | วิธีแก้ |
|---|---|
| calibrate ครั้งแรกโดยให้หัวจับลอยเหนือโต๊ะ ทำให้ตำแหน่งหยิบคลาดไปทางเดียวกันทุกชิ้น | calibrate ใหม่โดยให้ปลายหัวจับแตะกระดาษ |
| กล้องตั้งเอียง ผิวบนของบล็อก (สูงประมาณ 3 ซม.) ในภาพจึงเลื่อนไปจากตำแหน่งจริง | ย้ายกล้องให้มองลงมาเกือบตรง |
| ของรอบโต๊ะถูกนับเป็นวัตถุ | ใช้กรอบพื้นที่ทำงาน (4 มุมกระดาษ) แล้วมองนอกกรอบเป็นสีขาว |
| ต้องมีเส้นแบ่งฝั่งบล็อก/ฝั่งกระดาน ทำให้วางของได้ไม่อิสระ | ให้ก้อนสีเข้มที่ใหญ่ที่สุดเป็นกระดาน ช่อง = ส่วนสว่างข้างในกระดาน |
| ดูดบล็อกไม่ติดบางครั้ง | ถ่ายภาพใหม่หลังหยิบทุกชิ้น ถ้ายังเห็นบล็อกอยู่ให้ลงต่ำลงทีละ 1 มม. แล้วลองใหม่ |
| แขนชนแล้วโปรแกรมค้าง | เปิดการตรวจชน + เช็กสถานะแขนทุก 0.1 วินาที แทนการรอเฉยๆ |
| แขนขึ้น error 85 | สั่ง ClearError แล้ว EnableRobot ใหม่ (ทำให้อัตโนมัติตอนเริ่มโปรแกรม) |

**สิ่งที่พัฒนาต่อได้:** เรียงลำดับการหยิบด้วย Dynamic Programming (Held–Karp, O(2ⁿ·n²)) เพื่อได้ระยะทางสั้นที่สุดจริง,
calibrate อัตโนมัติจากสติกเกอร์ที่ติดถาวร, ชดเชยความสูงของบล็อก (parallax)
