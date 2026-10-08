from dobot_api import DobotApiDashboard, DobotApiMove

ip = "192.168.1.6"
dash = DobotApiDashboard(ip, 29999)
move = DobotApiMove(ip, 30003)

dash.ClearError()
dash.EnableRobot()
dash.SpeedFactor(20)          # ช้าๆ ก่อน
print(dash.GetPose())         # อ่านพิกัดปัจจุบัน

move.MovL(250, 0, 50, 0)      # ยกสูงไว้ก่อน ปลอดภัยกว่า
move.Sync()                   # รอให้ถึงก่อน
dash.DO(1, 1)                 # ลองเปิด gripper
dash.DO(1, 0)