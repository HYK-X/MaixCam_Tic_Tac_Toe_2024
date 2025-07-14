from types import CellType
from maix import camera, display, image, time, nn, app, pinmap, uart
import cv2
from maix._maix.image import image2cv
import numpy as np
import gc
import sys

import serial
sys.path.append(r'/root/moudles')
from board_checker_lite import *
from api_new import *

sys.path.append(r'/root/GUI')
from EZGUI import *  # 导入GUI框架

# 初始化EZGUI
ui = EZGUI()
out_detector = nn.YOLOv5(model="/root/models/maixhub/196601/model_196601.mud")
detector = nn.YOLOv5(model="/root/models/maixhub/model-333.maixcam/model_333.mud")
# detector = nn.YOLOv5(model="/root/models/maixhub/196601/model_196601.mud")
print(detector.input_width(), detector.input_height(), detector.input_format())

# --- UART 配置 (使用 maix.uart) ---
# --- UART Configuration (using maix.uart) ---
UART_DEVICE = "/dev/ttyS1"       # 串口设备文件 (Serial device file)
UART_BAUDRATE = 115200           # 波特率 (Baudrate)
# 根据你的 MaixPy 开发板和接线确定正确的引脚
# Determine the correct pins based on your MaixPy board and wiring
UART_RX_PIN = "A18"              # UART 接收引脚 (UART Receive Pin) - 请确认这是你开发板上 ttyS1 的 RX
UART_TX_PIN = "A19"              # UART 发送引脚 (UART Transmit Pin) - 请确认这是你开发板上 ttyS1 的 TX
last_uart_received = "UART Recv: None"
UART_READ_TIMEOUT_MS = 100 # 读取超时时间 (毫秒) - 用于 serial.read
# --- 全局变量 ---
# --- Global Variables ---
serial = None # 初始化 UART 对象变量 (Initialize UART object variable)

# --- UART 初始化函数 (使用 maix.uart) ---
# --- UART Initialization Function (using maix.uart) ---
def initialize_maix_uart():
    """
    使用 maix.pinmap 和 maix.uart 初始化串口通信。
    Initializes serial communication using maix.pinmap and maix.uart.
    """
    global serial # 声明我们要修改全局变量 uart (Declare modification of global uart variable)
    try:
        print(f"配置 UART 引脚: RX={UART_RX_PIN}, TX={UART_TX_PIN}")
        # print(f"Configuring UART pins: RX={UART_RX_PIN}, TX={UART_TX_PIN}")

        # 根据 UART 设备号动态设置引脚功能名称 (例如, UART1_RX)
        # Dynamically set pin function names based on UART device number (e.g., UART1_RX)
        # 注意: 假设设备号是设备路径的最后一个字符 (e.g., ttyS1 -> 1)
        # Note: Assumes device number is the last character of the device path (e.g., ttyS1 -> 1)
        device_num = UART_DEVICE[-1]
        pinmap.set_pin_function(UART_RX_PIN, f"UART{device_num}_RX")
        pinmap.set_pin_function(UART_TX_PIN, f"UART{device_num}_TX")

        print(f"初始化 UART: 设备={UART_DEVICE}, 波特率={UART_BAUDRATE}")
        # print(f"Initializing UART: Device={UART_DEVICE}, Baudrate={UART_BAUDRATE}")
        serial = uart.UART(UART_DEVICE, UART_BAUDRATE) # 创建 UART 实例 (Create UART instance)

        # 检查串口是否成功打开 (可选但推荐)
        # Check if the serial port is successfully opened (optional but recommended)
        if serial.is_open:
             print("Maix UART 初始化成功")
             # print("Maix UART Initialized Successfully.")
             return True
        else:
            print("Maix UART 初始化失败: 端口未能打开")
            # print("Maix UART Initialization Failed: Port could not be opened.")
            serial = None # 确保初始化失败时 uart 为 None (Ensure uart is None on failure)
            return False

    except Exception as e:
        print(f"Maix UART 初始化失败: {e}")
        # print(f"Maix UART Initialization Failed: {e}")
        serial = None # 确保异常时 uart 为 None (Ensure uart is None on exception)
        return False

def receive_uart_data():
    """
    如果 UART 有可用数据，则使用带超时的非阻塞 read 方法读取。

    返回:
        str: 接收到的数据行 (已解码并去除首尾空白)，如果没有数据或发生错误则返回 None。
    """
    global serial, last_uart_received
    # 检查 UART 是否已初始化并打开
    if serial is None or not serial.is_open:
        # 仅当状态改变时才更新，避免信息刷屏
        if last_uart_received != "UART Recv: Not Initialized": # 保留英文
             last_uart_received = "UART Recv: Not Initialized" # 保留英文
             print("[UART Recv] UART not initialized or closed.") # 保留英文
        return None

    received_line = None
    try:
        # 使用带超时的 read (如果超时时间很小或为0，则为非阻塞)
        # serial.read() 返回字节串，如果超时则返回 None
        data_bytes = serial.read(-1) # 使用定义的超时时间

        if data_bytes:
            # 将字节解码为字符串 (忽略潜在错误) 并去除首尾空白
            received_line = data_bytes.decode('utf-8', errors='ignore').strip()
            if received_line: # 仅在获取到非空数据时更新
                last_uart_received = f"UART Recv: {received_line}" # 保留英文前缀
                print(f"[UART Recv] Received: '{received_line}'") # 保留英文前缀
            else:
                # 接收到字节但解码后为空字符串 (例如，只有空白符)
                # 保持显示上一次有效信息
                pass
        else:
            # 在超时时间内未收到数据，保持显示上一次信息
            pass

    except Exception as e:
        last_uart_received = "UART Recv: Error Reading" # 保留英文
        print(f"[UART Recv] Error reading from UART: {e}") # 保留英文前缀
        received_line = None # 标记出错

    return received_line # 返回此周期接收到的行，否则返回 None

# --- 在脚本开始处调用初始化函数 ---
# --- Call the initialization function at the beginning of the script ---
initialize_maix_uart()

#-------------------------------------------------找棋盘----------------------------------------------------

def find_grid_centers(corners):
    """
    根据棋盘四个角点找出9个棋盘格子的中心点
    
    参数:
        corners: 包含4个角点坐标的列表，顺序为左上、右上、右下、左下
        
    返回:
        包含9个棋盘格子中心点坐标的列表
    """
    # 在标准化坐标空间中定义3x3网格点
    grid_points = np.array([[0, 0], [1/3, 0], [2/3, 0], [1, 0],
                            [0, 1/3], [1/3, 1/3], [2/3, 1/3], [1, 1/3],
                            [0, 2/3], [1/3, 2/3], [2/3, 2/3], [1, 2/3],
                            [0, 1], [1/3, 1], [2/3, 1], [1, 1]])

    # 将角点转换为numpy数组
    src_points = np.array(corners, dtype=np.float32)

    # 定义透视变换的目标点
    dst_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)

    # 计算透视变换矩阵
    transform_matrix = cv2.getPerspectiveTransform(dst_points, src_points)

    # 将网格点变换到图像坐标空间
    transformed_points = cv2.perspectiveTransform(np.array([grid_points], dtype=np.float32), transform_matrix)[0]

    # 计算网格单元的中心点
    centers = []
    for i in range(0, 3):
        for j in range(0, 3):
            cx = (transformed_points[i * 4 + j][0] + transformed_points[i * 4 + j + 1][0] + 
                  transformed_points[(i + 1) * 4 + j][0] + transformed_points[(i + 1) * 4 + j + 1][0]) / 4
            cy = (transformed_points[i * 4 + j][1] + transformed_points[i * 4 + j + 1][1] + 
                  transformed_points[(i + 1) * 4 + j][1] + transformed_points[(i + 1) * 4 + j + 1][1]) / 4
            centers.append((int(cx), int(cy)))

    return centers

def find_qipan():
    """
    查找并识别棋盘，自动调整处理模式直到返回输出
    """
    # 初始化相机
    cam = camera.Camera(320, 320, fps=60)
    # 初始化处理模式
    process_mode = 1
    # 记录连续失败次数
    consecutive_failures = 0
    # 最大连续失败次数，超过则切换模式
    max_failures = 10
    while True:
        img = cam.read()
        
        # 转换为OpenCV格式进行处理
        img_cv = image.image2cv(img, False, False)
        
        # 保存原始图像用于显示
        img_display = img.copy()

        # 根据当前处理模式进行图像处理
        if process_mode == 1:
             # 模式1: LAB颜色空间处理方法
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            ab_channels = cv2.addWeighted(a_channel, 0.5, b_channel, 0.5, 0)
            binary_adaptive = cv2.adaptiveThreshold(
                ab_channels, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 51, 7
            )
            morph = cv2.morphologyEx(binary_adaptive, cv2.MORPH_CLOSE, kernel)
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(morph, kernel, iterations=1)
            binary = dilated
        elif process_mode == 2:
            # 模式2: 经典灰度图处理方法
            gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            filtered = cv2.bilateralFilter(blurred, 9, 75, 75)
            binary_adaptive = cv2.adaptiveThreshold(
                filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            opened = cv2.morphologyEx(binary_adaptive, cv2.MORPH_OPEN, kernel)
            edged = cv2.Canny(opened, 100, 150)
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(edged, kernel, iterations=1)
            binary = dilated
        else:
            # 模式3: 优化的快速处理方法
            gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            filtered = cv2.blur(blurred, (5, 5))
            binary_adaptive = cv2.adaptiveThreshold(
                filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV, 9, 2)
            edged = cv2.Canny(binary_adaptive, 100, 150)
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edged, kernel, iterations=2)
            eroded = cv2.erode(dilated, kernel, iterations=1)
            binary = dilated

        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 筛选合适的四边形轮廓
        valid_quads = []
        for contour in contours:
            # 轮廓面积要足够大
            area = cv2.contourArea(contour)
            if area < 1000:  # 过滤小面积区域
                continue
            # 多边形近似
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # 检查是否为凸四边形
            if len(approx) == 4 and cv2.isContourConvex(approx):
                # 计算宽高比，过滤过于狭长的四边形
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.5 <= aspect_ratio <= 2.0:  # 合理的宽高比
                    valid_quads.append(approx)
                    
        # 如果找到了合适的四边形
        found_board = False
        if valid_quads:
            # 选择最大的四边形作为棋盘
            largest_quad = max(valid_quads, key=cv2.contourArea)
            corners = largest_quad.reshape((4, 2))
            # 按顺序排列角点（左上、右上、右下、左下）
            rect = np.zeros((4, 2), dtype="int")
            s = corners.sum(axis=1)
            rect[0] = corners[np.argmin(s)]        # 左上 (x+y最小)
            rect[2] = corners[np.argmax(s)]        # 右下 (x+y最大)
            diff = np.diff(corners, axis=1)
            rect[1] = corners[np.argmin(diff)]     # 右上 (y-x最小)
            rect[3] = corners[np.argmax(diff)]     # 左下 (y-x最大)
            corners = rect
            
            # 验证四边形是否合理(检查内角)
            angles = []
            for i in range(4):
                pt1 = corners[i]
                pt2 = corners[(i+1)%4]
                pt0 = corners[(i-1)%4]
                # 计算两条边的向量
                v1 = pt0 - pt1
                v2 = pt2 - pt1
                # 计算内角(余弦定理)
                cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cosine = np.clip(cosine, -1.0, 1.0)  # 确保余弦值在有效范围内
                angle = np.arccos(cosine) * 180 / np.pi
                angles.append(angle)
                
            # 检查四个角是否接近90度
            is_valid = all(70 <= angle <= 110 for angle in angles)
            if is_valid:
                found_board = True
                consecutive_failures = 0  # 重置失败计数
                
                # 角点向内收拢以获得更准确的棋盘边界
                half_x = 2
                half_y = 2
                corners[0] += (half_x, half_y)         # 左上角向右下移动
                corners[1] += (-half_x, half_y)        # 右上角向左下移动
                corners[2] += (-half_x, -half_y)       # 右下角向左上移动
                corners[3] += (half_x, -half_y)        # 左下角向右上移动
                
                # 在图像上标记角点
                for corner in corners:
                    cv2.circle(img_cv, tuple(corner), 4, (0, 255, 0), -1)
                
                # 计算棋盘外接矩形
                outer_rect = [
                    corners[:,0].min(), 
                    corners[:,1].min(), 
                    corners[:,0].max() - corners[:,0].min(), 
                    corners[:,1].max() - corners[:,1].min()
                ]
                
                # 使用角点几何计算出格子中心点
                centers = find_grid_centers(corners)
                
                # 画出棋盘外框
                img.draw_rect(
                    outer_rect[0], outer_rect[1], 
                    outer_rect[2], outer_rect[3], 
                    image.COLOR_WHITE
                )
                
                # 标记并编号格子中心点
                if len(centers) == 9:
                    for i in range(9):
                        x, y = centers[i][0], centers[i][1]
                        cv2.circle(img_cv, (x, y), 2, (0, 255, 0), -1)
                        img.draw_string(
                            x, y, f"{i + 1}", 
                            image.COLOR_WHITE, scale=2, thickness=-1
                        )
                    # 成功找到棋盘并且有9个格子，返回结果
                    print(f"corners:{corners}\n centers:{centers}\n")
                    return (corners,centers)
            else:
                # 四边形不够规则，算作失败
                consecutive_failures += 1
        else:
            # 没有找到合适的四边形，算作失败
            consecutive_failures += 1
            
        # 如果连续失败超过阈值，切换处理模式
        if consecutive_failures >= max_failures:
            process_mode = (process_mode % 3) + 1  # 循环切换模式 1->2->3->1
            consecutive_failures = 0  # 重置失败计数
            print(f"Switching to process_mode {process_mode}")
            
        # 在显示器上显示图像
        # 显示处理阶段的结果(便于调试)
        debug_img = image.cv2image(binary, False, False).resize(80, 60)
        img.draw_image(0, img.height() - debug_img.height(), debug_img)
        # 显示当前处理模式
        img.draw_string(5, 5, f"Mode: {process_mode}", image.COLOR_WHITE, scale=1)
        # 显示连续失败次数
        img.draw_string(5, 20, f"Fails: {consecutive_failures}/{max_failures}", 
                         image.COLOR_WHITE, scale=1)
        # 更新UI显示
        ui.update(img)
        time.sleep_ms(50)
    
    # 释放相机资源并回收内存

#-------------------------------------------------------------------------------------------------------------


def is_point_in_polygon(point, polygon):
    """
    判断一个点是否在一个多边形（由顶点列表定义）内部。
    使用射线法 (Ray Casting Algorithm)。

    参数:
    point -- 要检查的点坐标 (x, y)
    polygon -- 多边形的顶点列表 [(x1, y1), (x2, y2), ...]，需按顺序排列

    返回:
    True -- 如果点在多边形内部
    False -- 如果点在多边形外部或边上
    """
    try:
        x, y = float(point[0]), float(point[1]) # 确保坐标是浮点数
    except (TypeError, ValueError, IndexError):
        print(f"警告: is_point_in_polygon 收到无效点坐标: {point}")
        return False # 无法判断时返回 False

    n = len(polygon)
    if n < 3: # 多边形至少需要3个顶点
        # print("警告: is_point_in_polygon 收到无效多边形 (少于3个顶点)") # 可能过于频繁，注释掉
        return False

    inside = False
    try:
        # 确保多边形顶点也是浮点数
        poly_float = [(float(p[0]), float(p[1])) for p in polygon]

        p1x, p1y = poly_float[0]
        for i in range(1, n + 1):
            p2x, p2y = poly_float[i % n] # 获取下一个顶点，最后一个顶点连接回第一个

            # 核心射线法逻辑
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y: # 非水平边
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters: # 处理垂直边和普通斜边
                        inside = not inside
                # 水平边不影响穿越次数 (点正好在水平边上的情况不计入)
                # 点正好在垂直边上或左侧，且在 Y 范围内，会计入

            p1x, p1y = p2x, p2y

    except (TypeError, ValueError, IndexError) as e:
         print(f"警告: is_point_in_polygon 处理多边形顶点时出错: {e}")
         return False # 顶点数据有问题时返回 False

    return inside

def detect_chess_pieces_internal(image_cv, corners_numeric):
    """
    在给定的OpenCV图像中检测棋盘外的黑棋和白棋。

    参数:
    image_cv -- 输入的OpenCV图像 (BGR格式)
    corners_numeric -- 棋盘四个角点的数值坐标列表 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]

    返回:
    一个元组 (black_pieces, white_pieces)，其中：
      black_pieces: 检测到的黑棋列表 [((x, y), radius), ...]
      white_pieces: 检测到的白棋列表 [((x, y), radius), ...]
    如果输入图像无效，则返回 ([], [])。
    """
    if image_cv is None:
        print("内部错误：传入 detect_chess_pieces_internal 的图像为空")
        return [], []

    # 创建输出图像副本 (用于调试绘制，如果需要的话)
    # output = image_cv.copy() # 如果需要绘制标记，取消注释

    # 转换为灰度图
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # 创建HSV版本用于颜色过滤
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

    # 对比度增强以更好地区分棋子
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # ===== 黑棋检测 =====
    # 对黑棋使用较低的阈值
    _, black_thresh = cv2.threshold(enhanced_gray, 70, 255, cv2.THRESH_BINARY_INV)

    # 使用形态学操作改善黑棋轮廓
    black_kernel = np.ones((3, 3), np.uint8)
    black_mask = cv2.morphologyEx(black_thresh, cv2.MORPH_OPEN, black_kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, black_kernel)

    # ===== 白棋检测 - 改进部分 =====
    # 1. 创建红色区域的掩码(用于排除棋盘本身的红色区域，如果棋盘是红色的)
    # 注意：这些值可能需要根据你的棋盘颜色调整
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = red_mask1 | red_mask2 # 合并两个范围的红色掩码

    # 2. 使用背景建模分离白棋 (可能对光照变化敏感)
    background = cv2.GaussianBlur(enhanced_gray, (51, 51), 0) # 大核模糊创建背景估计
    diff = cv2.absdiff(enhanced_gray, background)             # 计算图像与背景的差异
    _, white_thresh_bg = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY) # 阈值化差异图

    # 形态学操作清理背景建模结果
    white_kernel = np.ones((3, 3), np.uint8)
    white_mask_bg = cv2.morphologyEx(white_thresh_bg, cv2.MORPH_OPEN, white_kernel)
    white_mask_bg = cv2.morphologyEx(white_mask_bg, cv2.MORPH_CLOSE, white_kernel)

    # 3. 创建一个专门用于检测白棋的掩码 (基于HSV颜色空间)
    # 灰白色在HSV中的特征是低饱和度、高亮度
    white_lower_hsv = np.array([0, 0, 150])    # HSV下界
    white_upper_hsv = np.array([180, 50, 255]) # HSV上界
    white_mask_hsv = cv2.inRange(hsv, white_lower_hsv, white_upper_hsv)

    # 排除掉之前定义的红色区域 (如果棋盘是红色的)
    white_mask_hsv = white_mask_hsv & ~red_mask

    # 形态学操作清理HSV掩码
    white_mask_hsv = cv2.morphologyEx(white_mask_hsv, cv2.MORPH_OPEN, white_kernel)
    white_mask_hsv = cv2.morphologyEx(white_mask_hsv, cv2.MORPH_CLOSE, white_kernel)

    # 4. 结合多种方法的结果得到最终的白棋掩码
    # 这里结合了背景建模和HSV颜色过滤的结果
    combined_white_mask = white_mask_hsv | white_mask_bg
    combined_white_mask = cv2.morphologyEx(combined_white_mask, cv2.MORPH_OPEN, white_kernel)
    combined_white_mask = cv2.morphologyEx(combined_white_mask, cv2.MORPH_CLOSE, white_kernel)

    # ===== 找到黑棋和白棋的轮廓 =====
    # 黑棋轮廓
    black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 白棋轮廓
    white_contours, _ = cv2.findContours(combined_white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ===== 处理黑棋 =====
    black_pieces_found = []
    # 最小和最大面积阈值，需要根据实际棋子大小和相机距离调整
    min_area_black, max_area_black = 300, 1300 # 示例值
    min_circularity = 0.6 # 圆形度阈值

    for contour in black_contours:
        area = cv2.contourArea(contour)
        # 过滤掉面积过小或过大的轮廓
        if not (min_area_black < area < max_area_black):
            continue

        # 获取最小外接圆
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (float(x), float(y)) # 使用浮点数以便精确判断位置
        radius_int = int(radius)

        # 计算圆形度
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        # 检查是否是圆形棋子，并且必须在棋盘外部
        if circularity > min_circularity and not is_point_in_polygon(center, corners_numeric):
            # 进一步检查颜色确认是黑棋 (在灰度图上检查平均亮度)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            mean_color = cv2.mean(gray, mask=mask)[0]

            # 黑棋应该是暗色的，阈值可能需要调整
            if mean_color < 100:
                black_pieces_found.append(((int(x), int(y)), radius_int))
    # ===== 处理白棋 =====
    white_pieces_found = []
    # 白棋的面积阈值可能与黑棋不同
    min_area_white, max_area_white = 300, 1300 # 示例值

    for contour in white_contours:
        area = cv2.contourArea(contour)
        # 过滤面积
        if not (min_area_white < area < max_area_white):
            continue

        # 获取最小外接圆
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (float(x), float(y))
        radius_int = int(radius)

        # 计算圆形度
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        # 检查是否是圆形棋子，并且必须在棋盘外部
        if circularity > min_circularity and not is_point_in_polygon(center, corners_numeric):
            # 检查颜色确认是白棋 (在灰度图上检查平均亮度)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            mean_color = cv2.mean(gray, mask=mask)[0]

            # 白棋应该是亮色的，阈值可能需要调整
            if mean_color > 120:
                 white_pieces_found.append(((int(x), int(y)), radius_int))
    # 返回找到的棋子列表
    return black_pieces_found, white_pieces_found


class DummyCamera:
    def read(self): return None

def find_qizi_out(corners):
    """
    持续检测棋盘外围的棋子，直到找到【正好】5个黑棋和5个白棋为止。
    结合改进的 CV 方法和可选的 Detector 模型。

    参数:
    corners -- 棋盘四个角点坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
               按顺序：左上, 右上, 右下, 左下. 可以是 list, tuple 或 numpy array.

    返回:
    black_xy_out_final, white_xy_out_final -- 包含【正好】5个黑棋和5个白棋位置的列表。
                                              格式如: ["black0: (x, y)", ...]
    如果无法完成任务或出错，返回 ([], [])
    """
    # --- 1. 输入有效性检查 ---
    if corners is None:
        print("错误：提供的角点 'corners' 为 None。")
        return [], []
    try:
        if len(corners) != 4:
            print(f"错误：提供的角点 'corners' 应包含4个点，但找到 {len(corners)} 个。")
            return [], []
    except TypeError:
        print(f"错误：提供的角点 'corners' 类型无效或没有长度: {type(corners)}。")
        return [], []

    # 尝试将角点转换为数值类型
    try:
        corners_numeric = [(float(p[0]), float(p[1])) for p in corners]
    except (TypeError, ValueError, IndexError) as e:
        print(f"错误：角点坐标 'corners' 包含无法转换为数值的数据: {e}")
        return [], []

    # --- 2. 初始化相机 ---
    cam = None
    try:
        cam = camera.Camera(320, 320, fps=30) # 使用 320x320 分辨率, 30fps
        print("相机初始化成功。")
    except Exception as e:
        print(f"错误：无法初始化相机: {e}")
        if isinstance(cam, DummyCamera): print("将使用虚拟相机继续，仅用于代码流程演示。")
        else: return [], [] # 如果不是虚拟相机且初始化失败，则无法继续

    # --- 3. 初始化状态变量 ---
    final_black_result = [] # 最终存储【正好5个】黑棋格式化字符串的列表
    final_white_result = [] # 最终存储【正好5个】白棋格式化字符串的列表
    found_target = False      # 标志位：是否已找到 5黑5白 目标
    current_attempt_number = 0 # 总尝试次数计数器

    print("开始持续检测棋盘外围棋子...")
    print("目标：【正好】 5 个黑色棋子 和 5 个白色棋子。")
    print(f"策略：循环 [CV 检测 -> Detector 检测(若需要且可用)] 直到成功。")
    if out_detector is None:
        print("提示：Detector 模型未加载，将仅使用 CV 方法。")

    # --- 4. 主检测循环 (使用 try...finally 保证资源释放) ---
    try:
        while not found_target:
            current_attempt_number += 1
            print(f"\n--- 开始检测周期 {current_attempt_number} ---")

            # --- 4.1 阶段一：执行一次 CV 检测 ---
            print("--- 阶段 1: 执行 CV 方法 ---")
            black_pieces_cv_raw = [] # CV找到的原始黑棋数据 ((x,y), r)
            white_pieces_cv_raw = [] # CV找到的原始白棋数据 ((x,y), r)
            num_black_cv = 0
            num_white_cv = 0
            cv_success_this_cycle = False # CV 是否在本周期直接找到 5+5

            try:
                img = cam.read() # 读取图像 (可能是特定格式)
                if img is None:
                    print("    错误: CV 读取图像失败，跳过此周期。")
                    time.sleep_ms(200)
                    continue # 进入下一个检测周期

                # 图像转换 (需要根据你的 'image' 库调整)
                img_cv = None
                if hasattr(image, 'image2cv'): # 检查是否有转换函数
                     img_cv = image.image2cv(img, copy=True) # 转为 BGR OpenCV 格式
                elif isinstance(img, np.ndarray): # 如果读取直接是 numpy 数组
                     img_cv = img # 假设格式兼容 (例如 BGR)
                # 添加其他可能的转换逻辑

                if img_cv is None:
                   print("    错误: CV 转换图像到 OpenCV 格式失败。跳过此周期。")
                   time.sleep_ms(100)
                   continue

                # 调用内部检测函数
                black_pieces_cv_raw, white_pieces_cv_raw = detect_chess_pieces_internal(img_cv, corners_numeric)

                num_black_cv = len(black_pieces_cv_raw)
                num_white_cv = len(white_pieces_cv_raw)
                print(f"    CV 检测到: 黑棋 = {num_black_cv}, 白棋 = {num_white_cv}")

                # --- 检查 CV 是否直接满足【正好】5+5 的条件 ---
                if num_black_cv == 5 and num_white_cv == 5:
                    print("    成功: CV 方法在本周期找到【正好】 5 黑 5 白！")
                    final_black_result = [f"black{i}: ({pos[0]}, {pos[1]})" for i, (pos, r) in enumerate(black_pieces_cv_raw)]
                    final_white_result = [f"white{i}: ({pos[0]}, {pos[1]})" for i, (pos, r) in enumerate(white_pieces_cv_raw)]
                    found_target = True  # 设置成功标志
                    cv_success_this_cycle = True # 标记 CV 直接成功
                    # 可选：成功后显示结果
                    if ui:
                        try:
                            disp_img = img.copy().resize(320, 320)
                            # 画棋盘框
                            pts_int = [(int(p[0]), int(p[1])) for p in corners_numeric]
                            for i in range(len(pts_int)):
                                p1, p2 = pts_int[i], pts_int[(i + 1) % len(pts_int)]
                                disp_img.draw_line(p1[0], p1[1], p2[0], p2[1], color=image.COLOR_YELLOW, thickness=2)
                            # 画找到的棋子
                            for pos, r in black_pieces_cv_raw: disp_img.draw_circle(pos[0], pos[1], r, color=image.COLOR_BLUE, thickness=2)
                            for pos, r in white_pieces_cv_raw: disp_img.draw_circle(pos[0], pos[1], r, color=image.COLOR_GREEN, thickness=2)
                            disp_img.draw_string(5, 5, f"Success! CV {current_attempt_number}", color=image.COLOR_WHITE, scale=1.5)
                            ui.update(disp_img)
                        except Exception as e_disp: print(f"    警告: 更新 UI 显示时出错 (CV Success): {e_disp}")

                # --- CV 调试显示 (即使未成功) ---
                elif ui: # 如果 CV 未成功，也显示一下结果
                    try:
                        disp_img = img.copy().resize(320, 320)
                        pts_int = [(int(p[0]), int(p[1])) for p in corners_numeric]
                        for i in range(len(pts_int)):
                            p1, p2 = pts_int[i], pts_int[(i + 1) % len(pts_int)]
                            disp_img.draw_line(p1[0], p1[1], p2[0], p2[1], color=image.COLOR_YELLOW, thickness=2)
                        # 画CV找到的所有棋子
                        for pos, r in black_pieces_cv_raw: disp_img.draw_circle(pos[0], pos[1], r, color=image.COLOR_BLUE, thickness=1)
                        for pos, r in white_pieces_cv_raw: disp_img.draw_circle(pos[0], pos[1], r, color=image.COLOR_GREEN, thickness=1)
                        disp_img.draw_string(5, 5, f"CV {current_attempt_number} B:{num_black_cv} W:{num_white_cv}", color=image.COLOR_WHITE, scale=1.5)
                        ui.update(disp_img)
                    except Exception as e_disp: print(f"    警告: 更新 UI 显示时出错 (CV Debug): {e_disp}")

            except Exception as e_cv:
                print(f"    错误: CV 阶段 (周期 {current_attempt_number}) 发生异常: {e_cv}")
                time.sleep_ms(100)
                # 即使CV出错，也可能尝试 Detector (如果可用)

            # --- CV 阶段结束 ---
            if found_target: # 如果 CV 直接成功，跳出主循环
                break

            # --- 4.2 阶段二：如果 CV 未成功 且 Detector 可用，尝试 Detector ---
            # 注意：这里的逻辑是，CV 没找到正好 5+5，就尝试用 Detector 找一次
            # 如果需要更复杂的策略（例如 CV 找到 >=5 黑，用 Detector 补白），需要修改
            detector_needed = not cv_success_this_cycle # 如果CV没直接成功就需要
            if detector_needed and out_detector is not None:
                print(f"--- 阶段 2: CV 未找到 5+5，尝试 Detector (周期 {current_attempt_number}) ---")

                black_pieces_det_raw = [] # Detector找到的原始黑棋数据 ((x,y), w, h, class_id)
                white_pieces_det_raw = [] # Detector找到的原始白棋数据
                num_black_det = 0
                num_white_det = 0

                try:
                    # Detector 可能需要不同的图像输入或预处理
                    # 假设 detector.detect() 使用与 CV 相同的 'img' 对象
                    if img is None: # 确保 CV 阶段读取了图像
                         img = cam.read()
                         if img is None:
                             print("    错误: Detector 无法读取图像，跳过 Detector 尝试。")
                             time.sleep_ms(200)
                             continue # 跳到下一个周期

                    # 执行检测 (参数需要根据你的模型调整)
                    objs = out_detector.detect(img, conf_th=0.4, iou_th=0.35) # 示例参数

                    qizi_list_det_display = [] # 仅用于调试显示

                    for obj in objs:
                        # 计算中心点
                        center_x = obj.x + obj.w / 2.0
                        center_y = obj.y + obj.h / 2.0
                        center_point = (float(center_x), float(center_y))

                        # 1. 位置筛选 (棋盘外)
                        if is_point_in_polygon(center_point, corners_numeric):
                            continue

                        # 2. 类别判断 (根据模型输出的标签)
                        try:
                            # 假设 detector.labels 是 ['background', 'black', 'white', ...]
                            piece_class = out_detector.labels[obj.class_id]
                        except (IndexError, AttributeError):
                             print(f"    警告：无法获取检测对象类别 (class_id: {obj.class_id})")
                             continue

                        color_name = piece_class.lower()
                        pos_int = (int(center_x), int(center_y)) # 使用中心点位置

                        # 存储原始检测结果，包含位置信息 (这里只存中心点)
                        # 可以根据需要存储 obj.x, obj.y, obj.w, obj.h
                        if color_name == "black":
                            black_pieces_det_raw.append({'position': pos_int}) # 可以添加更多信息
                            qizi_list_det_display.append({'position': pos_int, 'color': 'black'})
                            num_black_det += 1
                        elif color_name == "white":
                            white_pieces_det_raw.append({'position': pos_int})
                            qizi_list_det_display.append({'position': pos_int, 'color': 'white'})
                            num_white_det += 1

                    print(f"    Detector 检测到: 黑棋 = {num_black_det}, 白棋 = {num_white_det}")

                    # --- 检查 Detector 是否找到了【正好】5+5 ---
                    if num_black_det == 5 and num_white_det == 5:
                        print("    成功: Detector 方法在本周期找到【正好】 5 黑 5 白！")
                        # 格式化 Detector 的结果
                        final_black_result = [f"black{i}: ({p['position'][0]}, {p['position'][1]})" for i, p in enumerate(black_pieces_det_raw)]
                        final_white_result = [f"white{i}: ({p['position'][0]}, {p['position'][1]})" for i, p in enumerate(white_pieces_det_raw)]
                        found_target = True # 设置成功标志

                        # 可选：成功后显示结果
                        if ui:
                            try:
                                disp_img = img.copy().resize(320, 320)
                                pts_int = [(int(p[0]), int(p[1])) for p in corners_numeric]
                                for i in range(len(pts_int)):
                                    p1, p2 = pts_int[i], pts_int[(i + 1) % len(pts_int)]
                                    disp_img.draw_line(p1[0], p1[1], p2[0], p2[1], color=image.COLOR_YELLOW, thickness=2)
                                # 画 Detector 找到的棋子 (用小点表示中心)
                                for qizi in qizi_list_det_display:
                                    clr = image.COLOR_BLUE if qizi['color'] == 'black' else image.COLOR_GREEN
                                    disp_img.draw_circle(qizi['position'][0], qizi['position'][1], 5, color=clr, thickness=-1)
                                disp_img.draw_string(5, 5, f"Success! Det {current_attempt_number}", color=image.COLOR_WHITE, scale=1.5)
                                ui.update(disp_img)
                            except Exception as e_disp: print(f"    警告: 更新 UI 显示时出错 (Det Success): {e_disp}")

                    # --- Detector 调试显示 (即使未成功) ---
                    elif ui:
                         try:
                             disp_img = img.copy().resize(320, 320)
                             pts_int = [(int(p[0]), int(p[1])) for p in corners_numeric]
                             for i in range(len(pts_int)):
                                 p1, p2 = pts_int[i], pts_int[(i + 1) % len(pts_int)]
                                 disp_img.draw_line(p1[0], p1[1], p2[0], p2[1], color=image.COLOR_YELLOW, thickness=2)
                             for qizi in qizi_list_det_display:
                                 clr = image.COLOR_BLUE if qizi['color'] == 'black' else image.COLOR_GREEN
                                 disp_img.draw_circle(qizi['position'][0], qizi['position'][1], 3, color=clr, thickness=-1)
                             disp_img.draw_string(5, 5, f"Det {current_attempt_number} B:{num_black_det} W:{num_white_det}", color=image.COLOR_WHITE, scale=1.5)
                             ui.update(disp_img)
                         except Exception as e_disp: print(f"    警告: 更新 UI 显示时出错 (Det Debug): {e_disp}")


                except Exception as e_det:
                    print(f"    错误: Detector 阶段 (周期 {current_attempt_number}) 发生异常: {e_det}")
                    time.sleep_ms(100)

            elif detector_needed and out_detector is None:
                print(f"--- 阶段 2: CV 未成功，且 Detector 不可用。将等待后开始新周期。 ---")
                time.sleep_ms(300) # 等待一下

            # --- Detector 阶段结束 ---
            if found_target: # 如果 Detector 成功，跳出主循环
                break

            # 如果 CV 和 Detector (如果尝试了) 都没找到 5+5
            if not found_target:
                 print(f"--- 周期 {current_attempt_number} 结束，未找到 5+5。将继续... ---")
                 time.sleep_ms(100) # 短暂延时，准备下一个周期

            # 执行垃圾回收，对内存有限的设备可能有帮助
            gc.collect()

    # --- 5. 循环结束后的处理 ---
    finally:
        # 确保释放相机资源
        if cam and not isinstance(cam, DummyCamera): # 只释放真实相机
            print("\n释放相机资源...")
            try:
                # 根据你的相机库，可能需要调用 cam.close() 或 del cam
                del cam # 或者 cam.deinit() 等
                print("相机资源已释放。")
            except Exception as e_close:
                print(f"警告: 释放相机资源时出错: {e_close}")
        gc.collect() # 强制进行垃圾回收

    # --- 6. 返回结果 ---
    if found_target and len(final_black_result) == 5 and len(final_white_result) == 5:
        print(f"\n检测完成，在第 {current_attempt_number} 次尝试周期成功找到【正好】5 黑 5 白棋子。")
        return final_black_result, final_white_result
    else:
        # 如果循环是因为某种原因退出（例如手动中断，或未来可能加入的最大尝试次数限制）
        print(f"\n警告：检测过程结束（共 {current_attempt_number} 个周期），但未能找到满足【正好】 5 黑 5 白条件的棋子。")
        return [], [] # 返回空列表表示未完全成功
    
#-------------------------------------------------找棋局----------------------------------------------------

def find_qiju(corners, centers):
    cam = camera.Camera(320, 320, fps=60)
    # 假设棋盘大小（如标准围棋棋盘可设为19x19，此处示例为3x3）
    board_size = 3
    # 角点向内扩大以获得更准确的检测
    half_x = -10
    half_y = -10
    # 初始化 new_corners
    new_corners = [None] * len(corners)
    # 修改角点坐标
    new_corners[0] = (corners[0][0] + half_x, corners[0][1] + half_y)  # 左上角向右下移动
    new_corners[1] = (corners[1][0] - half_x, corners[1][1] + half_y)  # 右上角向左下移动
    new_corners[2] = (corners[2][0] - half_x, corners[2][1] - half_y)  # 右下角向左上移动
    new_corners[3] = (corners[3][0] + half_x, corners[3][1] - half_y)  # 左下角向右上移动
    # 用于记录每次棋局状态的出现次数，使用tuple作为键（因为list不可哈希）
    board_counts = {}
    iteration = 0

    # 循环10次，记录每次棋局状态
    while iteration < 10:
        img = cam.read()
        image2cv = image.image2cv(img, False, False)

        # 定义目标点（320x320正方形）
        dst_points = np.array([
            [0, 0],           # 左上角
            [320, 0],         # 右上角
            [320, 320],       # 右下角
            [0, 320]          # 左下角
        ], dtype=np.float32)

        # 将corners转换为源点数组
        src_points = np.array(new_corners, dtype=np.float32)

        # 获取透视变换矩阵并应用变换
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_image = cv2.warpPerspective(image2cv, perspective_matrix, (320, 320))
        warped_maixcam = image.cv2image(warped_image, False, False)

        # 在变换后的图像中检测棋子
        objs = detector.detect(warped_maixcam, conf_th=0.3, iou_th=0.35)

        # 初始化棋盘表示与检测到的棋子列表
        board = np.zeros((board_size, board_size), dtype=int)
        detected_pieces = []

        # 将原始棋盘交点转换到变换后的坐标系中
        warped_centers = []
        try:
            for i in range(len(centers)):
                center = centers[i]
                center_x, center_y = float(center[0]), float(center[1])
                center_homogeneous = np.array([center_x, center_y, 1.0])
                warped_point = perspective_matrix.dot(center_homogeneous)
                warped_point = warped_point / warped_point[2] if warped_point[2] != 0 else warped_point
                warped_centers.append((warped_point[0], warped_point[1]))
        except (TypeError, IndexError) as e:
            print(f"处理centers时出错: {e}")
            try:
                warped_centers = []
                for i in range(0, len(centers), 2):
                    if i+1 < len(centers):
                        center_x, center_y = float(centers[i]), float(centers[i+1])
                        center_homogeneous = np.array([center_x, center_y, 1.0])
                        warped_point = perspective_matrix.dot(center_homogeneous)
                        warped_point = warped_point / warped_point[2] if warped_point[2] != 0 else warped_point
                        warped_centers.append((warped_point[0], warped_point[1]))
            except Exception as e:
                print(f"第二种方法处理centers时出错: {e}")
                warped_centers = []
                for row in range(board_size):
                    for col in range(board_size):
                        # 生成均匀分布的默认点
                        x = col * (320 / (board_size - 1))
                        y = row * (320 / (board_size - 1))
                        warped_centers.append((x, y))

        # 对每个检测到的棋子，寻找最近的棋盘交点
        for obj in objs:
            piece_class = detector.labels[obj.class_id]  # 应为 "black" 或 "white"
            piece_x = obj.x + obj.w / 2  # 棋子中心X坐标
            piece_y = obj.y + obj.h / 2  # 棋子中心Y坐标

            min_dist = float('inf')
            closest_idx = -1

            for idx, (center_x, center_y) in enumerate(warped_centers):
                dist = np.sqrt((center_x - piece_x) ** 2 + (center_y - piece_y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = idx

            # 如果棋子足够接近某个中心点，则标记在棋盘上
            if min_dist < 40:
                row = closest_idx // board_size
                col = closest_idx % board_size
                piece_value = 1 if piece_class.lower() == "black" else 2
                if row < board_size and col < board_size:
                    board[row, col] = piece_value

                detected_pieces.append({
                    "type": piece_class,
                    "position": (row, col),
                    "confidence": obj.score
                })

            # 在图像上绘制检测框与信息
            warped_maixcam.draw_rect(
                obj.x, obj.y, obj.w, obj.h,
                color=image.COLOR_RED if piece_class.lower() == "black" else image.COLOR_GREEN
            )
            if closest_idx >= 0:
                row = closest_idx // board_size
                col = closest_idx % board_size
                msg = f'{piece_class}: {obj.score:.2f} @ ({row},{col})'
            else:
                msg = f'{piece_class}: {obj.score:.2f}'
            warped_maixcam.draw_string(
                obj.x, obj.y, msg,
                color=image.COLOR_RED if piece_class.lower() == "black" else image.COLOR_GREEN
            )

        # 打印棋局状态（转换为一维列表表示）
        new_board = [None] * (board_size * board_size)
        for row in range(board_size):
            for col in range(board_size):
                idx = row * board_size + col
                if board[row, col] == 1:
                    new_board[idx] = "X"  # 黑棋
                elif board[row, col] == 2:
                    new_board[idx] = "O"  # 白棋
                    
        # # 更新UI显示
        warped_maixcam.draw_string(5, 5, f"Progress: {iteration+1}/10", image.COLOR_WHITE, scale=1)
        ui.update(warped_maixcam)

        # 将当前棋局状态转换为tuple后记录出现次数
        board_tuple = tuple(new_board)
        if board_tuple in board_counts:
            board_counts[board_tuple] += 1
        else:
            board_counts[board_tuple] = 1

        iteration += 1

    # 统计出现次数最多的棋局状态
    max_board_tuple = max(board_counts, key=board_counts.get)
    max_count = board_counts[max_board_tuple]
    max_board = list(max_board_tuple)
    # 释放相机资源
    del cam
    gc.collect()
    
    return max_board

#-------------------------------------------------------------------------------------------------------------

# 发送UART指令控制机械臂移动棋子
# First, let's add global variables for managing black_xy and white_xy
global_black_xy = []
global_white_xy = []

# Modified send_move_command function to remove used coordinates
import numpy as np
import math

def map_camera_to_real_coordinates(camera_x, camera_y):
    """
    将摄像机坐标映射到实际物理坐标，使用已知的坐标对应关系
    
    参数:
        camera_x (int): 摄像机坐标系中的x坐标
        camera_y (int): 摄像机坐标系中的y坐标
        
    返回:
        tuple: 映射后的实际物理坐标 (real_x, real_y), 或者在失败时返回 None
    """
    # 已知的相机坐标与实际坐标对应点
    calibration_points = [
        (84, 71, 3300, 4500),      
        (227, 66, 14000, 4500),    
        (89, 208, 3300, 15000),   
        (229, 208, 14000, 15000), 
        # (63, 56,   2000,  5000),
        # (67, 124,  2000, 10000),
        # (72, 196,  2000, 14500),
        # (135, 77,   7000,  5000),
        # (134, 144,  7000, 10000),
        # (135, 213,  7000, 14500),
        # (202, 80,  13100,  5000),
        # (200, 144, 13100, 10000),
        # (202, 209, 13100, 14800),
    ]
    
    if len(calibration_points) >= 3:
        A = []
        b = [] # 直接构建 b 向量，不再需要 b_x, b_y
        
        for p in calibration_points:
            cam_x, cam_y, real_x, real_y = p
            # 添加对应 real_x 的行
            A.append([cam_x, cam_y, 1, 0, 0, 0])
            b.append(real_x) # 添加对应的 real_x
            
            # 添加对应 real_y 的行
            A.append([0, 0, 0, cam_x, cam_y, 1])
            b.append(real_y) # 添加对应的 real_y
        
        A = np.array(A)
        b = np.array(b) # 转换为 numpy 数组
        
        # 确保 A 和 b 的维度匹配 (A 的行数应等于 b 的元素数)
        if A.shape[0] != b.shape[0]:
             print(f"错误：矩阵 A ({A.shape}) 和向量 b ({b.shape}) 维度不匹配！")
             # 可以选择返回错误值或使用备用逻辑
             return None # 或者使用下面的简单线性变换作为临时方案

        # 求解变换矩阵
        try:
            # 使用最小二乘法求解超定方程组 Ax = p，其中 p 是变换参数 [a,b,c,d,e,f]
            # np.linalg.lstsq(A, b) 返回解 x, 残差, 秩, 奇异值
            transformation_params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            # 检查求解是否成功 (例如，秩是否足够)
            # print(f"求解结果: params={transformation_params}, residuals={residuals}, rank={rank}")
            if rank < 6 and len(calibration_points) >= 3 : # 对于仿射变换，期望秩为6
                 print(f"警告：最小二乘法求解可能不稳定，秩为 {rank} (期望 6)")
                 # 可以选择是否继续

            # 应用变换
            a, b_coeff, c, d, e, f = transformation_params # 注意变量名 b 与向量 b 冲突，重命名为 b_coeff
            real_x = int(round(a * camera_x + b_coeff * camera_y + c))
            real_y = int(round(d * camera_x + e * camera_y + f))
            
        except np.linalg.LinAlgError as e:
            print(f"错误：最小二乘法求解失败: {e}")
            # 求解失败时，可以返回 None 或使用非常基础的备用方案
            # real_x = int(camera_x * 1.0 + 0) # 示例：非常简单的备用
            # real_y = int(camera_y * 1.0 + 0)
            return None
    else:
        # 校准点不足时
        print("错误：校准点不足（至少需要3个），无法计算仿射变换。")
        # real_x = int(camera_x * 1.0 + 0) # 示例：非常简单的备用
        # real_y = int(camera_y * 1.0 + 0)
        return None # 或者返回一个明确的错误指示

    print(f"坐标映射: 相机({camera_x},{camera_y}) -> 实际({real_x},{real_y})") # 注释掉原来的打印，让调用者决定是否打印
    return real_x, real_y



# Modified send_move_command function to remove used coordinates
# Modified send_move_command function using maix.uart
def send_move_command(command_type, position=None, source=None, centers=None, black_xy=None, white_xy=None, change_result=None):
    """
    发送UART指令控制机械臂移动棋子 (使用 maix.uart)
    Sends UART commands to control the robotic arm for moving pieces (using maix.uart).

    参数 (Parameters):
    command_type (str): 指令类型，可以是 'move_back'（移回棋子）或 'place'（放置棋子）
                        Command type, can be 'move_back' or 'place'.
    position (int): 目标位置索引（0-8），仅在 'place' 模式下使用
                    Target position index (0-8), used only in 'place' mode.
    source (str): 棋子来源，可以是 'black' 或 'white'，仅在 'place' 模式下使用
                  Piece source, can be 'black' or 'white', used only in 'place' mode.
    centers (list): 棋盘9个格子的中心坐标列表
                    List of center coordinates for the 9 board squares.
    black_xy (list): 棋盘外黑棋的坐标信息列表
                     List of coordinate info for black pieces outside the board.
    white_xy (list): 棋盘外白棋的坐标信息列表
                     List of coordinate info for white pieces outside the board.
    change_result (dict): 棋局变化检测结果，用于 move_back 模式
                          Board change detection result, used for move_back mode.

    返回 (Returns):
    bool: 是否成功发送指令并收到 "OK" 响应
          Whether the command was sent successfully and an "OK" response was received.
    """
    global global_black_xy, global_white_xy, serial # 引用全局 uart 对象 (Reference the global uart object)

    # 检查 UART 是否已初始化并打开
    # Check if UART is initialized and open
    if serial is None or not serial.is_open:
        print("UART未初始化或未打开，无法发送指令")
        # print("UART not initialized or open, cannot send command.")
        return False

    command_to_send = "" # 初始化要发送的命令字符串 (Initialize the command string to send)

    try:
        if command_type == 'move_back':
            # 处理棋子移回的情况
            # Handle moving a piece back
            if change_result is None or centers is None:
                print("移回棋子需要提供 change_result 和 centers 参数")
                # print("Moving back piece requires change_result and centers parameters.")
                return False

            if change_result["status"] == "移动":
                moved_piece = change_result["piece"]
                from_pos = change_result["to_pos"]    # 当前位置（需要移回）(Current position (needs moving back))
                to_pos = change_result["from_pos"]    # 原来的位置（移回目标）(Original position (target))

                if 0 <= from_pos < len(centers) and 0 <= to_pos < len(centers):
                    from_x, from_y = centers[from_pos]
                    to_x, to_y = centers[to_pos]

                    real_from_x, real_from_y = map_camera_to_real_coordinates(from_x, from_y)
                    real_to_x, real_to_y = map_camera_to_real_coordinates(to_x, to_y)

                    # 格式化命令: 源坐标,目标坐标 (Format: source_coords,target_coords)
                    command_to_send = f"({real_from_x},{real_from_y}),({real_to_x},{real_to_y})\n" # 添加换行符 (Add newline)
                    print(f"准备发送棋子移回坐标: {command_to_send.strip()}")
                    # print(f"Preparing to send move_back coordinates: {command_to_send.strip()}")
                else:
                    print(f"无效的棋子位置索引: from_pos={from_pos}, to_pos={to_pos}")
                    # print(f"Invalid piece position index: from_pos={from_pos}, to_pos={to_pos}")
                    time.sleep_ms(1000)
                    return False
            else:
                print("无法确定需要移回的棋子位置 (change_result 状态不是 '移动')")
                # print("Cannot determine piece to move back (change_result status is not '移动').")
                time.sleep_ms(1000)
                return False

        elif command_type == 'place':
            # 处理放置棋子的情况
            # Handle placing a piece
            if centers is None or position is None or position < 0 or position >= len(centers):
                print("无效的目标位置或中心点坐标")
                # print("Invalid target position or centers coordinates.")
                time.sleep_ms(1000)
                return False

            target_x, target_y = centers[position]
            source_coords = None

            # --- 从全局列表中获取并移除棋子坐标 ---
            # --- Get and remove piece coordinates from global lists ---
            if source == 'black' and global_black_xy and len(global_black_xy) > 0:
                try:
                    coord_str = global_black_xy[0]
                    coord_part = coord_str.split(": ")[1].strip("()")
                    source_x, source_y = map(int, coord_part.split(", "))
                    source_coords = (source_x, source_y)
                    global_black_xy.pop(0) # 使用后移除 (Remove after use)
                except (IndexError, ValueError, TypeError) as e:
                    print(f"解析或移除黑棋坐标失败: {e}")
                    # print(f"Failed to parse or remove black piece coordinate: {e}")
                    time.sleep_ms(1000)
                    return False
            elif source == 'white' and global_white_xy and len(global_white_xy) > 0:
                try:
                    coord_str = global_white_xy[0]
                    coord_part = coord_str.split(": ")[1].strip("()")
                    source_x, source_y = map(int, coord_part.split(", "))
                    source_coords = (source_x, source_y)
                    global_white_xy.pop(0) # 使用后移除 (Remove after use)
                except (IndexError, ValueError, TypeError) as e:
                    print(f"解析或移除白棋坐标失败: {e}")
                    # print(f"Failed to parse or remove white piece coordinate: {e}")
                    time.sleep_ms(1000)
                    return False

            if source_coords is None:
                print(f"无法获取 {source} 棋子的坐标 (列表为空或解析失败)")
                # print(f"Could not get coordinates for {source} piece (list empty or parse failed).")
                time.sleep_ms(1000)
                return False
            # --- 坐标获取结束 ---
            # --- End of coordinate retrieval ---

            real_source_x, real_source_y = map_camera_to_real_coordinates(source_coords[0], source_coords[1])
            real_target_x, real_target_y = map_camera_to_real_coordinates(target_x, target_y)

            # 格式化命令: 源坐标,目标坐标 (Format: source_coords,target_coords)
            command_to_send = f"({real_source_x},{real_source_y}),({real_target_x},{real_target_y})\n" # 添加换行符 (Add newline)
            print(f"准备发送放置棋子坐标: {command_to_send.strip()}")
            # print(f"Preparing to send place piece coordinates: {command_to_send.strip()}")

        else:
            print(f"未知指令类型: {command_type}")
            # print(f"Unknown command type: {command_type}")
            time.sleep_ms(1000)
            return False

        # --- 发送指令 ---
        # --- Send Command ---
        if command_to_send:
            command_bytes = command_to_send.encode('utf-8') # 编码为字节 (Encode to bytes)
            bytes_sent = serial.write(command_bytes)          # 发送字节 (Send bytes)
            print(f"通过 UART 发送了 {bytes_sent} 字节")
            # print(f"Sent {bytes_sent} bytes via UART.")
        else:
             print("没有生成要发送的指令")
             # print("No command generated to send.")
             return False # 如果没有生成指令，直接返回失败 (Return failure if no command was generated)


        # --- 等待并读取响应 ---
        # --- Wait for and Read Response ---
        print("等待 UART 响应...")
        # print("Waiting for UART response...")
        # maix.uart.readline() 读取直到换行符或超时(内部定义)，返回 bytes
        # maix.uart.readline() reads until newline or internal timeout, returns bytes
        # response_bytes = serial.readline()
        response_bytes = receive_uart_data()

        if response_bytes:
            # 解码字节为字符串，去除首尾空白
            # Decode bytes to string, strip leading/trailing whitespace
            print(f"收到响应: '{response_bytes}'")
            # print(f"Received response: '{response}'")
            return True # 检查响应是否为 "OK" (Check if response is "OK")
        else:
            print("未收到 UART 响应或响应为空")
            # print("No UART response received or response was empty.")
            return False

    except Exception as e:
        print(f"发送或接收 UART 指令时发生错误: {e}")
        # print(f"Error sending or receiving UART command: {e}")
        import traceback
        traceback.print_exc() # 打印详细错误堆栈 (Print detailed error stack)
        time.sleep_ms(1000)
        return False


# 添加人类下棋确认函数
def wait_for_human_move(human_first):
    """
    显示等待人类下棋的界面，并提供确认按钮
    
    参数:
        human_first: 布尔值，True表示人类执黑先手，False表示人类执白后手
    
    返回:
        布尔值，表示人类是否确认完成下棋
    """
    # 创建确认页面
    confirm_page = ui.create_page("Waiting", "confirm", ui.current_page)
    ui.show_page(confirm_page)
    
    # 清除之前的按钮
    if hasattr(confirm_page, 'buttons'):
        confirm_page.buttons = []
    confirm_page.elements = []
    
    # 添加确认按钮
    confirmed = [False]  # 使用列表存储状态，以便在回调中修改
    
    def on_confirm():
        confirmed[0] = True
        ui.back()  # 返回上一页面
    
    ui.add_buttons((1, 1), ["confirm"], [on_confirm])
    
    # 添加返回按钮
    ui.add_back_button("Back")
    
    # 等待用户确认
    while not confirmed[0] and ui.current_page == confirm_page:
        ui.update()
        time.sleep_ms(50)
    
    return confirmed[0]

# 修改play_game函数，添加人类下棋确认逻辑
def play_game(corners, centers, black_xy, white_xy, human_first=True):
    """
    封装三子棋对弈流程，支持人类先手或机器先手
    
    参数:
        corners: 棋盘四个角点坐标
        centers: 棋盘9个格子的中心坐标
        black_xy: 棋盘外黑棋的坐标信息
        white_xy: 棋盘外白棋的坐标信息
        human_first: 布尔值，True表示人类先手，False表示机器先手
    
    返回:
        游戏结果字典，包含胜利者信息
    """
    # 创建游戏页面
    game_page = ui.get_page("game")
    if not game_page:
        game_page = ui.create_page("Tic Tac Toe", "game", ui.main_page)
    ui.show_page(game_page)
    
    # 显示游戏初始信息
    game_img = image.Image(320, 320)
    
    # 初始化游戏API
    game = TicTacToeGame()
    
    # 初始化空棋盘
    initial_board = [None] * 9
    # 初始化棋局变化检测器
    board_detector = BoardChangeDetector(initial_board)
    # 初始化游戏
    game_state = game.initialize_game(human_first=human_first)
    print(f"游戏初始化完成，{'人类' if human_first else '机器'}先手")
    print(f"初始棋盘状态: {game_state['board']}")
    time.sleep_ms(1500)
    # 如果机器先手，先让机器落子
    if not human_first:
        # 获取电脑的第一步落子
        computer_move = game_state.get("computer_move")
        if computer_move is not None:
            # 更新游戏信息
            game_img = image.Image(320, 320)
            print(f"电脑在位置 {computer_move} 落子")
            # 发送指令控制机械臂移动黑棋（机器先手时执黑）
            send_move_command('place', computer_move, 'black', centers, black_xy, white_xy)
            # 更新棋局检测器的状态
            updated_board = initial_board.copy()
            updated_board[computer_move] = "X"  # 先手使用X（黑棋）
            board_detector.update_board(updated_board)
    
    
    while True:
        # 显示轮到人类下棋的提示并等待确认
        if human_first or not game_state.get("is_first_move", False):
            # 显示确认界面并等待人类确认下棋完成
            confirmed = wait_for_human_move(human_first)
            if not confirmed:
                # 用户取消，返回主菜单
                return {"status": "cancelled"}
        
        # 检测当前棋局
        current_board = find_qiju(corners, centers)
        
        # 检测棋局变化
        change_result = board_detector.detect_change(current_board)
        print(f"检测到的棋局变化: {change_result}")
        # 处理棋局变化
        if change_result["status"] == "移动":
            print("检测到棋子被移动，发送指令要求将棋子移回原位")
            send_move_command('move_back', centers=centers, change_result=change_result)
            # 等待棋子被移回原位
            time.sleep(2)
            continue
        elif change_result["status"] == "新增":
            if human_first:
                # 人类先手情况：检查是否是人类玩家新增了一个黑棋(X)
                if change_result["piece"] == "X":
                    # 人类玩家（X）新增了一个棋子
                    human_move = change_result["position"]
                    print(f"人类玩家在位置 {human_move} 落子")
                    # 更新棋局检测器的状态
                    board_detector.update_board(current_board)
                    # 调用API处理人类玩家的落子
                    result = game.make_human_move(human_move)
                    if result["status"] == "success":
                        if result.get("game_over", False):
                            computer_move = result.get("computer_move")
                            if computer_move is not None:
                                print(f"电脑在位置 {computer_move} 落子")
                                # 发送指令控制机械臂移动白棋（人类先手时，电脑执白）
                                send_move_command('place', computer_move, 'white', centers, black_xy, white_xy)
                            # 游戏结束
                            print(f"游戏结束! 胜利者: {result.get('winner', '未知')}")
                            # 显示结果
                            result_img = image.Image(320, 320)
                            winner_text = "Draw!" if result.get('winner') == "Draw" else f"Winner: {result.get('winner', '未知')}"
                            result_img.draw_string(10, 10, winner_text, image.COLOR_WHITE, scale=2)
                            ui.update(result_img)
                            time.sleep(5)
                            # 返回游戏结果
                            return result
                        # 获取电脑的响应
                        computer_move = result.get("computer_move")
                        if computer_move is not None:
                            print(f"电脑在位置 {computer_move} 落子")
                            # 发送指令控制机械臂移动白棋（人类先手时，电脑执白）
                            send_move_command('place', computer_move, 'white', centers, black_xy, white_xy)
                            # 更新棋局检测器的状态，添加电脑的落子
                            updated_board = current_board.copy()
                            updated_board[computer_move] = "O"  # 后手使用O（白棋）
                            board_detector.update_board(updated_board)
                            # 检查游戏是否结束
                            if result.get("game_over", False):
                                print(f"游戏结束! 胜利者: {result.get('winner', '未知')}")
                                # 显示结果
                                result_img = image.Image(320, 320)
                                winner_text = "Draw!" if result.get('winner') == "Draw" else f"Winner: {result.get('winner', '未知')}"
                                result_img.draw_string(10, 10, winner_text, image.COLOR_WHITE, scale=2)
                                ui.update(result_img)
                                time.sleep(5)
                                # 返回游戏结果
                                return result
                    else:
                        print(f"API错误: {result.get('message', '未知错误')}")
                        time.sleep(2)
                else:
                    print("检测到的变化不符合预期，请确保正确放置棋子")
                    time.sleep(2)
            else:
                # 机器先手情况：检查是否是人类玩家新增了一个白棋(O)
                if change_result["piece"] == "O":
                    # 人类玩家（O）新增了一个棋子
                    human_move = change_result["position"]
                    print(f"人类玩家在位置 {human_move} 落子")
                    # 更新棋局检测器的状态
                    board_detector.update_board(current_board)
                    # 调用API处理人类玩家的落子
                    result = game.make_human_move(human_move)
                    if result["status"] == "success":
                        if result.get("game_over", False):
                            computer_move = result.get("computer_move")
                            if computer_move is not None:
                                
                                print(f"电脑在位置 {computer_move} 落子")
                                # 发送指令控制机械臂移动黑棋（机器先手时，电脑执黑）
                                send_move_command('place', computer_move, 'black', centers, black_xy, white_xy)
                            # 游戏结束
                            print(f"游戏结束! 胜利者: {result.get('winner', '未知')}")
                            # 显示结果
                            result_img = image.Image(320, 320)
                            winner_text = "Draw!" if result.get('winner') == "Draw" else f"Winner: {result.get('winner', '未知')}"
                            result_img.draw_string(10, 10, winner_text, image.COLOR_WHITE, scale=2)
                            ui.update(result_img)
                            time.sleep(5)
                            return result
                        
                        # 获取电脑的响应
                        computer_move = result.get("computer_move")
                        if computer_move is not None:
                            
                            print(f"电脑在位置 {computer_move} 落子")
                            
                            # 发送指令控制机械臂移动黑棋（机器先手时，电脑执黑）
                            send_move_command('place', computer_move, 'black', centers, black_xy, white_xy)
                            
                            # 更新棋局检测器的状态，添加电脑的落子
                            updated_board = current_board.copy()
                            updated_board[computer_move] = "X"  # 先手使用X（黑棋）
                            board_detector.update_board(updated_board)
                            
                            # 检查游戏是否结束
                            if result.get("game_over", False):
                                print(f"游戏结束! 胜利者: {result.get('winner', '未知')}")
                                # 显示结果
                                result_img = image.Image(320, 320)
                                winner_text = "Draw!" if result.get('winner') == "Draw" else f"Winner: {result.get('winner', '未知')}"
                                result_img.draw_string(10, 10, winner_text, image.COLOR_WHITE, scale=2)
                                ui.update(result_img)
                                time.sleep(5)
                                # 返回游戏结果
                                return result
                    else:
                        print(f"API错误: {result.get('message', '未知错误')}")
                        time.sleep(2)
                else:
                    print("检测到的变化不符合预期，请确保正确放置棋子")
                    time.sleep(2)
        elif change_result["status"] == "无变化":
            # 如果没有检测到变化，显示提示并继续等待
            # 重新显示确认界面
            confirmed = wait_for_human_move(human_first)
            if not confirmed:
                # 用户取消，返回主菜单
                return {"status": "cancelled"}


# 修改选择模式函数清空全局列表
def select_mode_1():
    global mode, global_black_xy, global_white_xy
    mode = 1
    # 清空全局列表
    global_black_xy = []
    global_white_xy = []
    print("选择模式1: 选棋放置")
    # 返回到主页面
    ui.show_page(ui.main_page)

def select_mode_2():
    global mode, global_black_xy, global_white_xy
    mode = 2
    # 清空全局列表
    global_black_xy = []
    global_white_xy = []
    print("选择模式2: 机器先手对弈")
    # 返回到主页面
    ui.show_page(ui.main_page)

def select_mode_3():
    global mode, global_black_xy, global_white_xy
    mode = 3
    # 清空全局列表
    global_black_xy = []
    global_white_xy = []
    print("选择模式3: 人类先手对弈")
    # 返回到主页面
    ui.show_page(ui.main_page)
def place_piece(color, corners, centers, black_xy, white_xy):
    """
    让用户选择位置放置棋子
    
    参数:
        color: 'black'或'white'，表示要放置的棋子颜色
        corners: 棋盘四个角点坐标
        centers: 棋盘9个格子的中心坐标
        black_xy: 棋盘外黑棋的坐标信息
        white_xy: 棋盘外白棋的坐标信息
    """
    global mode  # Add global mode to reset mode when returning to main menu
    
    # 创建位置选择页面
    pos_page = ui.create_page("Select Position", "position", ui.get_page("place"))
    ui.show_page(pos_page)
    
    # 清除之前的按钮
    if hasattr(pos_page, 'buttons'):
        pos_page.buttons = []
    pos_page.elements = []
    
    # 显示当前选择的棋子颜色
    color_img = image.Image(320, 320)
    color_img.draw_string(10, 10, f"Select position for {color} piece", 
                         image.COLOR_WHITE, scale=2)
    ui.update(color_img)
    
    # 定义位置回调函数
    position_callbacks = []
    for i in range(9):
        position_callbacks.append(lambda pos=i: do_place_piece(color, pos, centers, black_xy, white_xy))
    
    # 添加9个位置按钮
    ui.add_buttons((3, 3), [f"{i+1}" for i in range(9)], position_callbacks)
    
    # 添加返回按钮
    ui.add_back_button("Back")
    
    # 添加主页按钮的回调函数
    def return_to_main():
        global mode
        mode = 0  # Reset mode
        ui.show_page(ui.main_page)  # Return to main page
    
    # 在底部中间添加Home按钮 - 使用Button类并直接添加到页面元素列表
    home_btn = Button(300, 320, 80, 40, "Home", return_to_main)
    pos_page.add_element(home_btn)

# Modify do_place_piece function to properly position the Home button
def do_place_piece(color, position, centers, black_xy, white_xy):
    """
    执行放置棋子操作
    
    参数:
        color: 'black'或'white'，表示要放置的棋子颜色
        position: 0-8，表示要放置的位置
        centers: 棋盘9个格子的中心坐标
        black_xy: 棋盘外黑棋的坐标信息
        white_xy: 棋盘外白棋的坐标信息
    """
    global mode  # Add global mode to reset mode when returning to main menu
    
    # 创建操作页面
    op_page = ui.create_page("Operation", "operation", ui.get_page("position"))
    ui.show_page(op_page)
    
    # 显示操作信息
    op_img = image.Image(320, 320)
    op_img.draw_string(10, 10, f"Place {color} to position {position+1}", 
                      image.COLOR_WHITE, scale=2)
    op_img.draw_string(10, 50, "Processing...", image.COLOR_WHITE, scale=1.5)
    ui.update(op_img)
    
    # 发送指令执行放置
    result = send_move_command('place', position, color, centers, black_xy, white_xy)
    
    # 显示结果
    if result:
        op_img.draw_string(10, 80, "Success!", image.COLOR_GREEN, scale=1.5)
    else:
        op_img.draw_string(10, 80, "Failed!", image.COLOR_RED, scale=1.5)
    ui.update(op_img)
    
    # 清除之前的按钮
    if hasattr(op_page, 'buttons'):
        op_page.buttons = []
    op_page.elements = []
    
    # 添加主页按钮的回调函数
    def return_to_main():
        global mode
        mode = 0  # Reset mode
        ui.show_page(ui.main_page)  # Return to main page
    
    # 添加返回按钮
    ui.add_back_button("Back")
    
    # 在底部中间添加Home按钮 - 使用Button类直接
    home_btn = Button(300, 320, 80, 40, "Home", return_to_main)
    op_page.add_element(home_btn)
    
    # 短暂停留显示结果 - 但不自动返回，让用户选择
    time.sleep_ms(500)

# 修改main函数中的模式1处理部分
def main():
    """
    主函数，实现三子棋对弈流程：可以选择机器先手模式或人类先手模式
    每局游戏开始前都会进行初始化检测
    """
    global mode, ui, global_black_xy, global_white_xy
    
    try:
        # 初始化变量
        corners = None
        centers = None
        
        # 默认模式
        mode = 0  # 初始未选择模式
        
        # 设置主菜单界面
        main_page = ui.get_page("main")
        if not main_page:
            main_page = ui.create_page("Tic Tac Toe Robot", "main", ui.root_page)
        
        # 清除之前的按钮
        if hasattr(main_page, 'buttons'):
            main_page.buttons = []
        main_page.elements = []
        
        # 添加模式选择按钮
        ui.add_buttons((1, 3), 
                      ["Place Piece", "Machine First", "Human First"], 
                      [select_mode_1, select_mode_2, select_mode_3])
        
        # 添加退出按钮
        ui.add_exit_button("Exit")
        
        # 显示主菜单
        ui.show_page(main_page)
        # 创建模式信息页面
        info_page = ui.create_page("Mode Info", "info", main_page)
        # 主循环
        while not app.need_exit():
            ui.update()
            # 根据用户选择的模式进行处理
            if mode in [1, 2, 3] and ui.current_page == main_page:
                # 显示模式信息
                ui.show_page(info_page)
                
                # 清除之前的按钮
                if hasattr(info_page, 'buttons'):
                    info_page.buttons = []
                info_page.elements = []
                
                # 添加返回按钮
                ui.add_back_button("Back")
                
                if mode == 1:
                    ui.update()
                    
                    # 执行初始化检测
                    corners, centers = find_qipan()
                    black_xy, white_xy = find_qizi_out(corners)
                    
                    # 更新全局变量
                    global_black_xy = black_xy.copy() if black_xy else []
                    global_white_xy = white_xy.copy() if white_xy else []
                    
                    if not global_black_xy or not global_white_xy:
                        time.sleep(0.1)
                        mode = 0
                        ui.show_page(main_page)
                        continue
                    
                    # 创建棋子放置页面
                    place_page = ui.create_page("Place Piece", "place", info_page)
                    ui.show_page(place_page)
                    
                    # 清除之前的按钮
                    if hasattr(place_page, 'buttons'):
                        place_page.buttons = []
                    place_page.elements = []
                    
                    # 添加按钮用于选择棋子颜色和位置
                    ui.add_buttons((2, 1), ["Place Black", "Place White"], 
                                  [lambda: place_piece('black', corners, centers, black_xy, white_xy),
                                   lambda: place_piece('white', corners, centers, black_xy, white_xy)])
                    
                    # 添加返回按钮
                    ui.add_back_button("Back")
                    
                    # 添加主页按钮的回调函数
                    def return_to_main():
                        global mode
                        mode = 0  # Reset mode
                        ui.show_page(ui.main_page)  # Return to main page
                    
                    # 在底部中间添加Home按钮 - 直接创建Button并添加到元素列表
                    home_btn = Button(300, 320, 80, 40, "Home", return_to_main)
                    place_page.add_element(home_btn)
                    
                elif mode == 2:
                    # 机器先手模式
                    ui.update()
                    
                    # 执行初始化检测
                    corners, centers = find_qipan()
                    black_xy, white_xy = find_qizi_out(corners)
                    
                    # 更新全局变量
                    global_black_xy = black_xy.copy() if black_xy else []
                    global_white_xy = white_xy.copy() if white_xy else []
                    
                    if not global_black_xy or not global_white_xy:
                        time.sleep(0.1)
                        mode = 0
                        ui.show_page(main_page)
                        continue
                    
                    # 开始游戏，机器先手
                    result = play_game(corners, centers, global_black_xy, global_white_xy, human_first=False)
                    
                    ui.update()
                    time.sleep(0.5)
                    
                    # 重置模式并返回主菜单
                    mode = 0
                    ui.show_page(main_page)
                    
                elif mode == 3:
                    # 人类先手模式
                    ui.update()
                    
                    # 执行初始化检测
                    corners, centers = find_qipan()
                    black_xy, white_xy = find_qizi_out(corners)
                    
                    # 更新全局变量
                    global_black_xy = black_xy.copy() if black_xy else []
                    global_white_xy = white_xy.copy() if white_xy else []
                    
                    if not global_black_xy or not global_white_xy:
                        time.sleep(0.1)
                        mode = 0
                        ui.show_page(main_page)
                        continue
                    
                    result = play_game(corners, centers, global_black_xy, global_white_xy, human_first=True)
                    
                    # 显示游戏结果
                    ui.update()
                    time.sleep(0.5)
                    
                    # 重置模式并返回主菜单
                    mode = 0
                    ui.show_page(main_page)
            
            # 等待一段时间，避免过于频繁的更新
            time.sleep(0.1)
                
    except Exception as e:
        # 异常处理
        import traceback
        error_msg = traceback.format_exc()
        print(error_msg)
        
        # 在屏幕上显示错误信息
        img = image.Image(320, 320)
        img.draw_string(2, 2, error_msg, image.COLOR_WHITE, 
                       font="hershey_complex_small", scale=0.6)
        ui.update(img)
        
        # 保持错误显示直到应用退出
        while not app.need_exit():
            time.sleep(0.2)

if __name__ == '__main__':
    main()