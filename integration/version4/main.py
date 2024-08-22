import argparse
import os
import random
from tkinter import ttk

import psutil
import onnxruntime
import time
import socket

import multiprocessing
import threading

from PIL import Image, ImageTk
from ultralytics import YOLO

import struct
import math
import win32api
import win32con

import ctypes
from ctypes import wintypes
import numpy as np
import cv2
import tkinter as tk

# 解析命令行参数
parser = argparse.ArgumentParser(description="Aiming and shooting automation script.")
parser.add_argument('--bbox_confidence_threshold', type=float, default=0.8, help='置信度 大于该值的框才会被绘制')
parser.add_argument('--aim_strength', type=float, default=1.0, help='自瞄强度')
parser.add_argument('--target_adjustment', type=float, default=0.2, help='瞄准位置调整')
parser.add_argument('--showImage', type=bool, default=False, help='是否显示图像')
parser.add_argument('--power_key', type=int, default=win32con.VK_RBUTTON, help='自瞄监听的按键')
parser.add_argument('--file_dir', type=str, default=r"PressureInfo",
                    help='压枪配置文件目录')
parser.add_argument('--udp_ip', type=str, default="192.168.8.7", help='Hand APP 的 UDP IP地址')
parser.add_argument('--udp_port', type=int, default=12345, help='Hand APP 的 UDP 端口')
parser.add_argument('--pressForce', type=int, default=6, help='按下鼠标左键的力度')

args = parser.parse_args()

print("showImage: ", args.showImage)
# 使用解析的参数
bbox_confidence_threshold = args.bbox_confidence_threshold
aim_strength = args.aim_strength
target_adjustment = args.target_adjustment
showImage = args.showImage
power_key = args.power_key
file_dir = args.file_dir
udp_ip = args.udp_ip
udp_port = args.udp_port
pressForce = args.pressForce
selected_folder = r"D:\Code\pythonProject\Myself\autoPressDown\PressureInfo"


model = None
onnx_path = 'D:\Code\pythonProject\Myself\models\PUBG.onnx'
bbox_color = (203, 219, 120)  # 框的 BGR 颜色
bbox_thickness = 2  # 框的线宽
screen_width = 1920
screen_height = 1080
CAPTURE_SIZE = 320
CLASSES = ['player', 'head']  # coco80类别
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
directions = []  # 压枪方向数据
image_top_left_x = (screen_width - CAPTURE_SIZE) // 2
image_top_left_y = (screen_height - CAPTURE_SIZE) // 2
# 定义必要的 WinAPI 函数和结构
user32 = ctypes.WinDLL('user32', use_last_error=True)
gdi32 = ctypes.WinDLL('gdi32', use_last_error=True)


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD)
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", wintypes.DWORD * 3)  # This is a placeholder; actual size may vary
    ]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG)
    ]


def calculate_movement_vector(target_x, target_y, aim_strength, screen_width=1920, screen_height=1080):
    """
    计算从中心点到目标点的移动向量，并根据自瞄强度调整移动速度。
    :param target_x: 目标点的 x 坐标
    :param target_y: 目标点的 y 坐标
    :param screen_width: 屏幕的宽度
    :param screen_height: 屏幕的高度
    :param aim_strength: 自瞄强度（0 到 1 之间）
    :return: 鼠标移动的 x 和 y 增量
    """
    center_x = screen_width // 2
    center_y = screen_height // 2
    dx = target_x - center_x
    dy = target_y - center_y

    distance = math.sqrt(dx ** 2 + dy ** 2)
    if distance == 0:
        return 0, 0

    # 步长可以根据距离进行缩放，例如最大步长为 20，最小步长为 2
    max_step = 20
    min_step = 2
    step = max(min_step, min(max_step, distance / 10))

    # 根据 aim_strength 调整步长
    step *= aim_strength

    # 计算方向向量并乘以步长
    direction_x = dx / distance
    direction_y = dy / distance
    move_x = direction_x * step
    move_y = direction_y * step

    return int(move_x), int(move_y)


def sendMousePosition(x, y, sock1, udp_ip, udp_port):
    global sock
    if x == 0 and y == 0:
        return

    message = struct.pack('!Bii', 0x01, x, y)
    try:
        sock.sendto(message, (udp_ip, udp_port))
    except Exception as e:
        print("X: ", x, "Y: ", y, "Error: ", e, "UDP IP: ", udp_ip, "UDP Port: ", udp_port, "Message: ", message,
              "sock: ", sock)


def set_window_topmost(window_name):
    hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
    if hwnd:
        ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0003)


def logic_process(pipe_conn, exit_event):
    window_name = "Eyesight"
    if showImage:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)  # 设置窗口总是置顶
        # 窗口大小固定为 320x320
        cv2.resizeWindow(window_name, 320, 320)
        set_window_topmost(window_name)
    global directions, selected_folder, 魔法, globalExitEvent
    globalExitEvent = exit_event
    # 读取压枪配置文件
    select_sensitivity_folder()
    file_path = os.path.join(selected_folder, "main.txt")  # 默认方向数据文件路径
    魔法 = 0
    directions = load_direction_data(file_path)
    index = 0

    while not exit_event.is_set():
        processed_image, nearest_box_xy = pipe_conn.recv()
        if showImage and processed_image is not None:
            cv2.imshow(window_name, processed_image)
            cv2.waitKey(1)

        left_mouse_button_pressed = win32api.GetAsyncKeyState(win32con.VK_LBUTTON) & 0x8000  # 鼠标左键
        aiming_key_pressed = win32api.GetAsyncKeyState(power_key) & 0x8000  # 瞄准键
        menu_key_pressed = win32api.GetAsyncKeyState(win32con.VK_F6) & 0x8000  # 菜单键

        if not aiming_key_pressed:
            # 自动压枪
            if left_mouse_button_pressed:
                if index < len(directions):
                    x, y = directions[index]
                    # 引入随机偏移量
                    random_offset_x = random.uniform(-1.0, 1.0)
                    random_offset_y = random.uniform(-1.0, 1.0)

                    x = int(x * pressForce + random_offset_x * 10)
                    y = int(y * pressForce + random_offset_y * 10)
                    move_mouse(x, y)
                    index += 1

                sendMousePosition(0, 0, sock, udp_ip, udp_port)
                time.sleep(0.05)  # 控制移动速度

            # 监听菜单键
            if menu_key_pressed:
                print("按下F6键，选择配置文件")
                open_file_selection_gui(selected_folder)
                # 等待F6键释放
                while win32api.GetAsyncKeyState(win32con.VK_F6) & 0x8000:
                    time.sleep(0.1)  # 防止循环过快

            if not left_mouse_button_pressed:
                index = 0


        else:
            move_x = nearest_box_xy[0]
            move_y = nearest_box_xy[1]
            if left_mouse_button_pressed:
                if index < len(directions):
                    x, y = directions[index]
                    move_x = move_x + x * 魔法
                    move_y = move_y + y * 魔法
                    print("x压力度: ", move_x, "y压力度: ", move_y, "魔法: ", 魔法)
                    index += 1

            move_x, move_y = calculate_movement_vector(move_x, move_y, aim_strength)
            sendMousePosition(move_x, move_y, sock, udp_ip, udp_port)

        time.sleep(0.05)

    print("Send Process Exit")
    # 清理任务
    if showImage:
        cv2.destroyWindow(window_name)
    pipe_conn.close()
    sock.close()


# 定义BITMAPINFOHEADER和BITMAPINFO结构
class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ('biSize', ctypes.c_uint32),
        ('biWidth', ctypes.c_int32),
        ('biHeight', ctypes.c_int32),
        ('biPlanes', ctypes.c_uint16),
        ('biBitCount', ctypes.c_uint16),
        ('biCompression', ctypes.c_uint32),
        ('biSizeImage', ctypes.c_uint32),
        ('biXPelsPerMeter', ctypes.c_int32),
        ('biYPelsPerMeter', ctypes.c_int32),
        ('biClrUsed', ctypes.c_uint32),
        ('biClrImportant', ctypes.c_uint32)
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ('bmiHeader', BITMAPINFOHEADER),
        ('bmiColors', ctypes.c_uint32 * 3)
    ]


# 创建一次的对象
hdesktop = user32.GetDC(0)
hdc = gdi32.CreateCompatibleDC(hdesktop)


def capture_region(left, top, right, bottom):
    width = right - left
    height = bottom - top

    hbmp = gdi32.CreateCompatibleBitmap(hdesktop, width, height)
    gdi32.SelectObject(hdc, hbmp)
    gdi32.BitBlt(hdc, 0, 0, width, height, hdesktop, left, top, 0x00CC0020)

    bmpinfo = BITMAPINFO()
    bmpinfo.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmpinfo.bmiHeader.biWidth = width
    bmpinfo.bmiHeader.biHeight = -height  # 正数为底部到顶部
    bmpinfo.bmiHeader.biPlanes = 1
    bmpinfo.bmiHeader.biBitCount = 32
    bmpinfo.bmiHeader.biCompression = 0  # BI_RGB
    bmpinfo.bmiHeader.biSizeImage = 0
    bmpinfo.bmiHeader.biXPelsPerMeter = 0
    bmpinfo.bmiHeader.biYPelsPerMeter = 0
    bmpinfo.bmiHeader.biClrUsed = 0
    bmpinfo.bmiHeader.biClrImportant = 0

    buffer = ctypes.create_string_buffer(width * height * 4)
    gdi32.GetDIBits(hdc, hbmp, 0, height, buffer, ctypes.byref(bmpinfo), 0)

    image = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    gdi32.DeleteObject(hbmp)

    return image


def release_resources():
    gdi32.DeleteDC(hdc)
    user32.ReleaseDC(0, hdesktop)
    sock.close()  # 关闭套接字
    cv2.destroyAllWindows()  # 销毁所有OpenCV窗口


def get_center_region():
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    region_width = 320
    region_height = 320
    left = (screen_width - region_width) // 2
    top = (screen_height - region_height) // 2
    right = left + region_width
    bottom = top + region_height
    return left, top, right, bottom


def capture_center_screen():
    """
    捕捉屏幕中央指定大小的区域
    """
    left, top, right, bottom = get_center_region()
    region_screenshot = capture_region(left, top, right, bottom)
    return region_screenshot


def capture_process(pipe_conn, exit_event):
    while not exit_event.is_set():
        if win32api.GetAsyncKeyState(power_key):
            captured_image = capture_center_screen()
            pipe_conn.send(captured_image)
        else:
            pipe_conn.send(None)
            time.sleep(0.1)
    print("Capture Process Exit")


class YOLOV5():
    def __init__(self, onnxpath):
        self.onnx_session = onnxruntime.InferenceSession(onnxpath, providers=['CUDAExecutionProvider'])
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()

    # -------------------------------------------------------
    #   获取输入输出的名字
    # -------------------------------------------------------
    def get_input_name(self):
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_output_name(self):
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    # -------------------------------------------------------
    #   输入图像
    # -------------------------------------------------------
    def get_input_feed(self, img_tensor):
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = img_tensor
        return input_feed

    # -------------------------------------------------------
    #   1.cv2读取图像并resize
    #	2.图像转BGR2RGB和HWC2CHW
    #	3.图像归一化
    #	4.图像增加维度
    #	5.onnx_session 推理
    # -------------------------------------------------------
    def inference(self, img):
        or_img = cv2.resize(img, (320, 320))
        img = or_img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB和HWC2CHW
        img = img.astype(dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        input_feed = self.get_input_feed(img)
        return self.onnx_session.run(None, input_feed)[0]


# dets:  array [x,6] 6个值分别为x1,y1,x2,y2,score,class
# thresh: 阈值
def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    # 计算框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]

    return keep


def xywh2xyxy(x):
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def filter_box(org_box, conf_thres, iou_thres):  # 过滤掉无用的框
    # -------------------------------------------------------
    org_box = np.squeeze(org_box)
    conf_mask = org_box[..., 4] > conf_thres
    box = org_box[conf_mask]

    # 获取每个框的类别
    cls_conf = box[..., 5:]
    cls = np.argmax(cls_conf, axis=1)
    box[:, 5] = cls

    # 获取所有类别
    all_cls = np.unique(cls)
    output = []

    for curr_cls in all_cls:
        # 选择当前类别的框
        curr_cls_mask = box[:, 5] == curr_cls
        curr_cls_box = box[curr_cls_mask][:, :6]

        # 坐标转换
        curr_cls_box = xywh2xyxy(curr_cls_box)

        # 执行NMS
        curr_out_box_indices = nms(curr_cls_box, iou_thres)

        # 收集NMS后的框
        output.extend(curr_cls_box[curr_out_box_indices])

    output = np.array(output)
    return output


def process_image(image, model):
    """
    使用模型对图像进行推理，并在图像上绘制结果
    """
    result = model.inference(image)
    outbox = filter_box(result, 0.7, 0.5)

    if len(outbox) == 0:
        return image, (screen_width // 2, screen_height // 2)

    if showImage:
        boxes = outbox[..., :4].astype(np.int32)
        scores = outbox[..., 4]
        classes = outbox[..., 5].astype(np.int32)

        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = box

            cv2.rectangle(image, (top, left), (right, bottom), bbox_color, bbox_thickness)
            cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                        (top, left),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (33, 66, 131), bbox_thickness)

    return image, find_most_nearby_bbox(outbox, target_adjustment)


def find_most_nearby_bbox(boxes, target_adjustment=0.5):
    """
    找到与目标坐标最近的框，并根据 target_adjustment 调整瞄准位置。
    :param boxes: 所有检测到的框
    :param target_adjustment: 瞄准位置的调整参数，越大瞄准的位置越上面
    :return: 最近的框的中心坐标（经过调整）
    """
    target_x = CAPTURE_SIZE // 2
    target_y = CAPTURE_SIZE // 2
    min_distance = float('inf')
    nearest_bbox_x = target_x
    nearest_bbox_y = target_y

    for bbox_xyxy in boxes:
        bbox_x = (bbox_xyxy[0] + bbox_xyxy[2]) // 2
        bbox_y = (bbox_xyxy[1] + bbox_xyxy[3]) // 2

        # 调整 y 坐标，target_adjustment 越大，瞄准越上
        adjusted_bbox_y = bbox_y - int((bbox_xyxy[3] - bbox_xyxy[1]) * target_adjustment)

        distance = (bbox_x - target_x) ** 2 + (adjusted_bbox_y - target_y) ** 2
        if distance < min_distance:
            min_distance = distance
            nearest_bbox_x = bbox_x
            nearest_bbox_y = adjusted_bbox_y

    return nearest_bbox_x + image_top_left_x, nearest_bbox_y + image_top_left_y


def AI_process(pipe_conn_in, pipe_conn_out, exit_event):
    onnxModel = YOLOV5(onnx_path)

    while not exit_event.is_set():
        captured_image = pipe_conn_in.recv()
        if win32api.GetAsyncKeyState(power_key) and captured_image is not None:
            processed_image, nearest_box_xy = process_image(captured_image, onnxModel)
            pipe_conn_out.send((processed_image, nearest_box_xy))
        else:
            pipe_conn_out.send((None, (0, 0)))
            time.sleep(0.1)
    print("AI Process Exit")


# 模拟鼠标相对移动
def move_mouse(x, y):
    message = struct.pack('!Bii', 0x04, x, y)
    sock.sendto(message, (udp_ip, udp_port))


def on_file_selected(file_path):
    global directions, running
    directions = load_direction_data(file_path)
    root.destroy()


def quit():
    global globalExitEvent
    globalExitEvent.set()
    root.destroy()


# 弹出文件选择 GUI
def open_file_selection_gui(selected_folder):
    global root, running, bg_photo, 魔法
    root = tk.Tk()
    root.title("选择配置文件")
    root.geometry("600x400")  # 固定窗口大小
    root.resizable(False, False)  # 禁止调整窗口大小
    root.attributes("-topmost", True)  # 使窗口总是置顶
    running = False

    # 设置背景图片
    bg_image_path = r"D:\Code\pythonProject\Myself\autoPressDown\img.png"
    bg_image = Image.open(bg_image_path)
    bg_image = bg_image.resize((600, 400), Image.ANTIALIAS)
    bg_photo = ImageTk.PhotoImage(bg_image)

    canvas = tk.Canvas(root, width=600, height=400)
    canvas.pack(fill='both', expand=True)
    canvas.create_image(0, 0, image=bg_photo, anchor='nw')

    # 创建滚动条和可滚动框架
    scrollable_frame = tk.Frame(canvas, bg='')

    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    def on_mouse_wheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", on_mouse_wheel)

    files = [f for f in os.listdir(selected_folder) if f.endswith('.txt')]

    # 自定义按钮样式
    button_font = ('Helvetica', 12)
    button_foreground = 'white'
    button_background = '#333333'
    button_active_background = '#666666'

    # 使用grid布局并排显示按钮
    for idx, file in enumerate(files):
        file_path = os.path.join(selected_folder, file)

        # 创建带透明背景的Label作为按钮
        btn_frame = tk.Frame(scrollable_frame, bg='', bd=0)
        btn_frame.grid(row=idx // 4, column=idx % 4, padx=10, pady=5, sticky='ew')

        btn = tk.Label(btn_frame, text=file, font=button_font, fg=button_foreground,
                       bg=button_background, bd=2, relief="raised", cursor="hand2")
        btn.pack(fill='both', expand=True)
        btn.bind("<Button-1>", lambda e, fp=file_path: on_file_selected(fp))
        btn.bind("<Enter>", lambda e, b=btn: b.config(bg=button_active_background))
        btn.bind("<Leave>", lambda e, b=btn: b.config(bg=button_background))

    # 新增退出按钮
    exit_button = tk.Button(root, text="退出", command=quit, font=button_font,
                            bg=button_background, fg=button_foreground, activebackground=button_active_background)
    exit_button.place(x=10, y=360, width=80, height=30)  # 左下角位置

    def on_closing():
        root.destroy()

    # 新增确定按钮
    confirm_button = tk.Button(root, text="确定", command=on_closing, font=button_font,
                               bg=button_background, fg=button_foreground, activebackground=button_active_background)
    confirm_button.place(x=510, y=360, width=80, height=30)  # 右下角位置

    # 新增滑动条
    def update_magic(value):
        global 魔法
        魔法 = float(value)

    scale = tk.Scale(root, from_=-100, to=100, orient='horizontal', length=400,
                     label='魔法值', command=update_magic)
    scale.set(0)  # 初始值设为0
    scale.place(x=100, y=320)  # 根据需要调整位置

    root.mainloop()


# 从文件中读取方向数据
def load_direction_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    directions = [tuple(map(int, line.strip().split('|'))) for line in lines if line.strip()]
    return directions


def select_sensitivity_folder():
    root = tk.Tk()
    root.title("选择鼠标灵敏度文件夹")
    root.geometry("400x300")
    root.attributes("-topmost", True)

    canvas = tk.Canvas(root)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    def on_mouse_wheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", on_mouse_wheel)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    folders = [f for f in os.listdir(file_dir) if os.path.isdir(os.path.join(file_dir, f))]

    def on_folder_selected(folder_path):
        global selected_folder
        selected_folder = folder_path
        root.destroy()

    for folder in folders:
        folder_path = os.path.join(file_dir, folder)
        btn = tk.Button(scrollable_frame, text=folder, command=lambda fp=folder_path: on_folder_selected(fp))
        btn.pack(fill=tk.X, padx=10, pady=5)

    root.mainloop()


def main():
    parent_conn1, child_conn1 = multiprocessing.Pipe()
    parent_conn2, child_conn2 = multiprocessing.Pipe()
    exit_event = multiprocessing.Event()  # 退出事件
    #截图进程
    capture_proc = multiprocessing.Process(target=capture_process, args=(child_conn1, exit_event),name="CaptureProcess")
    capture_proc.start()
    #AI进程
    AI_proc = multiprocessing.Process(target=AI_process, args=(parent_conn1, child_conn2, exit_event), name="AIProcess")
    AI_proc.start()
    #逻辑进程
    logic_proc = multiprocessing.Process(target=logic_process, args=(parent_conn2, exit_event), name="SendProcess")
    logic_proc.start()

    try:
        capture_proc.join()
        AI_proc.join()
        logic_proc.join()
    finally:
        release_resources()


if __name__ == "__main__":
    main()
