import sys
import subprocess
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QCheckBox, QPushButton, QComboBox, QMessageBox)
from PyQt5.QtCore import Qt

# 定义 win32con 键值
try:
    import win32con
except ImportError:
    QMessageBox.critical(None, "错误", "请确保安装了 pywin32 库")
    sys.exit(1)

key_options = {
    "Ctrl": win32con.VK_CONTROL,
    "鼠标右键": win32con.VK_RBUTTON,
    "鼠标侧键1": win32con.VK_XBUTTON1,
    "鼠标侧键2": win32con.VK_XBUTTON2,
    "左Shift": win32con.VK_LSHIFT,
}

# 中文标签到英文参数的映射
label_to_param = {
    '边框置信度阈值': 'bbox_confidence_threshold',
    '瞄准强度': 'aim_strength',
    '目标调整': 'target_adjustment',
    '显示图像': 'showImage',
    '激活键': 'power_key',
    '文件目录': 'file_dir',
    'UDP IP': 'udp_ip',
    'UDP 端口': 'udp_port',
    '按压力度': 'pressForce'
}

# 定义启动参数的默认值
default_values = {
    'bbox_confidence_threshold': 0.8,
    'aim_strength': 1.0,
    'target_adjustment': 0.3,
    'showImage': False,
    'power_key': key_options["鼠标右键"],
    'file_dir': r"D:\Code\pythonProject\Myself\autoPressDown\PressureInfo",
    'udp_ip': "192.168.8.7",
    'udp_port': 12345,
    'pressForce': 6
}



class ParamAdjuster(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(400, 600)  # 设置窗口固定大小
        self.initUI()

    def get_param_type(self, param_name):
        param_types = {
            'bbox_confidence_threshold': float,
            'aim_strength': float,
            'target_adjustment': float,
            'showImage': bool,
            'power_key': int,
            'file_dir': str,
            'udp_ip': str,
            'udp_port': int,
            'pressForce': int
        }
        return param_types.get(param_name, str)

    def initUI(self):
        self.setWindowTitle('参数调整 GUI')
        # 设置字体大小
        font = self.font()
        font.setPointSize(12)
        self.setFont(font)

        def add_entry(layout, label_text, var_type, default_value):
            label = QLabel(label_text)
            layout.addWidget(label)

            if var_type is bool:
                entry = QCheckBox()
                entry.setChecked(default_value)
            else:
                entry = QLineEdit()
                entry.setText(str(default_value))

            layout.addWidget(entry)
            self.entries[label_text] = entry

        def add_dropdown(layout, label_text, options, default_value):
            label = QLabel(label_text)
            layout.addWidget(label)

            dropdown = QComboBox()
            dropdown.addItems(options.keys())
            dropdown.setCurrentText(default_value)
            layout.addWidget(dropdown)
            self.entries[label_text] = dropdown


        self.entries = {}
        layout = QVBoxLayout()

        # 修改各个标签文本
        # 调用方法时传入 layout
        add_entry(layout, '边框置信度阈值', float, default_values['bbox_confidence_threshold'])
        add_entry(layout, '瞄准强度', float, default_values['aim_strength'])
        add_entry(layout, '目标调整', float, default_values['target_adjustment'])
        add_entry(layout, '显示图像', bool, default_values['showImage'])
        add_dropdown(layout, '激活键', key_options, "鼠标右键")
        add_entry(layout, '文件目录', str, default_values['file_dir'])
        add_entry(layout, 'UDP IP', str, default_values['udp_ip'])
        add_entry(layout, 'UDP 端口', int, default_values['udp_port'])
        add_entry(layout, '按压力度', int, default_values['pressForce'])






        # 创建和放置按钮
        button = QPushButton('开启EH', self)
        button.clicked.connect(self.on_button_click)
        layout.addWidget(button)

        self.setLayout(layout)

    # 修改 on_button_click 方法
    def on_button_click(self):
        args = [
            r"D:\Code\pythonProject\Myself\venv\Scripts\python.exe",
            r"D:\Code\pythonProject\Myself\integration\version4\main2.py"
        ]

        for label_text, entry in self.entries.items():
            param_name = label_to_param[label_text]  # 使用映射的英文参数名称
            param_type = self.get_param_type(param_name)  # 获取参数的类型

            if isinstance(entry, QComboBox):
                value = key_options[entry.currentText()]
            elif isinstance(entry, QCheckBox):
                value = entry.isChecked()
            else:
                value = entry.text()

            # 根据参数类型进行转换
            if param_type == bool and not value:

                continue
            elif param_type == int:
                value = int(value)
            elif param_type == float:
                value = float(value)
            else:
                value = str(value)

            args.append(f"--{param_name}")
            args.append(str(value))

        try:
            subprocess.Popen(args)  # 启动脚本，不阻塞当前线程
        except Exception as e:
            QMessageBox.critical(self, "错误", f"脚本启动失败: {e}")
        finally:
            QApplication.instance().quit()  # 无论成功或失败，都退出当前程序

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ParamAdjuster()
    ex.show()
    sys.exit(app.exec_())