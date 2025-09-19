import os
import sys
import cv2
import time
from collections import Counter
from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import A_identify


class MainUi(QtWidgets.QMainWindow):
    # ================================ 实现鼠标长按移动窗口功能 ================================#
    def mousePressEvent(self, event):
        self.press_x = event.x()  # 记录鼠标按下的时候的坐标
        self.press_y = event.y()

    def mouseMoveEvent(self, event):
        x = event.x()
        y = event.y()  # 获取移动后的坐标
        if 0 < x < 1200 and 0 < y < 60:
            move_x = x - self.press_x
            move_y = y - self.press_y  # 计算移动了多少
            position_x = self.frameGeometry().x() + move_x
            position_y = self.frameGeometry().y() + move_y  # 计算移动后主窗口在桌面的位置
            self.move(position_x, position_y)  # 移动主窗口

    def __init__(self):
        # ================================ 参数定义 ================================#
        self.press_x = 0
        self.press_y = 0
        self.identify_api = A_identify.Identify()   # 调用识别API
        self.input_image = None                     # 输入图像
        self.output_image = None                    # 输出图像
        self.output_video = None                    # 输出视频
        self.identify_labels = []                   # 检测结果
        self.save_video_flag = False                # 保存视频标志位
        self.save_path = "./A_output/"              # 保存的路径目录
        self.present_time = QDateTime.currentDateTime().toString('yyyy-MM-dd hh:mm:ss')  # 当前时间
        # ================================ 界面生产 ================================#
        super(MainUi, self).__init__()
        self.resize(1060, 600)  # 界面大小
        self.setWindowFlag(Qt.FramelessWindowHint)  # 隐藏默认导航边框
        self.ui_title = "YOLOv5目标检测系统"  # 项目标题，改标题名称直接在这修改即可
        self.timer_time = QTimer()  # 设置定时器1
        self.timer_time.timeout.connect(self.update_time)  # 定时调用时间显示函数
        self.timer_time.start(1000)  # 每1000ms/1s执行一次
        self.timer_video = QTimer()  # 设置定时器2
        self.timer_video.timeout.connect(self.show_video)  # 定时调用视频流检测函数
        # ===== 上方导航栏区 ===== #
        # 导航文字区
        self.label = QLabel(self)
        self.label.setText(self.ui_title + "     " + self.present_time)
        self.label.setFixedSize(860, 60)
        self.label.move(0, 0)
        self.label.setStyleSheet("QLabel{padding-left:30px;background:#303030;color:#ffffff;border:none;"
                                 "font-weight:600;font-size:18px;font-family:'微软雅黑'; }")
        # 导航退出按钮
        self.b_exit1 = QPushButton(self)
        self.b_exit1.setText("☀ 退出系统")
        self.b_exit1.resize(200, 60)
        self.b_exit1.move(860, 0)
        self.b_exit1.setStyleSheet("QPushButton{background:#303030;text-align:center;border:none;"
                                   "font-weight:600;color:#909090;font-size:15px;}")
        self.b_exit1.setCursor(Qt.PointingHandCursor)
        self.b_exit1.clicked.connect(self.close)
        # ===== 左侧参数控制区 ===== #
        # 创建一个小区域
        self.left_widget = QWidget(self)
        self.left_widget.resize(200, 288)
        self.left_widget.move(0, 60)
        self.left_widget.setStyleSheet("QWidget{background:#ffffff;border:none;}")
        # 选择模型文本框
        self.pt_label = QLabel(self.left_widget)
        self.pt_label.setText("选择模型")
        self.pt_label.setFixedSize(190, 30)
        self.pt_label.move(5, 5)
        self.pt_label.setStyleSheet("QLabel{font-size: 18px;color:#999999;font-weight:600;font-family:'微软雅黑'; }")
        # 选择模型显示框
        self.pt_show = QLabel(self.left_widget)
        self.pt_show.setText("")
        self.pt_show.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)
        self.pt_show.setAlignment(Qt.AlignCenter)
        self.pt_show.setStyleSheet("QLabel{background:#ffffff;color:#999999;border:none;font-weight:600;"
                                   "font-size:15px;font-family:'黑体';}")
        self.scroll_area_pt = QScrollArea(self.left_widget)
        self.scroll_area_pt.resize(100, 50)
        self.scroll_area_pt.move(5, 40)
        self.scroll_area_pt.setWidget(self.pt_show)
        self.scroll_area_pt.setWidgetResizable(True)  # 设置 QScrollArea 大小可调整
        self.scroll_area_pt.setStyleSheet("QScrollArea{border: 1px solid #dddddd;}")
        # 选择模型按钮
        self.pt_button = QPushButton(self.left_widget)
        self.pt_button.setText("选择模型")
        self.pt_button.resize(85, 50)
        self.pt_button.move(110, 40)
        self.pt_button.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                     "font-size:18px;font-family:'黑体';border: 1px solid #dddddd;}"
                                     "QPushButton:hover{background:#e6e6e6;}")
        self.pt_button.setFocusPolicy(Qt.NoFocus)
        self.pt_button.setCursor(Qt.PointingHandCursor)
        self.pt_button.clicked.connect(self.input_pt)
        # 置信度阈值文本框
        self.conf_label = QLabel(self.left_widget)
        self.conf_label.setText("置信度conf")
        self.conf_label.setFixedSize(190, 30)
        self.conf_label.move(5, 95)
        self.conf_label.setStyleSheet("QLabel{font-size: 18px;color:#999999;font-weight:600;font-family:'微软雅黑'; }")
        # 置信度阈值调节框
        self.conf_spin_box = QDoubleSpinBox(self.left_widget)
        self.conf_spin_box.resize(55, 25)
        self.conf_spin_box.move(5, 130)
        self.conf_spin_box.setMinimum(0.0)  # 最小值
        self.conf_spin_box.setMaximum(1.0)  # 最大值
        self.conf_spin_box.setSingleStep(0.01)  # 步长
        self.conf_spin_box.setValue(self.identify_api.conf_thres)  # 当前值
        self.conf_spin_box.setStyleSheet("QDoubleSpinBox{background:#ffffff;color:#999999;font-size:14px;"
                                         "font-weight:600;border: 1px solid #dddddd;}")
        self.conf_spin_box.valueChanged.connect(self.change_conf_spin_box)  # 绑定函数
        # 置信度阈值滚动条
        self.conf_slider = QSlider(Qt.Horizontal, self.left_widget)
        self.conf_slider.resize(130, 25)
        self.conf_slider.move(65, 130)
        self.conf_slider.setMinimum(0)  # 最小值
        self.conf_slider.setMaximum(100)  # 最大值
        self.conf_slider.setSingleStep(1)  # 步长
        self.conf_slider.setValue(int(self.identify_api.conf_thres * 100))  # 当前值
        self.conf_slider.setCursor(Qt.PointingHandCursor)  # 鼠标光标变为手指
        self.conf_slider.setStyleSheet("QSlider::groove:horizontal{border:1px solid #999999;height:25px;}"
                                       "QSlider::handle:horizontal{background:#ffcc00;width:24px;border-radius:12px;}"
                                       "QSlider::add-page:horizontal{background:qlineargradient(spread:pad,x1:0,y1:1,"
                                       "x2:0,y2:0,stop:0 #d9d9d9,stop:0.25 #d9d9d9,stop:0.5 #d9d9d9,stop:1 #d9d9d9);}"
                                       "QSlider::sub-page:horizontal{background:qlineargradient(spread:pad,x1:0,y1:1,"
                                       "x2:0,y2:0,stop:0 #777777,stop:0.25 #777777,stop:0.5 #777777,stop:1 #777777);}")
        self.conf_slider.valueChanged.connect(self.change_conf_slider)  # 绑定函数
        # 交并比IoU阈值文本框
        self.iou_label = QLabel(self.left_widget)
        self.iou_label.setText("交并比IoU")
        self.iou_label.setFixedSize(190, 30)
        self.iou_label.move(5, 160)
        self.iou_label.setStyleSheet("QLabel{font-size: 18px;color:#999999;font-weight:600;font-family:'微软雅黑'; }")
        # 交并比IoU阈值调节框
        self.iou_spin_box = QDoubleSpinBox(self.left_widget)
        self.iou_spin_box.resize(55, 25)
        self.iou_spin_box.move(5, 195)
        self.iou_spin_box.setMinimum(0.0)  # 最小值
        self.iou_spin_box.setMaximum(1.0)  # 最大值
        self.iou_spin_box.setSingleStep(0.01)  # 步长
        self.iou_spin_box.setValue(self.identify_api.iou_thres)  # 当前值
        self.iou_spin_box.setStyleSheet("QDoubleSpinBox{background:#ffffff;color:#999999;font-size:14px;"
                                        "font-weight:600;border: 1px solid #dddddd;}")
        self.iou_spin_box.valueChanged.connect(self.change_iou_spin_box)  # 绑定函数
        # 交并比IoU阈值滚动条
        self.iou_slider = QSlider(Qt.Horizontal, self.left_widget)
        self.iou_slider.resize(130, 25)
        self.iou_slider.move(65, 195)
        self.iou_slider.setMinimum(0)  # 最小值
        self.iou_slider.setMaximum(100)  # 最大值
        self.iou_slider.setSingleStep(1)  # 步长
        self.iou_slider.setValue(int(self.identify_api.iou_thres * 100))  # 当前值
        self.iou_slider.setCursor(Qt.PointingHandCursor)  # 鼠标光标变为手指
        self.iou_slider.setStyleSheet("QSlider::groove:horizontal{border:1px solid #999999;height:25px;}"
                                      "QSlider::handle:horizontal{background:#ffcc00;width:24px;border-radius:12px;}"
                                      "QSlider::add-page:horizontal{background:qlineargradient(spread:pad,x1:0,y1:1,"
                                      "x2:0,y2:0,stop:0 #d9d9d9,stop:0.25 #d9d9d9,stop:0.5 #d9d9d9,stop:1 #d9d9d9);}"
                                      "QSlider::sub-page:horizontal{background:qlineargradient(spread:pad,x1:0,y1:1,"
                                      "x2:0,y2:0,stop:0 #777777,stop:0.25 #777777,stop:0.5 #777777,stop:1 #777777);}")
        self.iou_slider.valueChanged.connect(self.change_iou_slider)  # 绑定函数
        # 保存检测结果文本框
        self.save_label = QLabel(self.left_widget)
        self.save_label.setText("是否保存检测结果")
        self.save_label.resize(190, 25)
        self.save_label.move(5, 225)
        self.save_label.setStyleSheet("QLabel{font-size: 18px;color:#999999;font-weight:600;font-family:'微软雅黑'; }")
        # 保存检测结果单选框
        self.save_button_yes = QRadioButton(self.left_widget)
        self.save_button_yes.setText("  是")
        self.save_button_yes.resize(70, 25)
        self.save_button_yes.move(5, 255)
        self.save_button_yes.setCursor(Qt.PointingHandCursor)  # 鼠标光标变为手指
        self.save_button_yes.setStyleSheet(
            "QRadioButton{font-size: 16px;color:#999999;font-weight:600;font-family:'黑体'; }")
        self.save_button_no = QRadioButton(self.left_widget)
        self.save_button_no.setText("  否")
        self.save_button_no.resize(70, 25)
        self.save_button_no.move(80, 255)
        self.save_button_no.setCursor(Qt.PointingHandCursor)  # 鼠标光标变为手指
        self.save_button_no.setStyleSheet(
            "QRadioButton{font-size: 16px;color:#999999;font-weight:600;font-family:'黑体'; }")
        self.save_button_no.setChecked(True)  # 默认选中不保存
        # ===== 左侧结果显示区 ===== #
        self.result_label = QLabel(self)
        self.result_label.setText("检 测 结 果")
        self.result_label.setFixedSize(200, 32)
        self.result_label.move(0, 352)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("QLabel{background:#ffffff;color:#999999;border:none;font-weight:600;"
                                        "font-size:18px;font-family:'黑体';}")
        self.result = QLabel(self)
        self.result.setText("")
        self.result.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)
        self.result.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # 设置对齐方式为居上居左
        self.result.setStyleSheet("QLabel{background:#ffffff;color:#999999;border:none;font-weight:600;"
                                  "font-size:15px;font-family:'黑体';padding:5px;}")
        self.scroll_area = QScrollArea(self)
        self.scroll_area.resize(200, 215)
        self.scroll_area.move(0, 385)
        self.scroll_area.setWidget(self.result)
        self.scroll_area.setWidgetResizable(True)  # 设置 QScrollArea 大小可调整
        self.scroll_area.setStyleSheet("QScrollArea{border:none;}")
        # ===== 右侧显示区 ===== #
        # 检测速度显示
        self.identify_v = QLabel(self)
        self.identify_v.setText("检测速度：")
        self.identify_v.setFixedSize(820, 20)
        self.identify_v.move(220, 65)
        self.identify_v.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # 设置对齐方式为居上居左
        self.identify_v.setStyleSheet("QLabel{color:#999999;font-weight:600;font-size:15px;font-family:'黑体';}")
        # 原图显示区域
        self.input_img = QLabel(self)
        self.input_img.setText("输入显示区")
        self.input_img.setFixedSize(400, 400)
        self.input_img.move(220, 85)
        self.input_img.setAlignment(Qt.AlignCenter)
        self.input_img.setStyleSheet("QLabel{border: 2px solid gray;font-size:30px;font-family:'黑体';color:#999999;}")
        # 检测图显示区域
        self.output_img = QLabel(self)
        self.output_img.setText("输出显示区")
        self.output_img.setFixedSize(400, 400)
        self.output_img.move(640, 85)
        self.output_img.setAlignment(Qt.AlignCenter)
        self.output_img.setStyleSheet("QLabel{border: 2px solid gray;font-size:30px;font-family:'黑体';color:#999999;}")
        # ===== 右侧按钮区 ===== #
        # 右侧按钮1(图像检测)
        self.function1 = QPushButton(self)
        self.function1.setText("图 像 检 测")
        self.function1.resize(250, 60)
        self.function1.move(220, 510)
        self.function1.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                     "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                     "QPushButton:hover{background:#e6e6e6;}")
        self.function1.setFocusPolicy(Qt.NoFocus)
        self.function1.setCursor(Qt.PointingHandCursor)
        self.function1.clicked.connect(self.show_image)
        # 右侧按钮2(视频检测)
        self.function2 = QPushButton(self)
        self.function2.setText("开启视频检测")
        self.function2.resize(250, 60)
        self.function2.move(505, 510)
        self.function2.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                     "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                     "QPushButton:hover{background:#e6e6e6;}")
        self.function2.setFocusPolicy(Qt.NoFocus)
        self.function2.setCursor(Qt.PointingHandCursor)
        self.function2.clicked.connect(self.video_identify)
        # 右侧按钮3(摄像头检测)
        self.function3 = QPushButton(self)
        self.function3.setText("开启摄像检测")
        self.function3.resize(250, 60)
        self.function3.move(790, 510)
        self.function3.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                     "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                     "QPushButton:hover{background:#e6e6e6;}")
        self.function3.setFocusPolicy(Qt.NoFocus)
        self.function3.setCursor(Qt.PointingHandCursor)
        self.function3.clicked.connect(self.camera_identify)

    # ================================ 功能函数区 ================================#
    # 显示当前时间
    def update_time(self):
        self.present_time = QDateTime.currentDateTime().toString('yyyy-MM-dd hh:mm:ss')
        self.label.setText(self.ui_title + "     " + self.present_time)

    # 选择模型
    def input_pt(self):
        pt_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型", "./", "*.pt")
        if len(pt_path) > 3:
            pt_flag = self.identify_api.load_pt(pt_path)
            if pt_flag:
                pt_name = os.path.basename(pt_path)
                self.pt_show.setText(pt_name)
                # 创建一个消息框
                msg_box = QMessageBox()
                msg_box.setWindowTitle("提示")
                msg_box.setText("加载模型成功！")
                # 显示消息框
                msg_box.exec_()
            else:
                # 创建一个消息框
                msg_box = QMessageBox()
                msg_box.setWindowTitle("提示")
                msg_box.setText("加载模型失败！")
                # 显示消息框
                msg_box.exec_()

    # 调节框改变检测置信度
    def change_conf_spin_box(self):
        conf_thres = round(self.conf_spin_box.value(), 2)
        self.conf_slider.setValue(int(conf_thres * 100))
        self.identify_api.conf_thres = conf_thres

    # 滚动条改变检测置信度
    def change_conf_slider(self):
        conf_thres = round(self.conf_slider.value() * 0.01, 2)
        self.conf_spin_box.setValue(conf_thres)
        self.identify_api.conf_thres = conf_thres

    # 调节框改变检测交并比
    def change_iou_spin_box(self):
        iou_thres = round(self.iou_spin_box.value(), 2)
        self.iou_slider.setValue(int(iou_thres * 100))
        self.identify_api.iou_thres = iou_thres

    # 滚动条改变检测交并比
    def change_iou_slider(self):
        iou_thres = round(self.iou_slider.value() * 0.01, 2)
        self.iou_spin_box.setValue(iou_thres)
        self.identify_api.iou_thres = iou_thres

    # 图片检测
    def show_image(self):
        if len(self.pt_show.text()) != 0:
            image_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "打开图片", "./", "*.jpg;*.png;;All Files(*)")
            #  选择的图片名必须大于等于5
            if len(image_path) >= 5:
                self.input_image = cv2.imread(image_path)
                start_time = time.time()  # 记录起始时间
                self.input_image, self.output_image, self.identify_labels = self.identify_api.show_frame(self.input_image, False)
                if self.output_image is not None:
                    # ===== 显示界面和结果 ===== #
                    show_input_img = self.change_image(self.input_image)
                    show_output_img = self.change_image(self.output_image)
                    # 将检测图像画面显示在界面
                    show_input_img = cv2.cvtColor(show_input_img, cv2.COLOR_BGR2RGB)
                    show_input_img = QImage(show_input_img.data, show_input_img.shape[1], show_input_img.shape[0],
                                            show_input_img.shape[1] * 3, QImage.Format_RGB888)
                    self.input_img.setPixmap(QPixmap.fromImage(show_input_img))
                    show_output_img = cv2.cvtColor(show_output_img, cv2.COLOR_BGR2RGB)
                    show_output_img = QImage(show_output_img.data, show_output_img.shape[1], show_output_img.shape[0],
                                             show_output_img.shape[1] * 3, QImage.Format_RGB888)
                    self.output_img.setPixmap(QPixmap.fromImage(show_output_img))
                    # 显示检测速度和检测结果在结果显示区
                    end_time = time.time()  # 记录结束时间
                    execution_time = str(round(end_time - start_time, 2)) + " s"
                    self.identify_v.setText("检测速度： " + execution_time)
                    identify_result = ""
                    counter = Counter(self.identify_labels)  # 使用 Counter 统计元素出现次数
                    for element, count in counter.items():
                        identify_result = identify_result + str(element) + ": " + str(count) + "\n"
                    self.result.setText(identify_result)
                    # ===== 保存结果 ===== #
                    if self.save_button_yes.isChecked():
                        file_path = os.path.join(self.save_path, "images/" +
                                                 QDateTime.currentDateTime().toString('yyyy_MM_dd_hh_mm_ss') + ".jpg")
                        cv2.imwrite(file_path, self.output_image)  # 保存图像
                        # 创建一个消息框
                        msg_box = QMessageBox()
                        msg_box.setWindowTitle("提示")
                        msg_box.setText("检测图片已保存在./A_output路径中！")
                        # 显示消息框
                        msg_box.exec_()
                else:
                    self.reset()
            else:
                self.reset()
        else:
            # 创建一个消息框
            msg_box = QMessageBox()
            msg_box.setWindowTitle("提示")
            msg_box.setText("请先导入模型！")
            # 显示消息框
            msg_box.exec_()

    # 视频检测
    def video_identify(self):
        if len(self.pt_show.text()) != 0:
            if self.function2.text() == "开启视频检测" and not self.timer_video.isActive():
                video_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, "打开视频", "", "*.mp4;*.avi;;All Files(*)")
                if len(video_path) > 5:
                    flag = self.identify_api.cap.open(video_path)
                    if flag is False:
                        QtWidgets.QMessageBox.warning(
                            self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok,
                            defaultButton=QtWidgets.QMessageBox.Ok)
                    else:
                        self.timer_video.start(30)
                        self.function1.setDisabled(True)
                        self.function3.setDisabled(True)
                        self.function1.setStyleSheet("QPushButton{background:#e6e6e6;color:#999999;font-weight:600;"
                                                     "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                                     "QPushButton:hover{background:#e6e6e6;}")
                        self.function3.setStyleSheet("QPushButton{background:#e6e6e6;color:#999999;font-weight:600;"
                                                     "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                                     "QPushButton:hover{background:#e6e6e6;}")
                        self.function2.setText("关闭视频检测")
                else:
                    self.reset()
            else:
                self.identify_api.cap.release()
                self.timer_video.stop()
                self.function1.setDisabled(False)
                self.function3.setDisabled(False)
                self.function1.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                             "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                             "QPushButton:hover{background:#e6e6e6;}")
                self.function3.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                             "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                             "QPushButton:hover{background:#e6e6e6;}")
                self.function2.setText("开启视频检测")
                self.reset()
        else:
            # 创建一个消息框
            msg_box = QMessageBox()
            msg_box.setWindowTitle("提示")
            msg_box.setText("请先导入模型！")
            # 显示消息框
            msg_box.exec_()

    # 摄像头检测
    def camera_identify(self):
        if len(self.pt_show.text()) != 0:
            if self.function3.text() == "开启摄像检测" and not self.timer_video.isActive():
                # 默认使用第一个本地camera
                flag = self.identify_api.cap.open(0)
                if flag is False:
                    QtWidgets.QMessageBox.warning(
                        self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok,
                        defaultButton=QtWidgets.QMessageBox.Ok)
                else:
                    self.timer_video.start(30)
                    self.function1.setDisabled(True)
                    self.function2.setDisabled(True)
                    self.function1.setStyleSheet("QPushButton{background:#e6e6e6;color:#999999;font-weight:600;"
                                                 "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                                 "QPushButton:hover{background:#e6e6e6;}")
                    self.function2.setStyleSheet("QPushButton{background:#e6e6e6;color:#999999;font-weight:600;"
                                                 "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                                 "QPushButton:hover{background:#e6e6e6;}")
                    self.function3.setText("关闭摄像检测")
            else:
                self.identify_api.cap.release()
                self.timer_video.stop()
                self.function1.setDisabled(False)
                self.function2.setDisabled(False)
                self.function1.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                             "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                             "QPushButton:hover{background:#e6e6e6;}")
                self.function2.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                             "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                             "QPushButton:hover{background:#e6e6e6;}")
                self.function3.setText("开启摄像检测")
                self.reset()
        else:
            # 创建一个消息框
            msg_box = QMessageBox()
            msg_box.setWindowTitle("提示")
            msg_box.setText("请先导入模型！")
            # 显示消息框
            msg_box.exec_()

    # 展示图像与显示结果(视频与摄像头)
    def show_video(self):
        start_time = time.time()  # 记录起始时间
        self.input_image, self.output_image, self.identify_labels = self.identify_api.show_frame(None, True)
        if self.output_image is not None:
            # ===== 保存结果 ===== #
            if self.save_button_yes.isChecked():
                # 保存视频
                if self.save_video_flag is False:
                    self.save_video_flag = True
                    fps = self.identify_api.cap.get(cv2.CAP_PROP_FPS)
                    w = int(self.identify_api.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(self.identify_api.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if self.function2.text() == "关闭视频检测":
                        save_path = self.save_path + "/videos/saved_" + QDateTime.currentDateTime().toString(
                            'yyyy_MM_dd_hh_mm_ss') + ".mp4"
                    else:
                        save_path = self.save_path + "/camera/saved_" + QDateTime.currentDateTime().toString(
                            'yyyy_MM_dd_hh_mm_ss') + ".mp4"
                    self.output_video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                self.output_video.write(self.output_image)
            # ===== 显示界面和结果 ===== #
            show_input_img = self.change_image(self.input_image)
            show_output_img = self.change_image(self.output_image)
            # 将检测图像画面显示在界面
            show_input_img = cv2.cvtColor(show_input_img, cv2.COLOR_BGR2RGB)
            show_input_img = QImage(show_input_img.data, show_input_img.shape[1], show_input_img.shape[0],
                                    show_input_img.shape[1] * 3, QImage.Format_RGB888)
            self.input_img.setPixmap(QPixmap.fromImage(show_input_img))
            show_output_img = cv2.cvtColor(show_output_img, cv2.COLOR_BGR2RGB)
            show_output_img = QImage(show_output_img.data, show_output_img.shape[1], show_output_img.shape[0],
                                     show_output_img.shape[1] * 3, QImage.Format_RGB888)
            self.output_img.setPixmap(QPixmap.fromImage(show_output_img))
            # 显示检测速度和检测结果在结果显示区
            end_time = time.time()  # 记录结束时间
            execution_time = str(round(end_time - start_time, 2)) + " s"
            self.identify_v.setText("检测速度： " + execution_time)
            identify_result = ""
            counter = Counter(self.identify_labels)  # 使用 Counter 统计元素出现次数
            for element, count in counter.items():
                identify_result = identify_result + str(element) + ": " + str(count) + "\n"
            self.result.setText(identify_result)
        else:
            self.timer_video.stop()
            self.function1.setDisabled(False)
            self.function2.setDisabled(False)
            self.function3.setDisabled(False)
            self.function1.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                         "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                         "QPushButton:hover{background:#e6e6e6;}")
            self.function2.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                         "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                         "QPushButton:hover{background:#e6e6e6;}")
            self.function3.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                         "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                         "QPushButton:hover{background:#e6e6e6;}")
            self.function2.setText("开启视频检测")
            self.function3.setText("开启摄像检测")
            self.reset()

    # 改变图像大小在界面显示
    @staticmethod
    def change_image(input_image):
        if input_image is not None:
            # 更换为界面适应性大小显示
            wh = float(int(input_image.shape[0]) / int(input_image.shape[1]))
            show_wh = 1
            if int(input_image.shape[0]) > 400 or int(input_image.shape[1]) > 400:
                if show_wh - wh < 0:
                    h = 400
                    w = int(h / wh)
                    output_image = cv2.resize(input_image, (w, h))
                else:
                    w = 400
                    h = int(w * wh)
                    output_image = cv2.resize(input_image, (w, h))
            else:
                output_image = input_image
            return output_image
        else:
            return input_image

    # 清空重置数据
    def reset(self):
        if self.save_button_yes.isChecked() and self.output_video is not None:
            self.output_video.release()
            # 创建一个消息框
            msg_box = QMessageBox()
            msg_box.setWindowTitle("提示")
            msg_box.setText("检测结果已保存在./A_output路径中！")
            # 显示消息框
            msg_box.exec_()
        self.input_image = None  # 输入图像
        self.output_image = None  # 输出图像
        self.output_video = None  # 输出视频
        self.identify_labels = []  # 检测结果
        self.save_video_flag = False  # 保存视频标志位
        self.input_img.clear()  # 清空输入图像显示区
        self.input_img.setText("输入显示区")
        self.output_img.clear()  # 清空输出图像显示区
        self.output_img.setText("输出显示区")
        self.identify_v.setText("检测速度：")
        self.result.clear()  # 清空结果显示区
        self.save_button_no.setChecked(True)  # 选中不保存


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainUi = MainUi()
    mainUi.show()
    sys.exit(app.exec_())
