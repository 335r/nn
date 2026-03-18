import streamlit as st
import cv2
import torch
import numpy as np
import time
import json
import serial
import queue
import logging
import threading
import base64
import requests
from pathlib import Path
from collections import deque
from fastapi import FastAPI, APIRouter, Body, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, letterbox
from utils.torch_utils import select_device
import random
import uvicorn
from packaging import version

# ---------------------- 版本检查 ----------------------
ST_REQUIRED = "1.13.0"
if version.parse(st.__version__) < version.parse(ST_REQUIRED):
    st.error(f"需要 Streamlit {ST_REQUIRED}+ 版本，当前版本 {st.__version__}")
    st.stop()

# ---------------------- 配置管理 ----------------------
CONFIG_FILE = "app_config.json"
DEFAULT_CONFIG = {
    "video_source": 0,
    "serial_port": "COM3",
    "baudrate": 115200,
    "img_size": 640,
    "conf_thres": 0.25,
    "iou_thres": 0.45,
    "device": "cuda:0",
    "classes": None,
    "agnostic_nms": False,
    "augment": False,
    "max_reconnect_attempts": 5,
    "target_fps": 30,
    "frame_size": [640, 480],
    "serial_enabled": True
}


# ---------------------- API 相关组件 ----------------------
class DetectionRequest(BaseModel):
    image_url: str = None
    image_base64: str = None


class DetectionResultAPI(BaseModel):
    class_name: str
    confidence: float
    bbox: List[int]


def select_device(device_str=''):
    if device_str.lower() == 'cpu':
        return torch.device('cpu')
    elif torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        logging.warning("CUDA不可用，自动回退到CPU")
        return torch.device('cpu')


# ---------------------- 摄像头控制器 ----------------------
class CameraController:
    def __init__(self, config):
        self.config = config
        self.cap = None
        self._frame_buffer = deque(maxlen=3)  # 添加帧缓冲区
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._init_camera()
        self._start_capture_thread()

    def _init_camera(self):
        if self.cap is not None:
            self.cap.release()

        try:
            self.cap = cv2.VideoCapture(self.config["video_source"])
            if not self.cap.isOpened():
                logging.error(f"无法打开视频源: {self.config['video_source']}")
                raise RuntimeError("无法打开视频源")

            # 设置摄像头参数以优化性能
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["frame_size"][0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["frame_size"][1])
            self.cap.set(cv2.CAP_PROP_FPS, self.config["target_fps"])
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲区大小

            # 尝试设置MJPG格式以提高帧率（如果摄像头支持）
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            logging.info(f"摄像头初始化完成，分辨率: {actual_width}x{actual_height}, FPS: {actual_fps}")

        except Exception as e:
            logging.error(f"摄像头初始化失败: {str(e)}")
            # 尝试使用默认参数重新初始化
            try:
                self.cap = cv2.VideoCapture(self.config["video_source"])
                if not self.cap.isOpened():
                    raise RuntimeError("无法打开视频源")
            except:
                raise RuntimeError(f"摄像头初始化完全失败: {str(e)}")

    def _capture_frames(self):
        """在后台线程中持续捕获帧"""
        while self._running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    with self._lock:
                        self._frame_buffer.append(frame)
                else:
                    logging.warning("获取视频帧失败，尝试重新初始化摄像头")
                    self._init_camera()
                    time.sleep(0.1)  # 短暂等待后重试
            except Exception as e:
                logging.error(f"捕获帧时出错: {str(e)}")
                time.sleep(0.1)

    def _start_capture_thread(self):
        """启动后台帧捕获线程"""
        self._running = True
        self._thread = threading.Thread(target=self._capture_frames, daemon=True)
        self._thread.start()
        # 等待缓冲区有数据
        for _ in range(10):
            if len(self._frame_buffer) > 0:
                break
            time.sleep(0.1)

    def read(self):
        """获取最新帧"""
        retry_count = 0
        max_retries = 5

        while retry_count < max_retries:
            with self._lock:
                if len(self._frame_buffer) > 0:
                    # 获取最新帧并清空旧帧
                    frame = self._frame_buffer[-1]
                    self._frame_buffer.clear()
                    return frame

            # 如果没有可用帧，等待一小段时间
            time.sleep(0.01)
            retry_count += 1

        logging.error("连续多次获取视频帧失败")
        raise RuntimeError("无法获取视频帧")

    def release(self):
        """释放资源"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self.cap and self.cap.isOpened():
            self.cap.release()
            logging.info("摄像头资源已释放")


# ---------------------- 串口通信组件 ----------------------
class RobustSerialReader:
    def __init__(self, config):
        self.config = config
        self.serial = None
        self._data_queue = queue.Queue(maxsize=20)
        self._event = threading.Event()
        self._event.set()
        self._init_serial()
        self.thread = None

    def _init_serial(self):
        if not self.config["serial_enabled"]:
            return

        for attempt in range(self.config["max_reconnect_attempts"]):
            try:
                self.serial = serial.Serial(
                    port=self.config["serial_port"],
                    baudrate=self.config["baudrate"],
                    timeout=1
                )
                logging.info(f"串口连接成功: {self.config['serial_port']}")
                return
            except Exception as e:
                logging.error(f"串口连接失败 [{attempt + 1}/{self.config['max_reconnect_attempts']}]: {str(e)}")
                time.sleep(1)
        logging.error("达到最大重试次数，放弃串口连接")

    def _read_loop(self):
        buffer = ""
        while self._event.is_set():
            if self.serial is None:
                time.sleep(5)
                self._init_serial()
                continue

            try:
                raw = self.serial.read_all().decode(errors='ignore')
                if raw:
                    buffer += raw
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        self._process_line(line.strip())
            except Exception as e:
                logging.error(f"串口读取错误: {str(e)}")

    def _process_line(self, line):
        try:
            data = json.loads(line)
            if "detections" in data:
                self._data_queue.put(data["detections"])
        except json.JSONDecodeError:
            logging.warning(f"无效JSON数据: {line}")

    def get_detections(self):
        detections = []
        while not self._data_queue.empty():
            try:
                detections.append(self._data_queue.get_nowait())
            except queue.Empty:
                break
        return detections

    def start(self):
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self._event.clear()
        if self.serial and self.serial.is_open:
            self.serial.close()


# ---------------------- YOLOv7 检测器 ----------------------
class YOLOv7Detector:
    def __init__(self, config):
        self.config = config
        self.device = select_device(config["device"])
        self.model = attempt_load(r"E:\Git\网页\VT\best.pt", map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(config["img_size"], s=self.stride)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        if self.device.type != 'cpu':
            self.model.half()
            self.model(
                torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))

    def detect(self, img):
        img0 = img.copy()
        img = letterbox(img, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.device.type != 'cpu' else img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img, augment=self.config["augment"])[0]

        pred = non_max_suppression(
            pred,
            self.config["conf_thres"],
            self.config["iou_thres"],
            classes=self.config["classes"],
            agnostic=self.config["agnostic_nms"]
        )

        results = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                results.append(det.cpu().numpy())
        return results


# ---------------------- 辅助函数 ----------------------
def load_config():
    try:
        if Path(CONFIG_FILE).exists():
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"配置加载失败: {str(e)}")
    return DEFAULT_CONFIG.copy()


def save_config(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        logging.error(f"配置保存失败: {str(e)}")


def get_camera(config):
    return CameraController(config)


def load_yolov7_model(config):
    device = select_device(config["device"])
    model = attempt_load(r"E:\Git\网页\VT\best.pt", map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(config["img_size"], s=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != 'cpu':
        model.half()
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    return model, device, stride, imgsz, names, colors


# ---------------------- 应用主逻辑 ----------------------
class VideoAnalyticsApp:
    def __init__(self):
        self.config = load_config()
        self.frame_placeholder = st.empty()
        self.status_bar = st.empty()
        self.serial_reader = None
        self.camera = get_camera(self.config)
        model, device, stride, imgsz, names, colors = load_yolov7_model(self.config)
        self.detector = YOLOv7Detector(self.config)
        self._init_serial()
        self.api_thread = None
        self.detection_history = deque(maxlen=100)  # 检测历史记录
        self._start_api_server()

        # 添加缓存机制
        self._last_frame = None
        self._last_frame_time = 0

    def _init_serial(self):
        if self.config["serial_enabled"]:
            self.serial_reader = RobustSerialReader(self.config)
            self.serial_reader.start()

    def _start_api_server(self):
        def run_api():
            @api_app.post("/api/detect", response_model=List[DetectionResultAPI])
            async def api_detect(detection_request: DetectionRequest):
                try:
                    img = None
                    if detection_request.image_url:
                        resp = requests.get(detection_request.image_url)
                        img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
                    elif detection_request.image_base64:
                        img_data = base64.b64decode(detection_request.image_base64.split(",")[1])
                        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

                    if img is None:
                        return JSONResponse(status_code=400, content={"message": "无效图像数据"})

                    results = self.detector.detect(img)
                    return [
                        DetectionResultAPI(
                            class_name=self.detector.names[int(cls)] if int(cls) < len(
                                self.detector.names) else "unknown",
                            confidence=float(conf),
                            bbox=[int(x) for x in xyxy]
                        )
                        for det in results
                        for *xyxy, conf, cls in det
                    ]
                except Exception as e:
                    logging.exception("API错误")
                    return JSONResponse(status_code=500, content={"message": str(e)})

            config = uvicorn.Config(api_app, host="0.0.0.0", port=8001)
            server = uvicorn.Server(config)
            server.run()

        self.api_thread = threading.Thread(target=run_api, daemon=True)
        self.api_thread.start()

    def run(self):
        st.title("YOLOv7 实时目标检测")

        with st.sidebar:
            st.markdown("### 控制面板")
            cols = st.columns(2)
            with cols[0]:
                if st.button("▶️ 开始检测", key="start"):
                    st.session_state.running = True
            with cols[1]:
                if st.button("⏹️ 停止检测", key="stop"):
                    st.session_state.running = False

            st.markdown("---")
            st.markdown("### 参数设置")
            new_fps = st.slider("目标帧率 (FPS)",
                                min_value=1,
                                max_value=60,
                                value=self.config["target_fps"],
                                key="fps_control")
            if new_fps != self.config["target_fps"]:
                self.config["target_fps"] = new_fps
                save_config(self.config)

            st.markdown("---")
            st.markdown("### API文档")
            st.markdown("[访问API文档](http://localhost:8001/docs)")

            # 新增生成报告按钮
            st.markdown("---")
            if st.button("📊 生成检测报告"):
                st.session_state.show_report = True

        # 显示检测报告
        if st.session_state.get("show_report", False):
            self._show_detection_report()

        # 图片上传检测模块
        st.markdown("---")
        st.subheader("图片检测功能")
        uploaded_file = st.file_uploader(
            "上传图片（支持JPG/JPEG/PNG）",
            type=["jpg", "jpeg", "png"],
            key="file_uploader"
        )

        if uploaded_file is not None:
            with st.spinner("正在检测上传图片..."):
                try:
                    # 转换上传文件为OpenCV格式
                    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                    upload_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    # 执行目标检测
                    results = self.detector.detect(upload_img)

                    # 标注并显示结果
                    annotated_img = self._annotate_frame(upload_img.copy(), results)
                    st.image(annotated_img,
                             channels="BGR",
                             caption="上传图片检测结果",
                             use_container_width=True)

                    # 添加历史记录
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    for det in results:
                        for *xyxy, conf, cls in det:
                            self.detection_history.append({
                                "timestamp": timestamp,
                                "class": self.detector.names[int(cls)],
                                "confidence": float(conf),
                                "source": "图片上传",
                                "image_size": f"{upload_img.shape[1]}x{upload_img.shape[0]}"
                            })

                except Exception as e:
                    st.error(f"图片处理失败: {str(e)}")

        # 实时视频检测模块
        if st.session_state.get("running", False):
            start_time = time.time()
            try:
                frame = self.camera.read()
            except RuntimeError as e:
                st.error(f"摄像头错误: {str(e)}")
                st.session_state.running = False
                return

            # 添加帧率控制
            current_time = time.time()
            if current_time - self._last_frame_time < 1.0 / self.config["target_fps"]:
                # 使用上一帧
                frame = self._last_frame
            else:
                # 更新帧
                self._last_frame = frame
                self._last_frame_time = current_time

            processing_start = time.time()
            results = self.detector.detect(frame)
            processing_time = time.time() - processing_start

            # 添加实时检测记录
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            for det in results:
                for *xyxy, conf, cls in det:
                    self.detection_history.append({
                        "timestamp": timestamp,
                        "class": self.detector.names[int(cls)],
                        "confidence": float(conf),
                        "source": "实时检测",
                        "image_size": f"{frame.shape[1]}x{frame.shape[0]}"
                    })

            annotated_frame = self._annotate_frame(frame, results)
            _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

            self.frame_placeholder.image(
                buffer.tobytes(),
                channels="BGR",
                caption=f"实时检测画面 | 延迟: {processing_time * 1000:.1f}ms",
                use_container_width=True
            )

            elapsed_time = time.time() - start_time
            target_delay = max(0, (1 / self.config["target_fps"]) - elapsed_time)
            time.sleep(target_delay)

            st.rerun()

    def _annotate_frame(self, frame, results):
        for det in results:
            for *xyxy, conf, cls in det:
                label = f"{self.detector.names[int(cls)]} {conf:.2f}"
                color = self.detector.colors[int(cls)]
                cv2.rectangle(frame,
                              (int(xyxy[0]), int(xyxy[1])),
                              (int(xyxy[2]), int(xyxy[3])),
                              color, 2)
                cv2.putText(frame, label,
                            (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)
        return frame

    def _show_detection_report(self):
        st.subheader("检测分析报告")

        if not self.detection_history:
            st.warning("尚未有任何检测记录")
            return

        # 统计数据分析
        class_counts = {}
        confidences = {}
        sources = {}
        for entry in self.detection_history:
            cls = entry["class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1
            if cls not in confidences:
                confidences[cls] = []
            confidences[cls].append(entry["confidence"])
            sources[entry["source"]] = sources.get(entry["source"], 0) + 1

        # 创建布局
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📈 检测统计概览")
            st.markdown(f"**总检测次数:** {len(self.detection_history)}")
            st.markdown(f"**检测类别数:** {len(class_counts)}")
            st.markdown(f"**最高频类别:** {max(class_counts, key=class_counts.get, default='无')}")

            st.markdown("### 📍 检测来源分布")
            if sources:
                source_chart = {
                    "检测类型": list(sources.keys()),
                    "次数": list(sources.values())
                }
                st.bar_chart(source_chart, x="检测类型", y="次数")

        with col2:
            st.markdown("### 🏷️ 类别分布")
            if class_counts:
                class_data = {
                    "类别": list(class_counts.keys()),
                    "出现次数": list(class_counts.values())
                }
                st.bar_chart(class_data, x="类别", y="出现次数")

        # 置信度分析
        st.markdown("### 📊 置信度分析")
        conf_df = []
        for cls, vals in confidences.items():
            conf_df.append({
                "类别": cls,
                "平均置信度": f"{np.mean(vals):.2%}",
                "最高置信度": f"{max(vals):.2%}",
                "最低置信度": f"{min(vals):.2%}"
            })
        if conf_df:
            st.table(conf_df)
        else:
            st.write("暂无置信度数据")

        # 原始数据展示
        st.markdown("### 📋 原始检测记录")
        show_count = st.slider("选择显示记录数量", 1, 100, 10)
        st.table(list(self.detection_history)[-show_count:])

        # 报告下载功能
        st.markdown("---")
        st.markdown("### 📩 下载完整报告")
        report_content = self._generate_report_content(class_counts, confidences, sources)
        st.download_button(
            label="下载文本报告",
            data=report_content,
            file_name=f"detection_report_{time.strftime('%Y%m%d%H%M%S')}.txt",
            mime="text/plain"
        )

    def _generate_report_content(self, class_counts, confidences, sources):
        """生成文本格式的报告内容"""
        report = []
        report.append("=" * 40)
        report.append(f"检测分析报告生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"总检测记录数：{len(self.detection_history)}")
        report.append("\n--- 统计概览 ---")

        report.append("\n【类别分布】")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  - {cls}: {count} 次 ({count / len(self.detection_history):.1%})")

        report.append("\n【检测来源】")
        for src, count in sources.items():
            report.append(f"  - {src}: {count} 次")

        report.append("\n--- 置信度分析 ---")
        for cls in confidences:
            avg = np.mean(confidences[cls])
            max_c = max(confidences[cls])
            min_c = min(confidences[cls])
            report.append(
                f"{cls}: 平均 {avg:.2%}, 最高 {max_c:.2%}, 最低 {min_c:.2%}"
            )

        report.append("\n--- 原始数据样本 ---")
        for entry in list(self.detection_history)[-10:]:
            report.append(
                f"[{entry['timestamp']}] {entry['class']} "
                f"(置信度: {entry['confidence']:.2%}, 来源: {entry['source']})"
            )

        return "\n".join(report)

    def shutdown(self):
        if self.camera:
            self.camera.release()
        if self.serial_reader:
            self.serial_reader.stop()


# ---------------------- 应用入口 ----------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
    )

    if 'app' not in st.session_state:
        st.session_state.app = VideoAnalyticsApp()

    app = st.session_state.app
    try:
        app.run()
    finally:
        app.shutdown()