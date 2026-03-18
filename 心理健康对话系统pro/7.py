# -*- coding: utf-8 -*-  # 指定文件编码为UTF-8，支持中文显示

# 导入所需的库
import os  # 用于文件和目录操作
import re  # 用于正则表达式处理
import cv2  # 用于图像处理和摄像头操作
import time  # 用于时间相关操作
import json  # 用于JSON数据处理
import logging  # 用于日志记录
import numpy as np  # 用于数值计算
from PIL import Image  # 用于图像处理
from uuid import uuid4  # 用于生成唯一会话ID
from typing import Optional, Dict, Any, List  # 用于类型提示
from datetime import datetime  # 用于日期时间处理
from logging.handlers import RotatingFileHandler  # 用于日志轮转
import tempfile  # 用于创建临时文件
import requests  # 用于HTTP请求
import streamlit as st  # 用于构建Web应用
from ultralytics import YOLO  # 用于加载YOLOv8模型进行情绪检测
from gtts import gTTS  # 语音合成库，用于将文本转换为语音
import streamlit.components.v1 as components  # 用于在Streamlit中嵌入HTML组件，播放音频

# ================== 页面基础设置 ==================
st.set_page_config(
    page_title="心理咨询 + 实时情绪检测（YOLOv8）",  # 页面标题
    page_icon="🧠",  # 页面图标
    layout="wide"  # 页面布局为宽屏
)

# 隐藏 Streamlit 默认 UI
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}  # 隐藏主菜单
    footer {visibility: hidden;}  # 隐藏页脚
    header {visibility: visible;}  # 显示页眉
    </style>
    """,
    unsafe_allow_html=True,  # 允许使用HTML
)


# ================== 全局会话状态初始化 ==================
def _ensure_session_keys():
    # —— YOLO 检测相关
    st.session_state.setdefault("yolo_model", None)  # YOLO模型实例
    st.session_state.setdefault("yolo_started", False)  # YOLO检测是否已启动
    st.session_state.setdefault("yolo_stop_flag", [False])  # YOLO检测停止标志（使用列表实现可变对象）
    st.session_state.setdefault("yolo_log_file", None)  # YOLO检测日志文件路径
    st.session_state.setdefault("yolo_last_emotion", None)  # 最后检测到的情绪
    st.session_state.setdefault("yolo_last_emotion_list", [])  # 最后检测到的情绪列表
    # —— 咨询对话相关
    st.session_state.setdefault("session_id", str(uuid4()))  # 会话唯一ID
    st.session_state.setdefault("history", [])  # 对话历史，格式为[{role, content}]
    st.session_state.setdefault("assess",  # 评估结果
                                {"emotion": 0, "stress": 0, "sleep": 0, "risk": 0, "risk_terms": [], "name": None})
    st.session_state.setdefault("started", False)  # 对话是否已开始
    # —— 语音相关
    st.session_state.setdefault("audio_enabled", True)  # 默认开启语音播报


_ensure_session_keys()  # 调用函数初始化会话状态

# ================== 日志（复用原 app 的日志方案） ==================
APP_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前应用所在目录
LOG_DIR = os.path.join(APP_DIR, "logs")  # 日志目录路径
os.makedirs(LOG_DIR, exist_ok=True)  # 创建日志目录，若已存在则不报错

# 主日志配置 - 每次运行覆盖原有日志
logger = logging.getLogger("assessment_logger")  # 创建主日志记录器
if not logger.handlers:  # 若日志处理器未初始化
    handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "assessment_badges.log"),  # 日志文件路径
        mode='w',  # 每次运行覆盖原有内容
        encoding="utf-8",  # 编码为UTF-8
    )
    handler.setFormatter(logging.Formatter("%(asctime)s\t%(message)s"))  # 设置日志格式
    logger.addHandler(handler)  # 添加处理器
    logger.setLevel(logging.INFO)  # 设置日志级别为INFO

# 情绪检测专用日志配置 - 每次运行覆盖原有日志
emotion_logger = logging.getLogger("emotion_detection_logger")  # 创建情绪检测日志记录器
if not emotion_logger.handlers:  # 若日志处理器未初始化
    emotion_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "emotion_detections.log"),  # 日志文件路径
        mode='w',  # 每次运行覆盖原有内容
        encoding="utf-8",  # 编码为UTF-8
    )
    emotion_formatter = logging.Formatter(  # 设置日志格式
        "%(asctime)s\t%(session_id)s\t%(fps)d\t%(face_count)d\t%(emotions)s"
    )
    emotion_handler.setFormatter(emotion_formatter)  # 设置格式
    emotion_logger.addHandler(emotion_handler)  # 添加处理器
    emotion_logger.setLevel(logging.INFO)  # 设置日志级别为INFO

# ================== 通义千问 API 配置 ==================
TONGYI_API_KEY = "sk-f9e932582fd94d5e93b43bc8e74e1e35"  # 通义千问API密钥（请妥善保管）
TONGYI_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"  # 通义千问API地址

MODEL_NAME = "qwen-turbo"  # 使用的模型名称，可换为qwen-plus / qwen-max（视账号权限）

if not TONGYI_API_KEY:  # 若未设置API密钥，显示警告
    st.warning("未检测到通义千问 API 密钥。请在 .streamlit/secrets.toml 或环境变量中设置 TONGYI_API_KEY。")

# ================== 心理咨询系统提示词 ==================
PSYCHOLOGIST_SYSTEM_PROMPT = """
你是一名温暖、真诚、专业的中文心理咨询师。目标：在人性化对话中建立联结、进行初步评估并提供有边界的支持。
【专业边界】不能替代线下医疗或急救；出现风险信号时优先进行安全评估与转介；对医学诊断与处方保持克制。
【对话风格】自然口语化、同理、具体、少术语；优先提问引导；总结来访者表达；必要时结构化。
【评估四维】情绪/压力/睡眠/风险。对强风险信号（自伤/他伤/轻生）先做安全评估与转介，避免详细方法。
【开场基调】肯定求助勇气；给出保密/例外（危机时需联系紧急资源）的简短说明；邀请用喜欢的称呼并从舒适的话题开始。
【进行中】每 1-2 轮：小结+澄清；必要时提出具体、可执行的小练习；避免一次给太多任务。
【结束时】总结要点，建议下次可讨论的焦点；再次提示在紧急时联系当地危机热线或急救。

若用户希望明确咨询类型（情绪/压力/人际/筛查/危机/成长），自然承接并围绕该方向提问，不生硬切换话题。
"""  # 定义心理咨询师的系统提示词，指导AI的行为和风格

# ================== 评估器（关键词 + 启发式） ==================
CRISIS_KEYWORDS = [  # 危机关键词列表
    "自杀", "轻生", "结束生命", "不想活", "伤害自己", "割腕", "跳楼", "安眠药过量",
    "杀人", "报复", "伤害别人",
]
EMOTION_KEYWORDS = ["难过", "低落", "沮丧", "焦虑", "紧张", "恐慌", "崩溃", "空虚", "孤独", "愤怒", "烦躁"]  # 情绪关键词列表
STRESS_KEYWORDS = ["压力", "加班", "绩效", "deadline", "考试", "学业", "经济", "房贷", "催促", "内耗"]  # 压力关键词列表
SLEEP_KEYWORDS = ["失眠", "睡不着", "早醒", "噩梦", "多梦", "嗜睡", "睡眠差", "睡眠不好"]  # 睡眠关键词列表


def _count_hits(text: str, vocab) -> (int, list):
    """统计文本中出现的关键词数量及列表"""
    cnt = 0  # 计数器
    hits = []  # 命中的关键词列表
    for w in vocab:  # 遍历词汇表
        if w in text:  # 若关键词在文本中出现
            cnt += text.count(w)  # 累加出现次数
            hits.append(w)  # 添加到命中列表
    return cnt, hits  # 返回数量和列表


def assess_incremental(user_text: str, prior_state: Dict[str, Any]):
    """基于用户输入文本增量更新评估状态"""
    state = prior_state.copy()  # 复制之前的状态，避免修改原对象
    # 确保状态中存在必要的键
    state.setdefault("emotion", 0)
    state.setdefault("stress", 0)
    state.setdefault("sleep", 0)
    state.setdefault("risk", 0)
    state.setdefault("risk_terms", [])
    state.setdefault("name", None)

    # 统计各类关键词的命中情况
    e_cnt, e_hits = _count_hits(user_text, EMOTION_KEYWORDS)
    s_cnt, s_hits = _count_hits(user_text, STRESS_KEYWORDS)
    z_cnt, z_hits = _count_hits(user_text, SLEEP_KEYWORDS)
    r_cnt, r_hits = _count_hits(user_text, CRISIS_KEYWORDS)

    # 更新状态
    state["emotion"] += e_cnt
    state["stress"] += s_cnt
    state["sleep"] += z_cnt
    state["risk"] += r_cnt
    state["risk_terms"].extend(r_hits)

    # 从用户输入中提取姓名（若未提取过）
    name_match = re.search(r"(我叫|我名字是|我的名字是|可以叫我)([A-Za-z0-9\\u4e00-\\u9fa5]{1,12})", user_text)
    if name_match and not state.get("name"):
        state["name"] = name_match.group(2)

    # 生成总结信息
    summary = {
        "emotion_hits": e_hits,
        "stress_hits": s_hits,
        "sleep_hits": z_hits,
        "risk_hits": r_hits,
        "has_crisis": r_cnt > 0,
    }
    return state, summary  # 返回更新后的状态和总结


# ================== 徽标与日志 ==================
def log_badge_html(state: Dict[str, Any], html: str, extra: Optional[Dict[str, Any]] = None):
    """将评估状态和徽标HTML记录到日志"""
    payload: Dict[str, Any] = {  # 日志内容
        "session_id": st.session_state.get("session_id"),
        "state": state,
        "badge_html": html,
        "ts": datetime.utcnow().isoformat() + "Z",  # UTC时间
    }
    if extra:  # 若有额外信息，添加到日志
        payload.update(extra)
    logger.info(json.dumps(payload, ensure_ascii=False))  # 记录日志


def risk_badge_html(state: Dict[str, Any]) -> str:
    """生成风险评估徽标的HTML代码"""

    def badge(label, val):
        """生成单个徽标的HTML"""
        # 根据值确定级别和颜色
        level = "安全" if val == 0 else ("轻度" if val <= 2 else ("中度" if val <= 5 else "偏高"))
        color = "#19C37D" if val == 0 else ("#FACC15" if val <= 2 else ("#F59E0B" if val <= 5 else "#EF4444"))
        return f"""
        <div style=\\"display:inline-block;padding:6px 10px;margin-right:10px;border-radius:999px;background:{color}22;border:1px solid {color};font-size:12px;\\">
            <b>{label}</b>: {level}（{val}）
        </div>
        """

    crisis_hint = ""  # 危机提示信息
    if state.get("risk", 0) > 0:  # 若存在风险
        terms = sorted(set(state.get("risk_terms", [])))  # 去重并排序风险词
        crisis_hint = (  # 生成风险提示HTML
            f"""<div style=\\"margin-top:6px;font-size:12px;color:#b91c1c;\\">
        ⚠️ 监测到潜在风险词：<b>{'、'.join(terms[:6])}</b>。如有当下危险，请立刻联系当地急救或危机干预热线。
        </div>"""
        )

    # 组合所有徽标和提示信息
    html = f"""
    <div style=\\"padding:10px 12px;border:1px solid #e5e7eb;border-radius:12px;background:#f8fafc;\\">
        {badge("情绪", state.get("emotion", 0))}
        {badge("压力", state.get("stress", 0))}
        {badge("睡眠", state.get("sleep", 0))}
        {badge("风险", state.get("risk", 0))}
        {crisis_hint}
    </div>
    """
    return html.strip()  # 返回HTML字符串并去除首尾空白


# ================== 通义千问：对话接口 ==================
def call_qwen(prompt: str, conversation_history):
    """调用通义千问API生成回复"""
    headers = {"Authorization": f"Bearer {TONGYI_API_KEY}", "Content-Type": "application/json"}  # 请求头
    # 构建消息列表，包含系统提示、历史对话和当前用户输入
    messages = [{"role": "system", "content": PSYCHOLOGIST_SYSTEM_PROMPT}]
    messages.extend(conversation_history[-8:])  # 只取最近8轮对话，避免过长
    messages.append({"role": "user", "content": prompt})

    # API请求数据
    data = {
        "model": MODEL_NAME,
        "input": {"messages": messages},
        "parameters": {"result_format": "message", "stream": False, "incremental_output": False},
    }
    try:
        # 发送POST请求
        resp = requests.post(TONGYI_API_URL, headers=headers, json=data, timeout=40)
        resp.raise_for_status()  # 若请求失败，抛出异常
        j = resp.json()  # 解析JSON响应
        # 提取回复内容
        return j.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content", "（未返回内容）")
    except Exception as e:  # 捕获异常并返回错误信息
        return f"API 调用失败：{e}"


# ================== 轻量“打字机”效果 ==================
def typewriter_markdown(text: str, delay: float = 0.015):
    """实现打字机效果，逐字显示文本"""
    holder = st.empty()  # 创建一个空的Streamlit组件用于显示文本
    buf = ""  # 缓冲区，存储已显示的文本
    for ch in text:  # 遍历文本中的每个字符
        buf += ch  # 添加到缓冲区
        holder.markdown(buf)  # 更新显示
        time.sleep(delay)  # 延迟一段时间，模拟打字速度


# ================== 语音合成与播放功能 ==================
import base64, tempfile, os, subprocess  # 导入额外需要的库


def _convert_to_wav(src_path: str, dst_path: str, sr: int = 16000):
    """用 ffmpeg 将音频文件转换为16k单声道wav格式；阻塞执行但很快。"""
    # 构建ffmpeg命令
    cmd = ["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", str(sr), dst_path]
    try:
        # 执行命令，隐藏输出
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        # 转换失败时抛出异常
        raise RuntimeError("音频转码失败，请确认已安装 ffmpeg") from e


# --------- TTS：文本→语音 ---------
def text_to_speech(text: str, voice: str = "zh-CN"):
    """将文本转换为语音并返回音频文件路径（mp3）。优先 gTTS，失败时给出提示。"""
    try:
        from gtts import gTTS  # 导入gTTS库
        # 创建临时mp3文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            tts = gTTS(text=text, lang="zh-CN", slow=False)  # 创建语音合成实例
            tts.save(f.name)  # 保存到临时文件
            return f.name  # 返回临时文件路径
    except Exception as e:  # 捕获异常并显示错误
        st.error(f"语音合成失败（TTS）：{e}")
        return None


def play_audio(audio_path: str):
    """在网页中播放音频文件（隐藏播放器控件），播放后清理临时文件。"""
    if not audio_path or not os.path.exists(audio_path):  # 检查文件是否存在
        return
    # 确定文件后缀和MIME类型
    ext = os.path.splitext(audio_path)[1].lower()
    mime = "audio/wav" if ext == ".wav" else "audio/mp3"

    # 读取音频文件并编码为base64
    audio_b64 = base64.b64encode(open(audio_path, "rb").read()).decode()
    # 构建HTML音频播放器（隐藏控件）
    audio_html = f"""
    <audio autoplay style="display: none;">
      <source src="data:{mime};base64,{audio_b64}" type="{mime}">
      您的浏览器不支持音频播放
    </audio>
    """
    components.html(audio_html, height=0)  # 嵌入HTML组件，高度设为0
    try:
        os.unlink(audio_path)  # 播放后删除临时文件
    except Exception:
        pass  # 忽略删除失败的情况


# --------- ASR：上传/录音文件 → 识别文本 ---------
def transcribe_audio_file(uploaded_file) -> str:
    """
    对 st.file_uploader 返回的文件进行识别：
    优先 faster-whisper，没有则自动 fallback 到 whisper
    """
    if not uploaded_file:  # 若没有上传文件，返回空字符串
        return ""

    try:
        # 保存上传的原始文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as raw_f:
            raw_f.write(uploaded_file.read())
            raw_path = raw_f.name

        # 转换为wav格式
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_f:
            wav_path = wav_f.name
        _convert_to_wav(raw_path, wav_path, 16000)

        # 优先：faster-whisper（更快），可选依赖
        try:
            from faster_whisper import WhisperModel  # type: ignore
            model = WhisperModel("base", device="cpu", compute_type="int8")  # 加载模型
            segments, info = model.transcribe(wav_path, language="zh", vad_filter=True)  # 识别音频
            return "".join(seg.text for seg in segments).strip()  # 拼接识别结果

        except ImportError:
            # 备选：openai/whisper（也可能未安装）
            try:
                import whisper
                model = whisper.load_model("base")  # 加载模型
                result = model.transcribe(wav_path, language="zh")  # 识别音频
                return (result.get("text") or "").strip()  # 返回识别结果
            except ImportError:  # 若均未安装，显示警告
                st.warning("⚠ 未检测到语音识别依赖（faster-whisper 或 whisper），语音识别功能不可用。")
                return ""

    except Exception as e:  # 捕获异常并显示错误
        st.error(f"语音识别失败：{e}")
        return ""


# --------- 语音输入面板 ---------
def voice_input_panel(key: str = "voice_uploader") -> str:
    """
    最简MVP：使用 file_uploader 作为“语音输入”
    - 支持 webm/mp3/wav/ogg 等；上传后立即识别并返回文本
    """
    with st.expander("🎤 语音输入（上传或录音文件）", expanded=False):  # 创建可展开面板
        # 创建文件上传器
        f = st.file_uploader("上传/拖拽音频文件（建议 webm/mp3/wav/ogg）", type=["webm", "mp3", "wav", "ogg"], key=key)
        if f is not None:  # 若有文件上传
            with st.spinner("正在识别语音…"):  # 显示加载状态
                text = transcribe_audio_file(f)  # 识别音频
            if text:  # 若识别到文本
                st.success("识别完成，已将文本填入输入框上方（或请手动复制粘贴）。")
                st.markdown(f"> **识别结果：** {text}")  # 显示识别结果
                return text
            else:  # 若未识别到文本
                st.warning("没有识别到有效文本。")
    return ""  # 未上传文件或识别失败，返回空字符串


# ================== YOLOv8 模型加载（缓存） ==================
@st.cache_resource  # 缓存模型，避免重复加载
def load_yolov8_model(model_path: str):
    """加载YOLOv8模型"""
    try:
        model = YOLO(model_path)  # 加载模型
        st.success("✅ 模型加载成功")  # 显示成功信息
        return model  # 返回模型实例
    except Exception as e:  # 捕获异常并显示错误
        st.error(f"❌ 模型加载失败：{str(e)}")
        return None


# ================== 摄像头实时检测（生成器） ==================
def detect_camera(model, conf_threshold=0.25, max_fps=15, stop_flag=None, device_index: int = 0):
    """从摄像头捕获画面并进行实时情绪检测，返回生成器"""
    cap = cv2.VideoCapture(device_index)  # 打开摄像头
    if not cap.isOpened():  # 若无法打开摄像头，显示错误
        st.error("❌ 无法打开摄像头，请检查摄像头权限/占用")
        return

    prev_time = 0  # 上一帧的时间，用于计算帧率
    # 与训练类别一致的映射：索引到情绪名称
    emotion_classes = {
        0: "angry",
        1: "contempt",
        2: "disgust",
        3: "fear",
        4: "happy",
        5: "natural",
        6: "sad",
        7: "sleepy",
        8: "surprised",
    }

    try:
        while cap.isOpened():  # 循环读取摄像头画面
            if stop_flag and stop_flag[0]:  # 若收到停止信号，退出循环
                break
            ret, frame = cap.read()  # 读取一帧画面
            if not ret:  # 若无法获取画面，显示警告
                st.warning("⚠️ 无法获取摄像头画面")
                break

            # 计算帧率并控制最大帧率
            now = time.time()
            fps = 1 / (now - prev_time) if (now - prev_time) > 0 else 0
            prev_time = now
            if fps > max_fps:  # 若帧率超过最大值，休眠一段时间
                time.sleep(max(0, 1 / max_fps - (time.time() - now)))

            # 使用YOLO模型进行检测
            results = model(frame, conf=conf_threshold)
            result_frame = results[0].plot()  # 绘制检测结果到画面

            # 提取情绪检测结果
            emotion_data_list = []
            if len(results[0].boxes) > 0:  # 若检测到目标
                for box in results[0].boxes:  # 遍历每个检测框
                    cls = int(box.cls[0])  # 类别索引
                    conf = float(box.conf[0])  # 置信度
                    emotion = emotion_classes.get(cls, f"未知({cls})")  # 获取情绪名称
                    emotion_data_list.append(f"{emotion} (置信度: {conf:.2f})")  # 添加到列表

            # 转换画面格式为PIL Image（RGB格式）
            result_pil = Image.fromarray(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
            main_emotion = emotion_data_list[0] if emotion_data_list else None  # 主要情绪（第一个检测到的）
            # 生成器返回当前帧、帧率、主要情绪和情绪列表
            yield result_pil, round(fps, 1), main_emotion, emotion_data_list
    finally:  # 确保资源释放
        cap.release()  # 释放摄像头
        cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

    yield None, 0, None, []  # 结束时返回空值


# ================== 情绪检测日志功能 ==================
def log_emotion_detection(fps: float, face_count: int, emotions: List[str]):
    """记录情绪检测结果到专用日志"""
    emotions_str = "|".join(emotions) if emotions else "无检测结果"  # 情绪列表转换为字符串
    emotion_logger.info(
        "",  # 消息内容（使用extra参数传递额外信息）
        extra={
            "session_id": st.session_state.session_id,  # 会话ID
            "fps": int(fps),  # 帧率
            "face_count": face_count,  # 面部数量
            "emotions": emotions_str  # 情绪列表
        }
    )


def init_detailed_log_file():
    """初始化详细日志文件（每次运行会创建新文件，不会覆盖历史详细日志）"""
    det_dir = os.path.join(LOG_DIR, "detection_logs")  # 详细日志目录
    os.makedirs(det_dir, exist_ok=True)  # 创建目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 当前时间戳
    log_filename = os.path.join(det_dir, f"emotion_detection_{timestamp}.log")  # 日志文件路径
    # 写入日志头部信息
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"情绪检测详细日志 - 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"会话ID: {st.session_state.session_id}\n")
        f.write("=" * 50 + "\n\n")
    return log_filename  # 返回日志文件路径


def log_detailed_detection_data(log_file, timestamp, fps, emotion_list):
    """记录详细的检测数据到文件"""
    if not log_file or not os.path.exists(log_file):  # 检查日志文件是否存在
        return
    # 追加检测数据到日志文件
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"【检测记录】时间: {timestamp}\n")
        f.write(f"帧率: {fps} FPS\n")
        f.write(f"检测到的面部数量: {len(emotion_list)}\n")
        f.write("识别到的情绪:\n")
        for i, emotion in enumerate(emotion_list, 1):
            f.write(f"  {i}. {emotion}\n")
        f.write("-" * 50 + "\n")


# ================== UI：数字人部分 ==================

# 载入 Lottie 动画 JSON（本地文件）
def load_lottie(path: str):
    """加载Lottie动画的JSON文件"""
    try:
        with open(path, "r", encoding="utf-8") as f:  # 读取文件
            return json.load(f)  # 返回JSON数据
    except Exception as e:  # 捕获异常并显示错误
        st.error(f"加载动画失败: {e}")
        return None


# 渲染“全屏样式”的 Lottie 浮窗（固定在组件 iframe 内，可拖动+可关闭）
def render_dual_lottie_overlays(lottie_right: dict, lottie_left: dict,
                                speed: float = 1.0,
                                width="20vw", height="20vh"):
    """渲染两个可拖动、可关闭的Lottie动画浮窗"""
    # 构建HTML代码，包含动画浮窗和JavaScript逻辑
    html = f"""
    <style>
      /* 入场与悬停动画 */
      @keyframes fadeInScale {{
        from {{ opacity: 0; transform: scale(0.8); }}
        to {{ opacity: 1; transform: scale(1); }}
      }}
      @keyframes float {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-6px); }}
        100% {{ transform: translateY(0px); }}
      }}
      .lottie-box {{
        animation: fadeInScale 0.8s ease-out;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
      }}
      .lottie-box:hover {{
        animation: float 3s ease-in-out infinite;
        box-shadow: 0 12px 30px rgba(0,0,0,0.25);
      }}
    </style>

    <!-- 右侧动画浮窗 -->
    <div id="lottie-right" class="lottie-box" style="
        position:fixed; top:20px; right:20px;
        width:{width}; height:{height};
        z-index:9999; display:flex; justify-content:center; align-items:center;
        background:rgba(255, 255, 255, 0.6); backdrop-filter: blur(6px);
        border-radius:16px; box-shadow:0 8px 24px rgba(0,0,0,0.15);
        cursor:move;
    ">
      <div id="lottie-right-canvas" style="width:100%;height:100%;"></div>
      <button id="lottie-right-close" style="
        position:absolute; top:6px; right:6px;
        border:none; background:rgba(0,0,0,0.6); color:#fff;
        padding:6px 10px; border-radius:8px; font-size:12px;
        cursor:pointer;
      ">×</button>
    </div>

    <!-- 左侧动画浮窗 -->
    <div id="lottie-left" class="lottie-box" style="
        position:fixed; top:20px; left:20px;
        width:{width}; height:{height};
        z-index:9999; display:flex; justify-content:center; align-items:center;
        background:rgba(255, 255, 255, 0.6); backdrop-filter: blur(6px);
        border-radius:16px; box-shadow:0 8px 24px rgba(0,0,0,0.15);
        cursor:move;
    ">
      <div id="lottie-left-canvas" style="width:100%;height:100%;"></div>
      <button id="lottie-left-close" style="
        position:absolute; top:6px; right:6px;
        border:none; background:rgba(0,0,0,0.6); color:#fff;
        padding:6px 10px; border-radius:8px; font-size:12px;
        cursor:pointer;
      ">×</button>
    </div>

    <!-- 引入Lottie库 -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.12.2/lottie.min.js"></script>
    <script>
      const rightData = {json.dumps(lottie_right)};  // 右侧动画数据
      const leftData  = {json.dumps(lottie_left)};   // 左侧动画数据

      // 初始化右侧动画
      const animRight = lottie.loadAnimation({{
        container: document.getElementById('lottie-right-canvas'),
        renderer: 'svg',
        loop: true,
        autoplay: true,
        animationData: rightData
      }});
      animRight.setSpeed({speed});  // 设置动画速度

      // 初始化左侧动画
      const animLeft = lottie.loadAnimation({{
        container: document.getElementById('lottie-left-canvas'),
        renderer: 'svg',
        loop: true,
        autoplay: true,
        animationData: leftData
      }});
      animLeft.setSpeed({speed});  // 设置动画速度

      // 通用拖动逻辑
      function makeDraggable(boxId, closeId) {{
        const box = document.getElementById(boxId);
        let isDragging = false;
        let offsetX = 0, offsetY = 0;

        // 鼠标按下时开始拖动
        box.addEventListener('mousedown', (e) => {{
          if (e.target.id === closeId) return;  // 点击关闭按钮不触发拖动
          isDragging = true;
          const rect = box.getBoundingClientRect();
          offsetX = e.clientX - rect.left;
          offsetY = e.clientY - rect.top;
          box.style.left = rect.left + 'px';
          box.style.top = rect.top + 'px';
          box.style.right = 'auto';
          box.style.bottom = 'auto';
          document.addEventListener('mousemove', onMove);
          document.addEventListener('mouseup', onUp);
        }});

        // 鼠标移动时更新位置
        function onMove(e) {{
          if (!isDragging) return;
          e.preventDefault();
          const maxX = window.innerWidth - box.offsetWidth;
          const maxY = window.innerHeight - box.offsetHeight;
          let newX = e.clientX - offsetX;
          let newY = e.clientY - offsetY;
          newX = Math.max(0, Math.min(maxX, newX));  // 限制在窗口内
          newY = Math.max(0, Math.min(maxY, newY));  // 限制在窗口内
          box.style.left = newX + 'px';
          box.style.top  = newY + 'px';
        }}

        // 鼠标释放时结束拖动
        function onUp() {{
          isDragging = false;
          document.removeEventListener('mousemove', onMove);
          document.removeEventListener('mouseup', onUp);
        }}

        // 关闭按钮点击事件
        document.getElementById(closeId).onclick = () => {{
          box.style.display = 'none';
        }};
      }}

      // 使两个浮窗可拖动
      makeDraggable('lottie-right', 'lottie-right-close');
      makeDraggable('lottie-left', 'lottie-left-close');
    </script>
    """
    # 在Streamlit中显示HTML组件
    components.html(html, height=600, scrolling=False)


# ================== UI：两个标签页（咨询对话 / 情绪检测） ==================
def render_chat_ui():
    """渲染咨询对话界面"""
    st.title("🧠 专业心理咨询师助手")  # 页面标题
    st.caption("更类人化的对话 · 对话中自动初步评估 · 支持语音播报 · 无侧边栏沉浸式体验")  # 副标题

    # 语音开关
    col1, col2 = st.columns([4, 1])  # 分为两列
    with col2:
        # 语音播报开关
        st.session_state.audio_enabled = st.checkbox("启用AI语音播报", value=st.session_state.audio_enabled)

    # 保密与安全说明
    with st.expander("保密与安全说明", expanded=False):
        st.info("本服务由 AI 提供，仅作参考，不能替代医疗或紧急救助。若出现自/他伤风险或当下危险，请立即联系当地急救或危机热线。")

    # 顶部徽标（默认仅写日志，不在页面展示）
    SHOW_BADGES = False  # 是否显示徽标
    _badge_html_top = risk_badge_html(st.session_state.assess)  # 生成徽标HTML
    if SHOW_BADGES:  # 若显示徽标
        st.markdown(_badge_html_top, unsafe_allow_html=True)
    # 记录徽标日志
    log_badge_html(st.session_state.assess, _badge_html_top, extra={"where": "top"})
    st.divider()  # 分隔线

    # 开场
    if not st.session_state.started:  # 若对话未开始
        with st.chat_message("assistant", avatar="🧑‍⚕️"):  # 显示助手消息
            greet = (  # 问候语
                "您好，欢迎来到咨询室。我会尽力给到一个**安全、温暖、无评判**的空间。\n\n"
                "为了更好地陪伴您，我会在自然对话中留意**情绪/压力/睡眠/风险**四方面做一个**初步评估**；"
                "如果您愿意，也可以告诉我您希望我怎么称呼您，我们就从您觉得最舒适的地方聊起。"
            )
            typewriter_markdown(greet, delay=0.01)  # 打字机效果显示问候语

            # 播放开场语音
            if st.session_state.audio_enabled:  # 若启用语音播报
                audio_path = text_to_speech(greet.replace("**", ""))  # 去除markdown格式
                if audio_path:  # 若语音合成成功
                    play_audio(audio_path)  # 播放语音

        st.session_state.started = True  # 标记对话已开始

    # 历史消息
    for msg in st.session_state.history:  # 遍历对话历史
        with st.chat_message(msg["role"], avatar=("🧑‍⚕️" if msg["role"] == "assistant" else "🙂")):
            st.markdown(msg["content"])  # 显示消息内容

    # 输入框
    user_text = st.chat_input("请用自然的方式描述你现在的处境、感受或具体情境…")  # 聊天输入框
    if user_text:  # 若用户输入了文本
        st.session_state.history.append({"role": "user", "content": user_text})  # 添加到历史
        with st.chat_message("user", avatar="🙂"):  # 显示用户消息
            st.markdown(user_text)

        # 增量评估
        st.session_state.assess, summary = assess_incremental(user_text, st.session_state.assess)

        # 安全优先
        crisis_prefix = ""  # 危机提示前缀
        if summary.get("has_crisis"):  # 若检测到危机关键词
            crisis_prefix = (  # 危机提示内容
                "我注意到你刚才的表达里出现了可能涉及安全风险的内容。\n\n"
                "- 先确认你的**当下安全**：此刻你是否身处安全环境？是否有**具体计划或行动**的冲动？\n"
                "- 如果存在迫切风险，请**立刻**联系当地急救或危机干预热线；我也会尽量在这里陪你，帮助你把注意力拉回到安全与支持上。\n\n"
            )

        # 生成回复
        name = st.session_state.assess.get("name")  # 获取用户姓名
        name_part = f"{name}，" if name else ""  # 称呼部分
        # 构建提示词
        prompt_for_model = (
            f"{user_text}\n\n"
            "（请以上述系统风格进行自然回应；若检测到风险词，优先做安全评估；"
            "若无迫切风险，结合用户语境在四维中提出1-2个具体、可答的问题，并给到温和的小建议或下一步。）"
        )
        assistant_reply = call_qwen(prompt_for_model, st.session_state.history)  # 调用API生成回复

        with st.chat_message("assistant", avatar="🧑‍⚕️"):  # 显示助手回复
            final_text = (crisis_prefix + name_part + assistant_reply).strip()  # 组合最终回复
            typewriter_markdown(final_text, delay=0.008)  # 打字机效果显示

            # 播放AI回复语音
            if st.session_state.audio_enabled:  # 若启用语音播报
                # 去除markdown格式以便更好地语音合成
                clean_text = re.sub(r"\*\*", "", final_text)
                clean_text = re.sub(r"\n", " ", clean_text)
                audio_path = text_to_speech(clean_text)  # 合成语音
                if audio_path:  # 若合成成功
                    play_audio(audio_path)  # 播放语音

        st.session_state.history.append({"role": "assistant", "content": final_text})  # 添加到历史

        # 刷新徽标（仍默认仅日志）
        st.markdown("---")  # 分隔线
        _badge_html_after = risk_badge_html(st.session_state.assess)  # 生成徽标HTML
        # 记录日志
        log_badge_html(st.session_state.assess, _badge_html_after,
                       extra={"where": "after_message", "last_user_msg": user_text})

    st.divider()  # 分隔线
    st.caption("专业心理咨询师助手 · Streamlit × 通义千问")  # 页脚说明
    # 额外说明
    st.markdown(
        """
        <div style="text-align:center;color:#666;font-size:0.85rem;">
        遇到紧急情况，请联系当地紧急救助或心理危机干预热线。
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Lottie动画相关设置
    st.set_page_config(page_title="Lottie 浮窗 Demo", layout="wide")

    # 定义动画源文件（请按你的本地路径修改）
    animations = {
        "可爱女孩": r"Doctor, Medical, Surgeon, Healthcare Animation.json",
        "友好机器人": r"Live chatbot.json",
    }

    name = st.sidebar.selectbox("选择动画", list(animations.keys()))  # 选择动画
    show = st.sidebar.checkbox("显示动画浮窗", True)  # 是否显示动画

    lottie_source = animations[name]  # 动画文件路径
    lottie_json = load_lottie(lottie_source)  # 加载动画

    if not lottie_json:  # 若加载失败
        st.error("动画加载失败，请检查文件路径或 JSON 是否有效")
        return

    st.write("浮窗）")

    if show:  # 若显示动画
        # 加载两个动画
        doctor_json = load_lottie(r"Doctor, Medical, Surgeon, Healthcare Animation.json")
        chatbot_json = load_lottie(r"Live chatbot.json")
        render_dual_lottie_overlays(doctor_json, chatbot_json, speed=1)  # 渲染双浮窗


def render_yolo_ui():
    """渲染YOLO情绪检测界面"""
    st.title("😊 YOLOv8 实时情绪检测")  # 页面标题
    st.caption("摄像头实时检测面部情绪 · 默认自动加载并开启 · 可调置信度/帧率")  # 副标题

    # --------------- 默认参数 ---------------
    model_path = r"best.pt"  # 模型文件路径
    conf_threshold = 0.25  # 置信度阈值
    max_fps = 15  # 最大帧率
    device_index = 0  # 摄像头编号

    # --------------- 自动加载模型 ---------------
    if st.session_state.yolo_model is None:  # 若模型未加载
        st.session_state.yolo_model = load_yolov8_model(model_path)  # 加载模型

    # --------------- 页面布局 ---------------
    col_left, col_right = st.columns([3, 1])  # 分为左右两列

    with col_right:  # 右侧参数和状态面板
        st.subheader("参数")  # 参数标题
        # 置信度阈值滑块
        conf_threshold = st.slider("置信度阈值", 0.01, 1.0, conf_threshold, 0.01)
        # 最大帧率滑块
        max_fps = st.slider("最大帧率", 5, 30, max_fps)
        # 摄像头编号输入
        device_index = st.number_input("摄像头编号", 0, 5, device_index, 1)

        # 日志初始化
        if st.button("初始化详细日志", use_container_width=True):  # 初始化日志按钮
            st.session_state.yolo_log_file = init_detailed_log_file()  # 初始化日志文件
            st.success(f"详细日志文件已创建: {os.path.basename(st.session_state.yolo_log_file)}")  # 显示成功信息

        st.divider()  # 分隔线
        st.subheader("状态")  # 状态标题
        y_status = st.empty()  # 状态显示区域
        y_log = st.empty()  # 日志信息区域
        y_cnt = st.empty()  # 面部数量区域
        y_em = st.empty()  # 主要情绪区域
        y_fps = st.empty()  # 帧率区域

    with col_left:  # 左侧视频显示区域
        st.subheader("实时检测画面")  # 标题
        y_video = st.empty()  # 视频显示区域

    # --------------- 自动开启检测 ---------------
    if st.session_state.yolo_model and not st.session_state.yolo_started:  # 若模型已加载且未开始检测
        st.session_state.yolo_started = True  # 标记为已开始
        st.session_state.yolo_stop_flag[0] = False  # 重置停止标志

        if st.session_state.yolo_log_file:  # 若已初始化日志文件
            y_log.info(f"详细日志记录中: {os.path.basename(st.session_state.yolo_log_file)}")
        else:  # 未初始化详细日志
            y_log.info("已启用基础日志记录（点击'初始化详细日志'获取更完整记录）")

        y_status.success("检测中…")  # 显示检测中状态
        try:
            # 遍历检测生成器
            for frame, fps, main_emotion, emotion_list in detect_camera(
                    model=st.session_state.yolo_model,
                    conf_threshold=conf_threshold,
                    max_fps=max_fps,
                    stop_flag=st.session_state.yolo_stop_flag,
                    device_index=device_index,
            ):
                if frame is None:  # 若帧为空，退出循环
                    break

                # 更新会话状态
                st.session_state.yolo_last_emotion = main_emotion
                st.session_state.yolo_last_emotion_list = emotion_list

                # 更新UI
                y_video.image(frame, use_container_width=True)  # 显示画面
                y_fps.text(f"当前帧率: {fps} FPS")  # 显示帧率
                y_cnt.text(f"检测到的面部: {len(emotion_list)} 个")  # 显示面部数量
                if main_emotion:  # 若有主要情绪，显示
                    y_em.text(f"主要情绪: {main_emotion}")

                # 记录到情绪检测专用日志
                log_emotion_detection(fps, len(emotion_list), emotion_list)

                # 记录到详细日志文件（如果已初始化）
                if st.session_state.yolo_log_file:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 当前时间
                    log_detailed_detection_data(st.session_state.yolo_log_file, ts, fps, emotion_list)  # 记录详细日志

                if st.session_state.yolo_stop_flag[0]:  # 若收到停止信号，退出循环
                    break
        except Exception as e:  # 捕获异常
            y_status.error(f"检测出错: {str(e)}")  # 显示错误信息
        finally:  # 无论是否异常，都会执行
            y_status.info("检测已停止")  # 显示已停止状态
            st.session_state.yolo_started = False  # 标记为已停止

    # --------------- 手动停止 ---------------
    if st.session_state.yolo_started:  # 若检测已开始
        if st.button("停止检测", use_container_width=True, type="primary"):  # 停止按钮
            st.session_state.yolo_stop_flag[0] = True  # 设置停止标志
            st.session_state.yolo_started = False  # 标记为已停止
            y_status.info("正在停止检测…")  # 显示正在停止状态


# ================== 主入口：标签页切换 ==================
def main():
    """主函数，控制页面展示"""
    # 导入base64模块（用于音频编码）
    global base64
    import base64

    # 创建标签页
    tabs = st.tabs(["💬 咨询对话", " "])
    with tabs[0]:  # 第一个标签页：咨询对话
        render_chat_ui()
    with tabs[1]:  # 第二个标签页：情绪检测
        render_yolo_ui()


if __name__ == "__main__":  # 若作为主程序运行
    main()  # 调用主函数