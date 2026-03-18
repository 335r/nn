# -*- coding: utf-8 -*-
"""
情绪日志综合分析工具
用于解析finally_plus.py生成的日志文件，综合评估用户情绪状态
"""
import os
import re
import json
import logging
from datetime import datetime
from collections import defaultdict
import pandas as pd
# 首先配置Matplotlib后端（必须在导入pyplot之前）
import matplotlib
import matplotlib.font_manager

# 强制设置为Agg后端，确保在任何环境下都能正常生成图像文件
matplotlib.rcParams["backend"] = "Agg"

# ---------------------- 中文字体配置（修复中文显示为方框问题） ----------------------
if 'SimHei' in matplotlib.font_manager.fontManager.get_font_names():
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # Windows默认黑体
elif 'WenQuanYi Zen Hei' in matplotlib.font_manager.fontManager.get_font_names():
    matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']  # Linux开源字体
elif 'Arial Unicode MS' in matplotlib.font_manager.fontManager.get_font_names():
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']  # Mac默认字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 防止负号显示为方框
# ----------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates  # 时间格式化工具

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('emotion_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('emotion_analyzer')
# 情绪映射与权重配置
EMOTION_WEIGHTS = {
    # YOLO检测情绪权重
    'yolo': {
        'angry': -0.8,
        'contempt': -0.6,
        'disgust': -0.7,
        'fear': -0.9,
        'happy': 0.8,
        'natural': 0.0,
        'sad': -0.7,
        'sleepy': -0.3,
        'surprised': 0.2
    },
    # 文本关键词权重
    'text': {
        'emotion': {
            '难过': -0.6, '低落': -0.7, '沮丧': -0.7,
            '焦虑': -0.6, '紧张': -0.5, '恐慌': -0.8,
            '崩溃': -0.9, '空虚': -0.4, '孤独': -0.5,
            '愤怒': -0.7, '烦躁': -0.5
        },
        'stress': 0.3,  # 压力关键词整体权重系数
        'sleep': 0.2,  # 睡眠问题整体权重系数
        'risk': 0.4  # 风险关键词整体权重系数
    }
}
# 情绪评分区间定义
SCORE_RANGES = {
    (-1.0, -0.7): ('严重负面', 'red'),
    (-0.7, -0.3): ('中度负面', 'orange'),
    (-0.3, 0.3): ('中性', 'gray'),
    (0.3, 0.7): ('中度正面', 'lightgreen'),
    (0.7, 1.0): ('强烈正面', 'green')
}


class EmotionAnalyzer:
    def __init__(self, log_dir=None):
        """初始化分析器，默认日志目录与finally_plus.py保持一致"""
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_dir = log_dir if log_dir else os.path.join(self.base_dir, "logs")
        self.detection_log_path = os.path.join(self.log_dir, "emotion_detections.log")
        self.assessment_log_path = os.path.join(self.log_dir, "assessment_badges.log")
        self.detection_detail_dir = os.path.join(self.log_dir, "detection_logs")
        # 确保日志目录存在
        if not os.path.exists(self.log_dir):
            logger.warning(f"日志目录不存在: {self.log_dir}")
        # 分析结果存储
        self.session_data = defaultdict(lambda: {
            'yolo_emotions': [],
            'text_analysis': [],
            'comprehensive': []
        })

    def parse_emotion_detections(self):
        """解析情绪检测日志"""
        if not os.path.exists(self.detection_log_path):
            logger.error(f"情绪检测日志不存在: {self.detection_log_path}")
            return
        logger.info(f"开始解析情绪检测日志: {self.detection_log_path}")
        with open(self.detection_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 解析日志格式: 时间\t会话ID\tFPS\t面部数量\t情绪列表
                parts = line.split('\t')
                if len(parts) != 5:
                    continue
                try:
                    timestamp = datetime.strptime(parts[0], '%Y-%m-%d %H:%M:%S,%f')
                    session_id = parts[1]
                    fps = int(parts[2])
                    face_count = int(parts[3])
                    emotions_str = parts[4]
                    # 解析情绪列表
                    emotions = []
                    if emotions_str != "无检测结果":
                        for emo in emotions_str.split('|'):
                            # 提取情绪名称和置信度
                            match = re.match(r'(.+?)\s+\(置信度: (.+?)\)', emo)
                            if match:
                                emotion_name = match.group(1)
                                confidence = float(match.group(2))
                                emotions.append((emotion_name, confidence))
                    # 计算YOLO情绪得分
                    yolo_score = self._calculate_yolo_score(emotions)
                    # 存储结果
                    self.session_data[session_id]['yolo_emotions'].append({
                        'timestamp': timestamp,
                        'fps': fps,
                        'face_count': face_count,
                        'emotions': emotions,
                        'score': yolo_score
                    })
                except Exception as e:
                    logger.warning(f"解析日志行失败: {line}, 错误: {str(e)}")
        logger.info(f"情绪检测日志解析完成，共处理 {len(self.session_data)} 个会话")

    def parse_assessment_logs(self):
        """解析评估日志"""
        if not os.path.exists(self.assessment_log_path):
            logger.error(f"评估日志不存在: {self.assessment_log_path}")
            return
        logger.info(f"开始解析评估日志: {self.assessment_log_path}")
        with open(self.assessment_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 解析日志格式: 时间\tJSON数据
                parts = line.split('\t', 1)
                if len(parts) != 2:
                    continue
                try:
                    timestamp = datetime.strptime(parts[0], '%Y-%m-%d %H:%M:%S,%f')
                    data = json.loads(parts[1])
                    session_id = data.get('session_id')
                    if not session_id:
                        continue
                    state = data.get('state', {})
                    # 计算文本分析得分
                    text_score = self._calculate_text_score(state)
                    # 存储结果
                    self.session_data[session_id]['text_analysis'].append({
                        'timestamp': timestamp,
                        'state': state,
                        'score': text_score
                    })
                except Exception as e:
                    logger.warning(f"解析评估日志行失败: {line}, 错误: {str(e)}")
        logger.info("评估日志解析完成")

    def _calculate_yolo_score(self, emotions):
        """计算YOLO情绪检测得分"""
        if not emotions:
            return 0.0
        total_score = 0.0
        total_confidence = 0.0
        for emotion, confidence in emotions:
            # 情绪基础分 * 置信度
            if emotion in EMOTION_WEIGHTS['yolo']:
                score = EMOTION_WEIGHTS['yolo'][emotion] * confidence
                total_score += score
                total_confidence += confidence
        # 归一化
        return total_score / total_confidence if total_confidence > 0 else 0.0

    def _calculate_text_score(self, state):
        """计算文本分析得分"""
        emotion_score = 0.0
        emotion_hits = state.get('emotion_hits', [])
        # 计算情绪关键词得分
        for hit in emotion_hits:
            if hit in EMOTION_WEIGHTS['text']['emotion']:
                emotion_score += EMOTION_WEIGHTS['text']['emotion'][hit]
        # 计算压力、睡眠和风险因素得分
        stress_score = -state.get('stress', 0) * EMOTION_WEIGHTS['text']['stress']
        sleep_score = -state.get('sleep', 0) * EMOTION_WEIGHTS['text']['sleep']
        risk_score = -state.get('risk', 0) * EMOTION_WEIGHTS['text']['risk']
        # 综合文本得分（归一化处理）
        total_score = emotion_score + stress_score + sleep_score + risk_score
        return max(min(total_score, 1.0), -1.0)  # 限制在[-1, 1]范围内

    def calculate_comprehensive_score(self, session_id, yolo_weight=0.6, text_weight=0.4):
        """计算综合情绪得分"""
        if session_id not in self.session_data:
            return
        session = self.session_data[session_id]
        yolo_data = session['yolo_emotions']
        text_data = session['text_analysis']
        if not yolo_data and not text_data:
            return
        # 按时间排序
        yolo_data.sort(key=lambda x: x['timestamp'])
        text_data.sort(key=lambda x: x['timestamp'])
        # 合并时间序列并计算综合得分
        combined = []
        y_idx, t_idx = 0, 0
        while y_idx < len(yolo_data) and t_idx < len(text_data):
            y_time = yolo_data[y_idx]['timestamp']
            t_time = text_data[t_idx]['timestamp']
            if y_time <= t_time:
                # 寻找最近的文本分析
                closest_text = self._find_closest_text(text_data, t_idx, y_time)
                text_score = closest_text['score'] if closest_text else 0.0
                combined_score = (yolo_data[y_idx]['score'] * yolo_weight +
                                  text_score * text_weight)
                combined.append({
                    'timestamp': y_time,
                    'source': 'yolo',
                    'yolo_score': yolo_data[y_idx]['score'],
                    'text_score': text_score,
                    'combined_score': combined_score
                })
                y_idx += 1
            else:
                # 寻找最近的YOLO分析
                closest_yolo = self._find_closest_yolo(yolo_data, y_idx, t_time)
                yolo_score = closest_yolo['score'] if closest_yolo else 0.0
                combined_score = (yolo_score * yolo_weight +
                                  text_data[t_idx]['score'] * text_weight)
                combined.append({
                    'timestamp': t_time,
                    'source': 'text',
                    'yolo_score': yolo_score,
                    'text_score': text_data[t_idx]['score'],
                    'combined_score': combined_score
                })
                t_idx += 1
        # 添加剩余数据
        while y_idx < len(yolo_data):
            y_time = yolo_data[y_idx]['timestamp']
            closest_text = self._find_closest_text(text_data, t_idx, y_time)
            text_score = closest_text['score'] if closest_text else 0.0
            combined_score = (yolo_data[y_idx]['score'] * yolo_weight +
                              text_score * text_weight)
            combined.append({
                'timestamp': y_time,
                'source': 'yolo',
                'yolo_score': yolo_data[y_idx]['score'],
                'text_score': text_score,
                'combined_score': combined_score
            })
            y_idx += 1
        while t_idx < len(text_data):
            t_time = text_data[t_idx]['timestamp']
            closest_yolo = self._find_closest_yolo(yolo_data, y_idx, t_time)
            yolo_score = closest_yolo['score'] if closest_yolo else 0.0
            combined_score = (yolo_score * yolo_weight +
                              text_data[t_idx]['score'] * text_weight)
            combined.append({
                'timestamp': t_time,
                'source': 'text',
                'yolo_score': yolo_score,
                'text_score': text_data[t_idx]['score'],
                'combined_score': combined_score
            })
            t_idx += 1
        # 存储综合得分
        self.session_data[session_id]['comprehensive'] = combined

    def _find_closest_text(self, text_data, start_idx, target_time):
        """找到最接近目标时间的文本分析数据"""
        closest = None
        min_diff = None
        for i in range(start_idx):
            diff = abs(text_data[i]['timestamp'] - target_time)
            if min_diff is None or diff < min_diff:
                min_diff = diff
                closest = text_data[i]
        return closest

    def _find_closest_yolo(self, yolo_data, start_idx, target_time):
        """找到最接近目标时间的YOLO分析数据"""
        closest = None
        min_diff = None
        for i in range(start_idx):
            diff = abs(yolo_data[i]['timestamp'] - target_time)
            if min_diff is None or diff < min_diff:
                min_diff = diff
                closest = yolo_data[i]
        return closest

    def get_emotion_category(self, score):
        """根据得分获取情绪类别"""
        for (min_val, max_val), (category, color) in SCORE_RANGES.items():
            if min_val < score <= max_val:
                return category, color
        return ('未知', 'black')

    def analyze_all_sessions(self):
        """分析所有会话数据"""
        self.parse_emotion_detections()
        self.parse_assessment_logs()
        # 为每个会话计算综合得分
        for session_id in self.session_data:
            self.calculate_comprehensive_score(session_id)
        logger.info("所有会话分析完成")

    def generate_report(self, output_dir=None):
        """生成分析报告，默认输出目录为当前目录下analysis_reports/年月日_时分文件夹"""
        if not self.session_data:
            logger.warning("没有可分析的数据，无法生成报告")
            return

        # 生成以当前时间（年月日_时分）命名的目录名（精确到分钟）
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        # 默认为当前目录下的 analysis_reports/时间目录
        if output_dir is None:
            output_dir = os.path.join(self.base_dir, "analysis_reports", current_time)

        os.makedirs(output_dir, exist_ok=True)

        # 为每个会话生成报告
        for session_id, data in self.session_data.items():
            if not data['comprehensive']:
                continue
            # 创建会话报告目录
            session_dir = os.path.join(output_dir, session_id)
            os.makedirs(session_dir, exist_ok=True)
            # 生成数据表格
            self._generate_data_table(data, session_dir)
            # 生成情绪趋势图（优化后）
            self._generate_emotion_chart(data, session_dir)
            # 生成综合评估
            self._generate_comprehensive_evaluation(data, session_dir, session_id)

        logger.info(f"报告已生成至: {output_dir}")

    def _generate_data_table(self, data, output_dir):
        """生成数据表格"""
        df = pd.DataFrame(data['comprehensive'])
        if not df.empty:
            df.to_csv(os.path.join(output_dir, "emotion_scores.csv"), index=False)
            logger.info(f"数据表格已保存至: {output_dir}")

    def _generate_emotion_chart(self, data, output_dir):
        """生成情绪趋势图（优化版：显示完整+标注正常）"""
        if not data['comprehensive']:
            return
        df = pd.DataFrame(data['comprehensive'])
        # 确保使用Agg后端
        plt.switch_backend('Agg')
        # 1. 扩大画布尺寸，避免内容拥挤
        plt.figure(figsize=(15, 8))
        # 2. 绘制得分趋势，加粗线条提升可读性
        sns.lineplot(
            data=df,
            x='timestamp',
            y='combined_score',
            label='综合情绪得分',
            linewidth=2,
            color='#1f77b4'
        )
        sns.lineplot(
            data=df,
            x='timestamp',
            y='yolo_score',
            label='YOLO检测得分',
            alpha=0.6,
            linewidth=1.5,
            color='#ff7f0e'
        )
        sns.lineplot(
            data=df,
            x='timestamp',
            y='text_score',
            label='文本分析得分',
            alpha=0.6,
            linewidth=1.5,
            color='#2ca02c'
        )
        # 3. 优化时间轴：按15秒间隔显示，仅展示时分秒
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # 时间格式：时分秒
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=15))  # 每15秒一个刻度
        plt.xticks(rotation=30, fontsize=10)  # 旋转30度避免文字重叠
        # 4. 绘制情绪区间参考线+标注（带背景框，避免重叠）
        for (min_val, max_val), (category, color) in SCORE_RANGES.items():
            # 绘制区间分隔线（用对应颜色）
            ax.axhline(y=min_val, color=color, linestyle='--', alpha=0.6, linewidth=1)
            # 计算标注x轴位置：时间轴10%处，避免左边界重叠
            time_range = df['timestamp'].max() - df['timestamp'].min()
            x_pos = df['timestamp'].min() + time_range * 0.1
            # 添加标注文字（带半透明背景框）
            ax.text(
                x_pos,
                min_val + 0.02,  # y轴微上调，避免贴线
                category,
                fontsize=9,
                color='black',
                bbox=dict(
                    boxstyle='round,pad=0.4',  # 圆角矩形背景
                    facecolor=color,
                    alpha=0.3  # 半透明，不遮挡线条
                )
            )
        # 5. 调整坐标轴与标题
        plt.title('情绪变化趋势（完整数据）', fontsize=14, pad=20)
        plt.xlabel('时间', fontsize=12, labelpad=12)
        plt.ylabel('情绪得分', fontsize=12, labelpad=12)
        plt.ylim(-1.2, 1.2)  # 扩大y轴范围，容纳标注
        plt.xlim(df['timestamp'].min() - pd.Timedelta(seconds=5),
                 df['timestamp'].max() + pd.Timedelta(seconds=5))  # 时间轴留边
        # 6. 优化图例与布局
        plt.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True)
        plt.tight_layout()  # 自动调整布局，防止元素被截断
        # 7. 高分辨率保存图片（dpi=300）
        save_path = os.path.join(output_dir, "emotion_trend_complete.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图表释放资源
        logger.info(f"完整情绪趋势图已保存至: {save_path}")

    def _generate_comprehensive_evaluation(self, data, output_dir, session_id):
        """生成综合评估报告"""
        if not data['comprehensive']:
            return
        # 计算整体情绪指标
        scores = [item['combined_score'] for item in data['comprehensive']]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        trend = scores[-1] - scores[0]  # 情绪变化趋势
        # 确定整体情绪类别
        overall_category, _ = self.get_emotion_category(avg_score)
        # 生成报告文本（评估时间精确到分钟）
        report = f"""# 情绪综合评估报告
会话ID: {session_id}
分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}
## 整体评估
- 平均情绪得分: {avg_score:.2f} ({overall_category})
- 最高情绪得分: {max_score:.2f} {self.get_emotion_category(max_score)[0]}
- 最低情绪得分: {min_score:.2f} {self.get_emotion_category(min_score)[0]}
- 情绪变化趋势: {'上升' if trend > 0 else '下降' if trend < 0 else '平稳'} ({trend:.2f})
## 情绪分析详情
- 检测到的主要正面情绪: {self._get_dominant_positive_emotions(data)}
- 检测到的主要负面情绪: {self._get_dominant_negative_emotions(data)}
## 建议
{self._generate_suggestions(avg_score)}
"""
        # 保存报告
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"综合评估报告已保存至: {report_path}")

    def _get_dominant_positive_emotions(self, data):
        """获取主要正面情绪"""
        emotion_counts = defaultdict(int)
        for item in data['yolo_emotions']:
            for emotion, _ in item['emotions']:
                if EMOTION_WEIGHTS['yolo'].get(emotion, 0) > 0:
                    emotion_counts[emotion] += 1
        if not emotion_counts:
            return "无明显正面情绪"
        return ", ".join([e for e, _ in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3]])

    def _get_dominant_negative_emotions(self, data):
        """获取主要负面情绪"""
        emotion_counts = defaultdict(int)
        for item in data['yolo_emotions']:
            for emotion, _ in item['emotions']:
                if EMOTION_WEIGHTS['yolo'].get(emotion, 0) < 0:
                    emotion_counts[emotion] += 1
        if not emotion_counts:
            return "无明显负面情绪"
        return ", ".join([e for e, _ in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3]])

    def _generate_suggestions(self, score):
        """根据情绪得分生成建议"""
        if score <= -0.7:
            return "检测到严重负面情绪，建议进行专业心理干预，密切关注情绪变化，必要时寻求紧急帮助。"
        elif score <= -0.3:
            return "检测到中度负面情绪，建议进行适当的放松活动，与亲友交流，避免独处过久。"
        elif score <= 0.3:
            return "情绪处于中性状态，保持规律的作息和适当的运动有助于维持良好的情绪状态。"
        elif score <= 0.7:
            return "情绪状态良好，继续保持积极的生活方式和社交活动，有助于维持当前状态。"
        else:
            return "情绪状态非常积极，继续保持良好的心态和生活习惯。"


if __name__ == "__main__":
    # 再次确保使用Agg后端
    matplotlib.use('Agg')
    # 创建分析器实例
    analyzer = EmotionAnalyzer()
    # 分析所有会话数据
    analyzer.analyze_all_sessions()
    # 生成报告
    analyzer.generate_report()
    # 输出简要结果（显示完整时间路径）
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    print("\n情绪分析完成！")
    print(f"共分析 {len(analyzer.session_data)} 个会话")
    print(f"详细报告已生成在 analysis_reports/{current_time} 目录下")