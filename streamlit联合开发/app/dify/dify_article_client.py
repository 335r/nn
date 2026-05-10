import json

import httpx
from typing import Dict, Any, AsyncGenerator, Optional
from loguru import logger
from app.config.settings import settings

"""Dify Article API客户端"""
class DifyArticleClient:

    # 从配置中读取API密钥、工作流ID和基础URL，并构建请求头。
    def __init__(self):
        self.api_key = settings.DIFY_ARTICLE_API_KEY
        self.workflow_id = settings.DIFY_ARTICLE_WORKFLOW_ID
        self.base_url = settings.DIFY_API_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }


    # 用于运行Dify工作流——blocking
    async def run_workflow(
        self,
        inputs: Dict[str, Any],  # 工作流输入参数
        response_mode: str = "blocking", # response_mode: 响应模式 (blocking, streaming)
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        data = {
            "inputs": inputs,
            "response_mode": response_mode,
            "user": user_id,
            "files": []  # 如果有文件上传，可以在这里添加
        }
        # 为了简化，这里直接拼接URL，实际可以使用httpx的params参数
        endpoint = f"workflows/run?workflow_id={self.workflow_id}"
        url = f"{self.base_url}/{endpoint}"  # 构建完整的请求URL

        async with httpx.AsyncClient(timeout=60.0) as client:  # 创建异步HTTP客户端上下文管理器
            try:
                logger.info(f"调用Dify API: {endpoint}")
                response = await client.post(url, headers=self.headers,
                                             json=data)  # 发送POST请求,使用异步方式发送请求,传递headers（包含认证信息）和json数据
                response.raise_for_status()  # 检查HTTP状态码。如果状态码不是2xx，会抛出httpx.HTTPError异常
                result = response.json()
                # 解析结果
                if result.get("data",{}).get("status") == "succeeded":
                    outputs = result.get("data", {}).get("outputs", {})
                    return {
                        "success": True,
                        "article": outputs.get("article", ""),
                        "workflow_run_id": result.get("workflow_run_id")
                    }
                else:
                    return {
                        "success": False,
                        "error": result.get("message", "未知错误")
                    }
            except httpx.HTTPError as e:
                logger.error(f"Dify API请求失败: {str(e)}")
                raise Exception(f"Dify API请求失败: {str(e)}")


    # 用于运行Dify工作流——streaming
    async def run_workflow_streaming(
        self,
        inputs: Dict[str, Any],
        response_mode: str = "streaming",  # response_mode: 响应模式 (blocking, streaming)
        user_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
            data = {
                "inputs": inputs,
                "response_mode": response_mode,
                "user": user_id,
                "files": []  # 如果有文件上传，可以在这里添加
            }
            endpoint = f"workflows/run?workflow_id={self.workflow_id}"
            url = f"{self.base_url}/{endpoint}"

            async with httpx.AsyncClient(timeout=None) as client: # 创建异步HTTP客户端上下文管理器
                try:
                    logger.info(f"调用Dify API: {endpoint}")
                    async with client.stream(
                        "POST",
                        url,
                        headers=self.headers,
                        json=data
                    ) as response:  # 发送POST请求,使用异步方式发送请求,传递headers（包含认证信息）和json数据
                        response.raise_for_status() # 检查HTTP状态码。如果状态码不是2xx，会抛出httpx.HTTPError异常
                        # 是处理 Server-Sent Events (SSE) 流的典型模式
                        async for line in response.aiter_lines():  #异步迭代器，异步逐行读取响应内容的方法
                            if line.startswith("data: "): #只有以 "data: " 开头的行才是有效的事件数据
                                try:
                                    event_data = json.loads(line[6:]) # 跳过 "data: " 这 6 个字符，获取实际数据部分
                                    if event_data.get("event") == "text_chunk":  #检查解析后的数据中 event 字段是否为 "text_chunk"
                                        text = event_data.get("data",{}).get("text")
                                        if text:
                                            yield text
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    logger.error(f"流式请求失败: {str(e)}")
                    raise



# 创建全局客户端实例
dify_article_client = DifyArticleClient()
