import json

import httpx
from typing import Dict, Any, AsyncGenerator, Optional
from loguru import logger
from app.config.settings import settings

"""Dify Translation API客户端 - 使用 Chat API"""


class DifyTranslationClient:

    def __init__(self):
        self.api_key = settings.DIFY_TRANSLATION_API_KEY
        self.base_url = settings.DIFY_API_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    # 用于运行Dify Chat API - blocking 模式
    async def run_workflow(
            self,
            inputs: Dict[str, Any],
            response_mode: str = "blocking",
            user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        # Chat API 格式 - 传递所有输入变量到 inputs 字段
        data = {
            "inputs": {
                "source_language": inputs.get("source_language", "中文"),
                "target_language": inputs.get("target_language", "英文"),
                # 如果有其他变量，也在这里添加
            },
            "query": inputs.get("topic", ""),  # 主要查询内容
            "response_mode": response_mode,
            "user": user_id,
            "conversation_id": ""  # 空字符串表示新对话
        }

        url = f"{self.base_url}/chat-messages"

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                logger.info(f"调用Dify Chat API: {url}")
                logger.debug(f"请求数据: {json.dumps(data, ensure_ascii=False)}")

                response = await client.post(url, headers=self.headers, json=data)
                response.raise_for_status()
                result = response.json()

                # Chat API 返回格式
                return {
                    "success": True,
                    "article": result.get("answer", ""),
                    "conversation_id": result.get("conversation_id"),
                    "message_id": result.get("id")
                }
            except httpx.HTTPError as e:
                # 尝试获取详细错误信息
                try:
                    error_body = e.response.text if hasattr(e, 'response') else str(e)
                    logger.error(f"Dify API错误详情: {error_body}")
                except:
                    pass
                logger.error(f"Dify API请求失败: {str(e)}")
                raise Exception(f"Dify API请求失败: {str(e)}")

    # 用于运行Dify Chat API - streaming 模式
    async def run_workflow_streaming(
            self,
            inputs: Dict[str, Any],
            response_mode: str = "streaming",
            user_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        data = {
            "inputs": {
                "source_language": inputs.get("source_language", "中文"),
                "target_language": inputs.get("target_language", "英文"),
            },
            "query": inputs.get("topic", ""),
            "response_mode": "streaming",
            "user": user_id,
            "conversation_id": ""
        }

        url = f"{self.base_url}/chat-messages"

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                logger.info(f"调用Dify Chat API: {url}")
                logger.debug(f"请求数据: {json.dumps(data, ensure_ascii=False)}")

                async with client.stream(
                        "POST",
                        url,
                        headers=self.headers,
                        json=data
                ) as response:
                    response.raise_for_status()
                    # Chat API 的 SSE 格式
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                event_data = json.loads(line[6:])
                                event_type = event_data.get("event")
                                # Chat API 使用 message 事件返回内容
                                if event_type == "message":
                                    answer = event_data.get("answer")
                                    if answer:
                                        yield answer
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                logger.error(f"流式请求失败: {str(e)}")
                raise


# 创建全局客户端实例
dify_translation_client = DifyTranslationClient()
