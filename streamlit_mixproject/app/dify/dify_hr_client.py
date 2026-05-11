import json

import httpx
from typing import Dict, Any, AsyncGenerator, Optional
from loguru import logger
from app.config.settings import settings

"""Dify HR API客户端 - Agent Chat App (仅支持streaming/blocking)"""


class DifyHrClient:

    def __init__(self):
        self.api_key = settings.DIFY_HR_API_KEY
        self.base_url = settings.DIFY_API_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    # 阻塞模式调用 (blocking)
    async def run_workflow(
            self,
            inputs: Dict[str, Any],
            query: str,
            response_mode: str = "blocking",
            conversation_id: Optional[str] = "",
            user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        data = {
            "inputs": inputs,
            "query": query,
            "response_mode": response_mode,
            "user": user_id,
            "conversation_id": conversation_id
        }

        url = f"{self.base_url}/chat-messages"

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                logger.info(f"调用Dify Agent HR API (blocking): {url}")
                logger.debug(f"请求数据: {json.dumps(data, ensure_ascii=False)}")

                response = await client.post(
                    url,
                    headers=self.headers,
                    json=data
                )
                response.raise_for_status()
                response_data = response.json()

                # 构造统一返回格式
                return {
                    "success": True,
                    "answer": response_data.get("answer", ""),
                    "conversation_id": response_data.get("conversation_id", ""),
                    "message_id": response_data.get("message_id", "")
                }
            except Exception as e:
                logger.error(f"阻塞请求失败: {str(e)}")
                raise

    # 流式模式调用 (streaming)
    async def run_workflow_streaming(
            self,
            inputs: Dict[str, Any],
            response_mode: str = "streaming",
            user_id: Optional[str] = None,
            conversation_id: Optional[str] = ""
    ) -> AsyncGenerator[str, None]:
        data = {
            "inputs": inputs,
            "query": inputs.get("query", ""),
            "response_mode": "streaming",
            "user": user_id,
            "conversation_id": conversation_id
        }

        url = f"{self.base_url}/chat-messages"

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                logger.info(f"调用Dify Agent HR API (streaming): {url}")
                logger.debug(f"请求数据: {json.dumps(data, ensure_ascii=False)}")

                async with client.stream(
                        "POST",
                        url,
                        headers=self.headers,
                        json=data
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                event_data = json.loads(line[6:])
                                event_type = event_data.get("event")
                                if event_type in ["agent_message", "message"]:
                                    answer = event_data.get("answer")
                                    if answer:
                                        yield answer
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                logger.error(f"流式请求失败: {str(e)}")
                raise


# 创建全局客户端实例
dify_hr_client = DifyHrClient()
