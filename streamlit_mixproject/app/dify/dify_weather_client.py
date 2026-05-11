import json

import httpx
from typing import Dict, Any, AsyncGenerator, Optional
from loguru import logger
from app.config.settings import settings

"""Dify Weather API客户端 - Agent Chat App (仅支持streaming)"""


class DifyWeatherClient:

    def __init__(self):
        self.api_key = settings.DIFY_WEATHER_API_KEY
        self.base_url = settings.DIFY_API_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    # Agent Chat App 不支持 blocking，使用 streaming 模式
    async def run_workflow_streaming(
            self,
            inputs: Dict[str, Any],
            response_mode: str = "streaming",
            user_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        data = {
            "inputs": inputs,
            "query": inputs.get("query", ""),  # Agent使用query字段
            "response_mode": "streaming",  # Agent只支持streaming
            "user": user_id,
            "conversation_id": inputs.get("conversation_id", "")
        }

        url = f"{self.base_url}/chat-messages"

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                logger.info(f"调用Dify Agent Weather API: {url}")
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
                                # Agent API 事件类型
                                if event_type == "agent_message":
                                    answer = event_data.get("answer")
                                    if answer:
                                        yield answer
                                elif event_type == "message":
                                    answer = event_data.get("answer")
                                    if answer:
                                        yield answer
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                logger.error(f"流式请求失败: {str(e)}")
                raise


# 创建全局客户端实例
dify_weather_client = DifyWeatherClient()
