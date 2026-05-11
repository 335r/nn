import json

import httpx
from typing import Dict, Any, AsyncGenerator, Optional
from loguru import logger
from app.config.settings import settings

"""Dify Code Convert API客户端 - 使用 Completion API"""


class DifyCodeConvertClient:

    def __init__(self):
        self.api_key = settings.DIFY_CCF_API_KEY
        self.base_url = settings.DIFY_API_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    # 用于运行Dify Completion API - blocking 模式
    async def run_workflow(
            self,
            inputs: Dict[str, Any],
            response_mode: str = "blocking",
            user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        # Completion API 格式 - 所有输入都在 inputs 中
        data = {
            "inputs": {
                "Target_code": inputs.get("Target_code", "JAVA"),  # 必须是 JAVA 或 Python
                "default_input": inputs.get("default_input", "")  # 代码内容
            },
            "response_mode": response_mode,
            "user": user_id
        }

        url = f"{self.base_url}/completion-messages"

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                logger.info(f"调用Dify Completion API: {url}")
                logger.debug(f"请求数据: {json.dumps(data, ensure_ascii=False)}")

                response = await client.post(url, headers=self.headers, json=data)
                response.raise_for_status()
                result = response.json()

                return {
                    "success": True,
                    "answer": result.get("answer", ""),
                    "message_id": result.get("id")
                }
            except httpx.HTTPError as e:
                try:
                    error_body = e.response.text if hasattr(e, 'response') else str(e)
                    logger.error(f"Dify API错误详情: {error_body}")
                except:
                    pass
                logger.error(f"Dify API请求失败: {str(e)}")
                raise Exception(f"Dify API请求失败: {str(e)}")

    # 用于运行Dify Completion API - streaming 模式
    async def run_workflow_streaming(
            self,
            inputs: Dict[str, Any],
            response_mode: str = "streaming",
            user_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        data = {
            "inputs": {
                "Target_code": inputs.get("Target_code", "JAVA"),
                "default_input": inputs.get("default_input", "")
            },
            "response_mode": "streaming",
            "user": user_id
        }

        url = f"{self.base_url}/completion-messages"

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                logger.info(f"调用Dify Completion API: {url}")
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
dify_ccf_client = DifyCodeConvertClient()
