from pydantic import BaseModel
from typing import Optional


class MentalChatCreate(BaseModel):
    message: str                  # 用户输入内容
    user_id: str                  # 用户ID
    conversation_id: Optional[str] = None  # 会话ID（多轮对话）


class MentalChatResponse(BaseModel):
    success: bool
    answer: str
    conversation_id: Optional[str] = None
