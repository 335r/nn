from pydantic import BaseModel, Field
from typing import Optional


class HrCreate(BaseModel):
    """人事查询请求模型"""
    query: str = Field(..., min_length=1, max_length=500, description="人事查询内容")
    conversation_id: Optional[str] = Field("", description="会话ID")
    user_id: Optional[str] = Field(None, description="用户ID")


class HrResponse(BaseModel):
    """人事查询响应"""
    success: bool
    answer: Optional[str] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    error: Optional[str] = None
