from pydantic import BaseModel, Field
from typing import Optional


class CodeConvertCreate(BaseModel):
    """代码转换请求模型"""
    default_input: str = Field(..., min_length=1, max_length=5000, description="输入代码")
    target_code: str = Field(..., description="目标语言 (JAVA 或 Python)")
    user_id: Optional[str] = Field(None, description="用户ID")


class CodeConvertResponse(BaseModel):
    """代码转换响应"""
    success: bool
    answer: Optional[str] = None
    message_id: Optional[str] = None
    error: Optional[str] = None
