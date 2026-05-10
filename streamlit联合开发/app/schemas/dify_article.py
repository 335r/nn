from pydantic import BaseModel, Field
from typing import Optional


class ArticleCreate(BaseModel):
    """文章基础模型"""
    topic: str = Field(..., min_length=1, max_length=200, description="文章主题")
    user_id: Optional[str] = Field(None, description="用户ID")


class ArticleResponse(BaseModel):
    """文章生成响应"""
    success: bool
    article: Optional[str] = None
    workflow_run_id: Optional[str] = None
    error: Optional[str] = None
