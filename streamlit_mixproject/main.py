from fastapi import FastAPI, Request  # FastAPI核心类和请求对象
from fastapi.responses import JSONResponse  # JSON响应类
from fastapi.middleware.cors import CORSMiddleware  # CORS跨域中间件
from contextlib import asynccontextmanager  # 异步上下文管理器
import uvicorn  # ASGI服务器

from app.config.settings import settings  # 应用配置
from app.controllers.dify_article_controller import dify_article  # 导入路由器
from app.controllers.dify_mental_controller import dify_mental  # 导入路由器


@asynccontextmanager  # 异步上下文管理器装饰器
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    """
    print(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} 启动中...")

    # 启动时初始化
    # async with engine.begin() as conn:
    #     # 这里可以执行数据库初始化
    #     pass

    yield  # 分隔启动和关闭逻辑

    # 关闭时清理
    print(f"🛑 {settings.APP_NAME} 关闭中...")



# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,  # 应用标题（从配置读取）
    description="第一个FastAPI系统",  # 应用描述
    version=settings.APP_VERSION,  # 应用版本（从配置读取）
    docs_url="/docs" if settings.DEBUG else None,  # 调试模式开启文档
    redoc_url="/redoc" if settings.DEBUG else None,  # 调试模式开启ReDoc
    openapi_url="/openapi.json" if settings.DEBUG else None,  # 调试模式开启OpenAPI
    lifespan=lifespan  # 应用生命周期管理
)

# 配置CORS（跨域资源共享）
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,  # 允许的源（从配置读取）
    allow_credentials=True,  # 允许凭证（如cookies）
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有HTTP头
)

# 健康检查
@app.get("/", tags=["Root"])  # 根路径路由
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs" if settings.DEBUG else None  # 调试模式显示文档地址
    }

# 路由注册
app.include_router(dify_article, prefix="/api/v1")
app.include_router(dify_mental, prefix="/api/v1")

@app.get("/health", tags=["Health"])  # 健康检查路由
async def health_check():
    return {
        "status": "healthy",
        "database": "connected"  # 可以添加数据库连接检查
    }


# 开发服务器启动
if __name__ == "__main__":  # 直接运行此文件时执行
    uvicorn.run(
        "main:app",  # ASGI应用路径（模块名:应用实例）
        host=settings.HOST,  # 监听地址
        port=settings.PORT,  # 监听端口
        reload=settings.DEBUG,  # 调试模式开启热重载
        log_level="info"  # 日志级别
    )
