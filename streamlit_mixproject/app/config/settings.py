"""
应用配置模块
支持从环境变量加载配置
提供了完整、类型安全的配置管理方案，支持环境变量覆盖，便于不同环境的部署。
支持优先级：环境变量 > .env文件 > 代码默认值
"""
from typing import List  # 导入List类型，用于类型注解
from pydantic_settings import BaseSettings  # Pydantic的BaseSettings，用于配置管理
from functools import lru_cache  # LRU缓存装饰器，用于缓存配置实例
import os  # 操作系统模块，用于目录操作


class Settings(BaseSettings):  # 继承BaseSettings，支持环境变量和.env文件
    # 应用配置
    APP_NAME: str = "第一个FastAPI系统"  # 应用名称，默认值"第一个FastAPI系统"
    APP_VERSION: str = "1.0.0"  # 应用版本，默认值"1.0.0"
    DEBUG: bool = True  # 调试模式开关，默认True

    # 服务器配置
    HOST: str = "127.0.0.1"  # 服务器监听地址，默认本地回环地址
    PORT: int = 8000  # 服务器端口，默认8000

    # # 数据库配置
    # # 使用异步MySQL驱动
    # DATABASE_URL: str = "mysql+aiomysql://root:root@localhost:3306/test_db?charset=utf8mb4" # 数据库连接URL，格式：dialect+driver://username:password@host:port/database?options
    # DATABASE_POOL_SIZE: int = 20  # 数据库连接池大小，默认20
    # DATABASE_MAX_OVERFLOW: int = 40  # 连接池最大溢出数量，默认40
    # DATABASE_POOL_RECYCLE: int = 3600  # 连接回收时间（秒），默认3600秒（1小时）
    # DATABASE_ECHO: bool = True  # 是否打印SQL语句，调试用，默认True

    # # JWT配置
    # """
    # JWT（JSON Web Token） 是一种用于安全传输信息的开放标准（RFC 7519）。它是一种紧凑的、自包含的令牌格式，通常用于认证和授权。
    # """
    # SECRET_KEY: str = "your-secret-key-change-in-production"  # JWT密钥，生产环境需修改
    # ALGORITHM: str = "HS256"  # JWT签名算法，默认HS256
    # ACCESS_TOKEN_EXPIRE_MINUTES: int = 30  # 访问令牌过期时间（分钟），默认30

    # CORS配置（跨域资源共享）
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3001"]
    # 允许跨域请求的源地址，通常为前端地址

    # # 分页配置
    # DEFAULT_PAGE_SIZE: int = 10  # 默认每页数据条数
    # MAX_PAGE_SIZE: int = 100  # 最大每页数据条数，防止过大数据量请求

    # # 文件上传配置
    # MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB  # 最大上传文件大小，10MB
    # UPLOAD_DIR: str = "./uploads"  # 上传文件存储目录
    # ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "gif", "pdf"]  # 允许的文件扩展名

    # 日志配置
    LOG_LEVEL: str = "INFO"  # 日志级别：DEBUG/INFO/WARNING/ERROR/CRITICAL
    LOG_FILE: str = "./logs/app.log"  # 日志文件路径

    # API文档配置（FastAPI相关）
    DOCS_URL: str = "/docs"  # Swagger UI文档路径
    REDOC_URL: str = "/redoc"  # ReDoc文档路径
    OPENAPI_URL: str = "/openapi.json"  # OpenAPI规范JSON文件路径

    class Config:  # Pydantic配置类
        env_file = ".env"  # 指定环境变量文件路径
        case_sensitive = True  # 注意：这里设置为True，表示环境变量名大小写敏感
        # 这意味着环境变量名必须完全匹配字段名的大小写

    # Dify配置
    # 智能文章生成器
    DIFY_API_BASE_URL: str = "http://localhost/v1"
    DIFY_ARTICLE_API_KEY: str = "app-YwFNDQ1JuHqAo3ienulD3KRP"
    DIFY_ARTICLE_WORKFLOW_ID: str = "24f19b91-d92d-47c9-aa3d-77974fef1cd3"


    # 口算做题王
    DIFY_MENTAL_API_KEY: str = "app-flCsqJLyGQYGcbPxU533CfcO"
    DIFY_MENTAL_WORKFLOW_ID: str = "app/d21faed0-4edb-4658-9319-ce4e641af603"


    # 翻译助手
    DIFY_TRANSLATION_API_KEY: str = "app-dPlTmKbfv2ykVeeRWSId9JwG"
    DIFY_TRANSLATION_WORKFLOW_ID: str = "ac544c9a-d255-4f99-b1a3-c4a251f15e6b"


    # 代码转换
    DIFY_CCF_API_KEY: str = "app-bQ3jbXblAmrHq4LKRmqvyoqm"
    DIFY_CCF_WORKFLOW_ID: str = "ad497ed2-3439-475b-8a55-500289a015cb"


    # 天气助手
    DIFY_WEATHER_API_KEY: str = "app-rK4DH1BBuU5Pv2Z7UbUgpqek"
    DIFY_WEATHER_WORKFLOW_ID: str = "65a09af4-eed5-4f66-bd45-b9c363385e3e"


    # 人事助手
    DIFY_HR_API_KEY: str = "app-jwW1Pc7p2apz0zFewVkimkL3"
    DIFY_HR_WORKFLOW_ID: str = "523337ab-7c75-443c-9a9d-fb5d79462347"





"""
使用@lru_cache()装饰器缓存配置实例
避免重复读取环境变量和解析
提高性能，确保单例模式
"""
@lru_cache()  # 使用LRU缓存装饰器，缓存函数结果
def get_settings() -> Settings:
    """获取配置实例（使用缓存）"""
    return Settings()  # 创建并返回Settings实例
# 全局配置实例
settings = get_settings()  # 调用函数获取缓存的配置实例



def init_dirs():
    """初始化必要的目录"""  # 函数文档字符串
    dirs = [  # 需要创建的目录列表
        "./logs",  # 日志目录
        "./uploads"  # 上传文件根目录
    ]

    for dir_path in dirs:  # 遍历目录列表
        os.makedirs(dir_path, exist_ok=True)  # 创建目录，exist_ok=True表示如果已存在则不报错
        print(f"✅ 创建目录: {dir_path}")  # 打印创建成功的消息
# 初始化目录
init_dirs()  # 调用函数创建所需目录