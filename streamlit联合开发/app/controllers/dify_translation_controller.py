import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi import status
from starlette.responses import StreamingResponse, HTMLResponse

from app.dify.dify_translation_client import dify_translation_client
from app.schemas.dify_translation import ArticleResponse, ArticleCreate

dify_translation = APIRouter(prefix="/articles", tags=["文章生成"])


@dify_article.post(
    "/generate",
    response_model=ArticleResponse)
async def generate_article(
    translation_data: ArticleCreate
):
    try:
        inputs = {"topic": translation_data.topic}
        result = await dify_translation_client.run_workflow(
            inputs=inputs,
            user_id=translation_data.user_id
        )
        return ArticleResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"生成文章失败: {str(e)}"
        )


"""
流式生成文章
返回Server-Sent Events (SSE) 流
"""
@dify_article.post("/generate-stream")
async def generate_article_stream(
    translation_data: ArticleCreate
):
    async def event_generator():
        """生成SSE事件流"""
        try:
            # 先发送一个开始事件
            yield "event: start\ndata: {\"message\": \"开始生成文章...\"}\n\n"

            inputs = {"topic": translation_data.topic}
            user_id = translation_data.user_id

            full_content = ""
            # 异步迭代：从Dify客户端获取流式响应
            # 每次迭代获得一个文本块(chunk)
            async for chunk in dify_translation_client.run_workflow_streaming(inputs=inputs,user_id=translation_data.user_id):
                if chunk: # 检查chunk是否非空（过滤掉空数据）
                    full_content += chunk
                    # 发送内容片段
                    event_data = json.dumps({"chunk": chunk, "is_complete": False})  # json.dumps将Python字典转换为JSON格式
                    yield f"data: {event_data}\n\n"  # yield返回SSE格式的数据
                    await asyncio.sleep(0.01)  # 控制发送速度

            # 发送完成事件
            completion_event = json.dumps({
                "message": "文章生成完成",
                "is_complete": True,
                "total_length": len(full_content)
            })
            yield f"event: complete\ndata: {completion_event}\n\n"  # 客户端可以通过监听complete事件来处理完成逻辑
        except Exception as e:
            error_event = json.dumps({"error": str(e), "is_complete": True})
            yield f"event: error\ndata: {error_event}\n\n"
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 禁用Nginx缓冲
        }
    )


@dify_article.get("/test-stream", response_class=HTMLResponse)
async def stream_test_page(request: Request):
    """流式接口测试页面"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>文章流式生成测试</title>
        <style>
            body { font-family: Arial; padding: 20px; }
            #output { 
                border: 1px solid #ccc; 
                padding: 20px; 
                min-height: 300px;
                white-space: pre-wrap;
                background: #f5f5f5;
            }
            button { 
                padding: 10px 20px; 
                margin: 5px; 
                background: #007bff; 
                color: white; 
                border: none; 
                border-radius: 4px;
            }
            button:hover { background: #0056b3; }
            .status { color: #666; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h2>文章流式生成测试</h2>

        <div>
            <input type="text" id="topic" value="人工智能的未来" style="width: 300px; padding: 8px;">
            <button onclick="startStream()">开始生成</button>
            <button onclick="stopStream()" style="background: #dc3545;">停止</button>
        </div>

        <div class="status" id="status">准备就绪</div>

        <div id="output"></div>

        <script>
            let controller = null;

            async function startStream() {
                const topic = document.getElementById('topic').value;
                const output = document.getElementById('output');
                const status = document.getElementById('status');

                output.textContent = '';
                status.textContent = '正在连接...';

                // 创建 AbortController 以便停止请求
                if (controller) controller.abort();
                controller = new AbortController();

                try {
                    const response = await fetch('/api/v1/articles/generate-stream', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            topic: topic,
                            user_id: 'sunys'
                        }),
                        signal: controller.signal
                    });

                    if (!response.ok) throw new Error(`HTTP ${response.status}`);

                    status.textContent = '已连接，正在接收数据...';

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) {
                            status.textContent = '生成完成';
                            break;
                        }

                        const chunk = decoder.decode(value);
                        const lines = chunk.split('\\n');

                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                try {
                                    const data = JSON.parse(line.substring(6));
                                    console.log(data);
                                    if (data.chunk) {
                                        // 逐字显示效果
                                        for (let char of data.chunk) {
                                            output.textContent += char;
                                            // 滚动到底部
                                            output.scrollTop = output.scrollHeight;
                                            // 控制显示速度
                                            await new Promise(resolve => setTimeout(resolve, 10));
                                        }
                                    }
                                    if (data.message) {
                                        status.textContent = data.message;
                                    }
                                } catch (e) {
                                    console.warn('解析失败:', line);
                                }
                            }
                        }
                    }
                } catch (error) {
                    if (error.name !== 'AbortError') {
                        status.textContent = '错误: ' + error.message;
                    } else {
                        status.textContent = '已停止';
                    }
                } finally {
                    controller = null;
                }
            }

            function stopStream() {
                if (controller) {
                    controller.abort();
                    controller = null;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
