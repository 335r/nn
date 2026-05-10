import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi import status
from starlette.responses import StreamingResponse, HTMLResponse

from app.dify.dify_ccf_client import dify_ccf_client
from app.schemas.dify_ccf import CodeConvertResponse, CodeConvertCreate

dify_ccf = APIRouter(prefix="/code-convert", tags=["代码转换"])


@dify_ccf.post(
    "/convert",
    response_model=CodeConvertResponse)
async def convert_code(
        code_data: CodeConvertCreate
):
    try:
        inputs = {
            "default_input": code_data.default_input,
            "Target_code": code_data.target_code
        }
        result = await dify_ccf_client.run_workflow(
            inputs=inputs,
            user_id=code_data.user_id
        )
        return CodeConvertResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"代码转换失败: {str(e)}"
        )


"""
流式代码转换
返回Server-Sent Events (SSE) 流
"""


@dify_ccf.post("/convert-stream")
async def convert_code_stream(
        code_data: CodeConvertCreate
):
    async def event_generator():
        """生成SSE事件流"""
        try:
            # 先发送一个开始事件
            yield "event: start\ndata: {\"message\": \"开始转换代码...\"}\n\n"

            inputs = {
                "default_input": code_data.default_input,
                "Target_code": code_data.target_code
            }
            user_id = code_data.user_id

            full_content = ""
            # 异步迭代：从Dify客户端获取流式响应
            async for chunk in dify_ccf_client.run_workflow_streaming(
                    inputs=inputs,
                    user_id=user_id
            ):
                if chunk:
                    full_content += chunk
                    # 发送内容片段
                    event_data = json.dumps({"chunk": chunk, "is_complete": False})
                    yield f"data: {event_data}\n\n"
                    await asyncio.sleep(0.01)

            # 发送完成事件
            completion_event = json.dumps({
                "message": "代码转换完成",
                "is_complete": True,
                "total_length": len(full_content)
            })
            yield f"event: complete\ndata: {completion_event}\n\n"
        except Exception as e:
            error_event = json.dumps({"error": str(e), "is_complete": True})
            yield f"event: error\ndata: {error_event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@dify_ccf.get("/test-stream", response_class=HTMLResponse)
async def stream_test_page(request: Request):
    """流式接口测试页面"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>代码流式转换测试</title>
        <style>
            body { font-family: Arial; padding: 20px; }
            #output { 
                border: 1px solid #ccc; 
                padding: 20px; 
                min-height: 300px;
                white-space: pre-wrap;
                background: #f5f5f5;
                font-family: monospace;
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
            select { padding: 8px; margin: 5px; }
        </style>
    </head>
    <body>
        <h2>代码流式转换测试</h2>

        <div>
            <textarea id="code" rows="5" style="width: 500px; padding: 8px;" placeholder="输入代码...">print('hello world')</textarea>
        </div>
        <div>
            <select id="target" style="padding: 8px;">
                <option value="JAVA">转换为JAVA</option>
                <option value="Python">Python代码</option>
            </select>
            <button onclick="startStream()">开始转换</button>
            <button onclick="stopStream()" style="background: #dc3545;">停止</button>
        </div>

        <div class="status" id="status">准备就绪</div>

        <div id="output"></div>

        <script>
            let controller = null;

            async function startStream() {
                const code = document.getElementById('code').value;
                const target = document.getElementById('target').value;
                const output = document.getElementById('output');
                const status = document.getElementById('status');

                output.textContent = '';
                status.textContent = '正在连接...';

                if (controller) controller.abort();
                controller = new AbortController();

                try {
                    const response = await fetch('/api/v1/code-convert/convert-stream', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            default_input: code,
                            target_code: target,
                            user_id: 'test-user'
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
                            status.textContent = '转换完成';
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
                                        output.textContent += data.chunk;
                                        output.scrollTop = output.scrollHeight;
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
