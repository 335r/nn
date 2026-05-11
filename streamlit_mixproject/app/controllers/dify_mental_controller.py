import asyncio
import json

from fastapi import APIRouter, HTTPException, Request, status
from starlette.responses import StreamingResponse, HTMLResponse

from app.dify.dify_mental_client import dify_mental_client
from app.schemas.dify_mental import MentalChatCreate, MentalChatResponse

dify_mental = APIRouter(prefix="/mental", tags=["口算做题王"])


# 阻塞式对话接口
@dify_mental.post(
    "/chat",
    response_model=MentalChatResponse
)
async def mental_chat(
        mental_data: MentalChatCreate
):
    try:
        result = await dify_mental_client.run_workflow(
            inputs={},
            query=mental_data.message,
            conversation_id=mental_data.conversation_id or "",
            user_id=mental_data.user_id
        )
        return MentalChatResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"口算对话失败: {str(e)}"
        )


@dify_mental.post("/chat-stream")
async def mental_chat_stream(
        mental_data: MentalChatCreate
):
    async def event_generator():
        try:
            yield "event: start\ndata: {\"message\": \"口算王连接成功...\"}\n\n"

            full_content = ""
            current_conv_id = mental_data.conversation_id or ""

            async for chunk_data in dify_mental_client.run_workflow_streaming(
                inputs={},
                query=mental_data.message,
                conversation_id=mental_data.conversation_id or "",
                user_id=mental_data.user_id
            ):
                text = chunk_data.get("answer", "")
                conv_id = chunk_data.get("conversation_id", "")

                if conv_id:
                    current_conv_id = conv_id  # 刷新会话ID

                if text:
                    full_content += text
                    event_data = json.dumps({
                        "chunk": text,
                        "conversation_id": current_conv_id,  # 关键！返回给前端
                        "is_complete": False
                    })
                    yield f"data: {event_data}\n\n"
                    await asyncio.sleep(0.01)

            completion_event = json.dumps({
                "message": "口算回答完成",
                "conversation_id": current_conv_id,
                "is_complete": True,
                "total_length": len(full_content)
            })
            yield f"event: complete\ndata: {completion_event}\n\n"

        except Exception as e:
            error_event = json.dumps({
                "error": str(e),
                "is_complete": True
            })
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


@dify_mental.get("/test-stream", response_class=HTMLResponse)
async def math_stream_test_page(request: Request):
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>口算做题王 - 流式测试</title>
        <style>
            body { font-family: Arial; padding: 20px; }
            #output { 
                border: 1px solid #ccc; 
                padding: 20px; 
                min-height: 300px;
                white-space: pre-wrap;
                background: #f9f9f9;
                font-size: 16px;
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
            #conversationId { color: #007bff; font-weight: bold; }
        </style>
    </head>
    <body>
        <h2>🧮 口算做题王（流式对话）</h2>

        <div>
            <input type="text" id="message" placeholder="输入：生成口算题 / 3+5=8" style="width: 400px; padding: 8px;">
            <button onclick="startStream()">发送</button>
            <button onclick="stopStream()" style="background: #dc3545;">停止</button>
            <button onclick="clearOutput()" style="background: #28a745;">清空</button>
        </div>

        <div class="status" id="status">准备就绪</div>
        <div>会话ID：<span id="conversationId">-</span></div>
        <div id="output"></div>

        <script>
            let controller = null;
            let conversationId = "";

            async function startStream() {
                const message = document.getElementById('message').value;
                const output = document.getElementById('output');
                const status = document.getElementById('status');
                const convIdSpan = document.getElementById('conversationId');

                if (!message.trim()) return;
                document.getElementById('message').value = '';

                output.textContent += '\\n👤 你：' + message + '\\n';
                output.textContent += '🤖 口算王：';
                status.textContent = '思考中...';

                if (controller) controller.abort();
                controller = new AbortController();

                try {
                    const response = await fetch('/api/v1/mental/chat-stream', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            message: message,
                            user_id: 'student_001',
                            conversation_id: conversationId
                        }),
                        signal: controller.signal
                    });

                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) {
                            status.textContent = '回答完成';
                            break;
                        }

                        const chunk = decoder.decode(value);
                        const lines = chunk.split('\\n');

                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                try {
                                    const data = JSON.parse(line.substring(6));

                                    // 自动保存会话ID
                                    if (data.conversation_id) {
                                        conversationId = data.conversation_id;
                                        convIdSpan.textContent = conversationId;
                                    }

                                    if (data.chunk) {
                                        for (let char of data.chunk) {
                                            output.textContent += char;
                                            output.scrollTop = output.scrollHeight;
                                            await new Promise(resolve => setTimeout(resolve, 10));
                                        }
                                    }
                                    if (data.message) {
                                        status.textContent = data.message;
                                    }
                                } catch (e) {}
                            }
                        }
                    }
                } catch (error) {
                    if (error.name !== 'AbortError') {
                        status.textContent = '错误：' + error.message;
                    }
                } finally {
                    controller = null;
                }
            }

            function stopStream() {
                if (controller) {
                    controller.abort();
                    controller = null;
                    document.getElementById('status').textContent = '已停止';
                }
            }

            function clearOutput() {
                document.getElementById('output').textContent = '';
                conversationId = '';
                document.getElementById('conversationId').textContent = '-';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
