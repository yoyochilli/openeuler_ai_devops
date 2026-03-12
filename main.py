import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
# 导入你整理好的逻辑
from final_logic import agent_executor
import json
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

# 配置 CORS，允许任何前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 定义 OpenAI 标准 API 数据格式
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    user_input = request.messages[-1].content
    print(f"🚀 [请求] {user_input}")

    try:
        # 1. 调用你的 RAG + Agent 逻辑
        response = agent_executor.invoke({"input": user_input})
        answer = response["output"]

        # 获取当前时间戳和响应 ID
        current_time = int(time.time())
        resp_id = f"chatcmpl-{current_time}"

        # 2. 如果前端请求流式输出 (stream=True)
        if request.stream:
            async def generate_stream():
                # 按照 OpenAI SSE 格式伪装成流式吐出（这里为了简单，直接一次性吐出所有内容）
                chunk = {
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "created": current_time,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": answer},
                        "finish_reason": "stop"
                    }]
                }
                # 必须以 data: 开头，并以 \n\n 结尾
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        # 3. 如果前端请求非流式输出 (stream=False)
        payload = {
            "id": resp_id,
            "object": "chat.completion",
            "created": current_time,
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        return JSONResponse(content=payload, media_type="application/json")

    except Exception as e:
        print(f"❌ [错误] {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn

    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8000)