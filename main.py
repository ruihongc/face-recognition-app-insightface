import uvicorn
import streamsync.serve
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import os
import sys

@asynccontextmanager
async def lifespan_context(app: FastAPI):
    async with sub_asgi_app.router.lifespan_context(app):
        yield

root_asgi_app = FastAPI(lifespan=lifespan_context)

@root_asgi_app.get("/")
async def init():
    # return Response("""
    #     <h1>Welcome</h1>
    # """)
    return RedirectResponse("/app")

mode = sys.argv[1] if len(sys.argv) > 1 else "run"
sub_asgi_app = streamsync.serve.get_asgi_app(os.path.join(os.getcwd(), "app"), mode)

root_asgi_app.mount("/app", sub_asgi_app)

if __name__ == "__main__":
    uvicorn.run(root_asgi_app,
        host="127.0.0.1",
        # host="0.0.0.0",
        port=3005,
        log_level="warning",
        ws_ping_interval=5.0,
        ws_ping_timeout=None,
        timeout_keep_alive=10000000.0*60,
        # ws_ping_timeout=1e6,
        ws_max_size=streamsync.serve.MAX_WEBSOCKET_MESSAGE_SIZE)
