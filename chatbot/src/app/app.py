from fastapi import FastAPI
from app.routes import router

app = FastAPI()


async def lifespan():
    print("Startup")

    #init db 

    # init model 

    yield
    print("Shutdown")


app.include_router(router, prefix="/", tags=["RagBot"])
