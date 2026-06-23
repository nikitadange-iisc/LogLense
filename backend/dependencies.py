"""FastAPI dependencies shared across routes."""


async def get_current_user() -> str:
    return "local"
