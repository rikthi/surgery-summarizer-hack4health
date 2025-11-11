from pydantic import BaseModel


class ProcessRequest(BaseModel):
    file_id: str
    model: str | None = None
