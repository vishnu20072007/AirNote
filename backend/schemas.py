from pydantic import BaseModel


class DrawingCreate(BaseModel):
    name: str = "Untitled Drawing"
    stroke_data: str


class DrawingResponse(BaseModel):
    id: int
    name: str
    stroke_data: str

    class Config:
        from_attributes = True


class DrawingUpdate(BaseModel):
    name: str
    stroke_data: str