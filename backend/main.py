from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from database import engine, Base, SessionLocal
import models
from schemas import DrawingCreate, DrawingResponse


Base.metadata.create_all(bind=engine)

app = FastAPI(title="AirNote API")


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


@app.get("/")
def root():
    return {
        "message": "AirNote Backend is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/drawings", response_model=DrawingResponse)
def create_drawing(
    drawing: DrawingCreate,
    db: Session = Depends(get_db)
):
    new_drawing = models.Drawing(
        name=drawing.name,
        stroke_data=drawing.stroke_data
    )

    db.add(new_drawing)
    db.commit()
    db.refresh(new_drawing)

    return new_drawing
@app.get("/drawings", response_model=list[DrawingResponse])
def get_drawings(db: Session = Depends(get_db)):
    return db.query(models.Drawing).all()