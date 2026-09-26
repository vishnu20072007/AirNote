from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend.database import engine, Base, SessionLocal
from backend import models
from backend.schemas import (
    DrawingCreate,
    DrawingResponse,
    DrawingUpdate
)


# =========================================================
# DATABASE
# =========================================================

Base.metadata.create_all(bind=engine)


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="AirNote API",
    version="1.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# DATABASE SESSION
# =========================================================

def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "message": "AirNote Backend is running"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# =========================================================
# CREATE DRAWING
# POST /drawings
# =========================================================

@app.post(
    "/drawings",
    response_model=DrawingResponse
)
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


# =========================================================
# GET ALL DRAWINGS
# GET /drawings
# =========================================================

@app.get(
    "/drawings",
    response_model=list[DrawingResponse]
)
def get_drawings(
    db: Session = Depends(get_db)
):

    drawings = (
        db.query(models.Drawing)
        .all()
    )

    return drawings


# =========================================================
# GET SINGLE DRAWING
# GET /drawings/{drawing_id}
# =========================================================

@app.get(
    "/drawings/{drawing_id}",
    response_model=DrawingResponse
)
def get_drawing(
    drawing_id: int,
    db: Session = Depends(get_db)
):

    drawing = (
        db.query(models.Drawing)
        .filter(
            models.Drawing.id == drawing_id
        )
        .first()
    )

    if drawing is None:

        raise HTTPException(
            status_code=404,
            detail="Drawing not found"
        )

    return drawing


# =========================================================
# UPDATE DRAWING
# PUT /drawings/{drawing_id}
# =========================================================

@app.put(
    "/drawings/{drawing_id}",
    response_model=DrawingResponse
)
def update_drawing(
    drawing_id: int,
    drawing: DrawingUpdate,
    db: Session = Depends(get_db)
):

    existing_drawing = (
        db.query(models.Drawing)
        .filter(
            models.Drawing.id == drawing_id
        )
        .first()
    )

    if existing_drawing is None:

        raise HTTPException(
            status_code=404,
            detail="Drawing not found"
        )

    existing_drawing.name = drawing.name

    existing_drawing.stroke_data = (
        drawing.stroke_data
    )

    db.commit()

    db.refresh(existing_drawing)

    return existing_drawing


# =========================================================
# DELETE DRAWING
# DELETE /drawings/{drawing_id}
# =========================================================

@app.delete(
    "/drawings/{drawing_id}"
)
def delete_drawing(
    drawing_id: int,
    db: Session = Depends(get_db)
):

    drawing = (
        db.query(models.Drawing)
        .filter(
            models.Drawing.id == drawing_id
        )
        .first()
    )

    if drawing is None:

        raise HTTPException(
            status_code=404,
            detail="Drawing not found"
        )

    db.delete(drawing)

    db.commit()

    return {
        "message": "Drawing deleted successfully",
        "id": drawing_id
    }