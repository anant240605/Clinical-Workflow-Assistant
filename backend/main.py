from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from agents.pipeline import ClinicalWorkflowPipeline
from models.database import Base, engine, get_db
from models.schemas import AnalyzeNoteRequest, AnalyzeNoteResponse, ReportResponse
from services.config import settings
from services.report_repository import ReportRepository


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title=settings.app_name,
    description="Agent-based MVP for clinical note analysis and report generation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


@app.post("/analyze-note", response_model=AnalyzeNoteResponse)
async def analyze_note(
    payload: AnalyzeNoteRequest,
    db: Session = Depends(get_db),
) -> AnalyzeNoteResponse:
    pipeline = ClinicalWorkflowPipeline()
    report = await pipeline.run(payload.note)

    repository = ReportRepository(db)
    repository.save(report)
    return report


@app.get("/report/{report_id}", response_model=ReportResponse)
def get_report(report_id: UUID, db: Session = Depends(get_db)) -> ReportResponse:
    repository = ReportRepository(db)
    report = repository.get(str(report_id))
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return report
