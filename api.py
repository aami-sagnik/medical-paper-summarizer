from summary import summarize
from fastapi import APIRouter

router = APIRouter()

@router.get("/summary/{id}")
def summary(id: str):
    return {
        "id": id,
        "summary": summarize(f"https://pubmed.ncbi.nlm.nih.gov/{id}/")
    }
