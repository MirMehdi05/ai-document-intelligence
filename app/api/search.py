from fastapi import APIRouter


router = APIRouter()


@router.get("/search")
def search(query: str):

    return {
        "query": query,
        "message": "Search endpoint created"
    }