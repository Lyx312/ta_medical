import base64

from fastapi import APIRouter, File, UploadFile

from app.core.config import get_settings
from app.schemas.ai import AIMedicalScanResponse
from app.services.ai_service import DocumentImageInput, MedicalDocumentProcessor

router = APIRouter(prefix="/ai", tags=["ai"])
processor = MedicalDocumentProcessor(settings=get_settings())


@router.post("/scan-documents", response_model=AIMedicalScanResponse)
async def scan_documents(
    files: list[UploadFile] = File(...),
) -> AIMedicalScanResponse:
    images: list[DocumentImageInput] = []
    for file in files:
        content = await file.read()
        images.append(
            DocumentImageInput(
                filename=file.filename or "uploaded_file",
                mime_type=file.content_type or "application/octet-stream",
                base64_data=base64.b64encode(content).decode("utf-8"),
            )
        )

    return await processor.process(images)
