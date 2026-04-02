from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes.admin_ai_models import router as admin_ai_models_router
from app.api.routes.admin_claims import router as admin_claims_router
from app.api.routes.admin_scan_logs import router as admin_scan_logs_router
from app.api.routes.auth import router as auth_router
from app.api.routes.family_members import router as family_members_router
from app.api.routes.test import router as test_router
from app.api.routes.ai import router as ai_router
from app.api.routes.claims import router as claims_router
from app.db.init import initialize_database


def build_error_response(
    status_code: int,
    errors: list[dict[str, str]],
    message: str = "Input validation failed.",
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "message": message,
            "errors": errors,
        },
    )


def normalize_http_exception_detail(detail: Any) -> tuple[str, list[dict[str, str]]]:
    if isinstance(detail, dict):
        message = str(detail.get("message", "Input validation failed."))
        raw_errors = detail.get("errors", [])
        if isinstance(raw_errors, list):
            errors = []
            for error in raw_errors:
                if isinstance(error, dict):
                    field = str(error.get("field", "non_field"))
                    error_message = str(error.get("message", "Invalid value."))
                    errors.append({"field": field, "message": error_message})
            if errors:
                return message, errors

    if isinstance(detail, str):
        return "Input validation failed.", [{"field": "non_field", "message": detail}]

    return "Input validation failed.", [{"field": "non_field", "message": "Invalid request."}]


async def validation_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    del request

    validation_error = cast(RequestValidationError, exc)

    errors = []
    for error in validation_error.errors():
        field_parts = [str(part) for part in error.get("loc", ()) if part != "body"]
        field = ".".join(field_parts)
        message = str(error.get("msg", "Invalid value."))
        if message.startswith("Value error, "):
            message = message.replace("Value error, ", "", 1)
        errors.append({"field": field, "message": message})

    return build_error_response(status_code=422, errors=errors)


async def http_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    del request

    http_error = cast(HTTPException, exc)
    message, errors = normalize_http_exception_detail(http_error.detail)
    return build_error_response(
        status_code=http_error.status_code,
        message=message,
        errors=errors,
    )


async def starlette_http_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    del request

    http_error = cast(StarletteHTTPException, exc)
    message, errors = normalize_http_exception_detail(http_error.detail)
    return build_error_response(
        status_code=http_error.status_code,
        message=message,
        errors=errors,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    initialize_database()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Python Backend", version="0.1.0", lifespan=lifespan)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, starlette_http_exception_handler)
    app.include_router(auth_router, prefix="/api")
    app.include_router(admin_ai_models_router, prefix="/api")
    app.include_router(admin_claims_router, prefix="/api")
    app.include_router(admin_scan_logs_router, prefix="/api")
    app.include_router(family_members_router, prefix="/api")
    app.include_router(test_router, prefix="/api")
    app.include_router(ai_router, prefix="/api")
    app.include_router(claims_router, prefix="/api")
    return app


app = create_app()
