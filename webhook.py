import hmac
import hashlib
from fastapi import Request, HTTPException


async def verify_github_signature(request: Request, secret: str) -> bytes:
    signature_header = request.headers.get("X-Hub-Signature-256")
    if not signature_header:
        raise HTTPException(status_code=403, detail="Missing X-Hub-Signature-256 header")

    body = await request.body()
    computed_signature = "sha256=" + hmac.new(
        key=secret.encode("utf-8"),
        msg=body,
        digestmod=hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(computed_signature, signature_header):
        raise HTTPException(status_code=403, detail="Invalid HMAC webhook signature")

    return body



