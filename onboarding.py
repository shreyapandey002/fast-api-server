from fastapi import FastAPI, UploadFile, Form
import os
import shutil
from typing import List
import smtplib
from email.message import EmailMessage

app = FastAPI()

# Required documents
required_docs = ["aadhar", "pan", "release_letter"]

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/init_user/")
async def init_user(email: str = Form(...)):
    """
    Initialize a new onboarding session for a user.
    Stateless: just ensures the folder exists.
    """
    folder = os.path.join(UPLOAD_DIR, email)
    os.makedirs(folder, exist_ok=True)

    # Determine which documents are already uploaded (if any)
    uploaded_docs = [
        doc.split(".pdf")[0]
        for doc in os.listdir(folder)
        if doc.endswith(".pdf") and doc.split(".pdf")[0] in required_docs
    ]

    missing_docs = [doc for doc in required_docs if doc not in uploaded_docs]

    return {
        "status": "initialized",
        "required_docs": required_docs,
        "next_doc": missing_docs[0] if missing_docs else None,
        "uploaded_docs": uploaded_docs,
        "missing_docs": missing_docs,
    }


@app.post("/upload_doc/")
async def upload_doc(
    email: str = Form(...), doc_type: str = Form(...), file: UploadFile = None
):
    """
    Upload a document for a user.
    Stateless: checks uploaded files to track progress.
    """
    folder = os.path.join(UPLOAD_DIR, email)
    if not os.path.exists(folder):
        return {"status": "error", "message": "User not initialized"}

    if not file:
        return {"status": "error", "message": "No file uploaded"}

    # Normalize doc_type
    doc_map = {doc.lower(): doc for doc in required_docs}
    normalized_doc = doc_map.get(doc_type.strip().lower())

    if not normalized_doc:
        return {
            "status": "error",
            "message": f"Unknown document type '{doc_type}'. Provide one of {required_docs}",
        }

    # Save uploaded file
    file_path = os.path.join(folder, f"{normalized_doc}.pdf")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Determine uploaded and missing documents
    uploaded_docs = [
        doc.split(".pdf")[0]
        for doc in os.listdir(folder)
        if doc.endswith(".pdf") and doc.split(".pdf")[0] in required_docs
    ]
    missing_docs = [doc for doc in required_docs if doc not in uploaded_docs]

    # Build doc_status map
    doc_status = {doc: ("completed" if doc in uploaded_docs else "incompleted") for doc in required_docs}

    if not missing_docs:
        # All documents uploaded → send email
        await send_email(email, folder)

        # Cleanup
        shutil.rmtree(folder)

        return {
            "status": "complete",
            "message": "All documents received. Email sent.",
            "doc_status": doc_status,
        }

    next_doc = missing_docs[0]
    return {
        "status": "incomplete",
        "message": f"{normalized_doc} uploaded successfully.",
        "missing_docs": missing_docs,
        "next_doc": next_doc,
        "doc_status": doc_status,
    }


async def send_email(user_email: str, folder: str):
    """
    Send an email to HR with all uploaded documents attached.
    """
    hr_email = "shreya.p@nighthack.in"
    msg = EmailMessage()
    msg["From"] = "noreply@example.com"
    msg["To"] = hr_email
    msg["Cc"] = "shreya.p@nighthack.in"
    msg["Subject"] = f"New Employee Onboarding - {user_email}"
    msg.set_content(f"All required documents for {user_email} have been collected and are attached.")

    # Attach all PDFs in the folder
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        with open(filepath, "rb") as fp:
            msg.add_attachment(fp.read(), maintype="application", subtype="pdf", filename=filename)

    # Mailtrap SMTP (blocking, suitable for small apps)
    with smtplib.SMTP("sandbox.smtp.mailtrap.io", 2525) as s:
        s.login("b6c2a4b4798d4f", "fad68538a2eee8")
        s.send_message(msg)
