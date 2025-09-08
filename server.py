from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import pdfplumber
import tempfile
import os
import math
from fpdf import FPDF

app = FastAPI()


# ---------- Helpers ----------
def extract_pdf_to_csv(pdf_path: str, csv_path: str):
    all_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if table:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    all_data.append(df)
    if not all_data:
        raise ValueError("No tables found in PDF")
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(csv_path, index=False)
    return csv_path


def expense_analysis(file_path: str, period: str = "monthly"):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    elif file_path.endswith(".json"):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format")

    required_cols = {"date", "amount", "department"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    if period == "monthly":
        df["period"] = df["date"].dt.to_period("M").astype(str)  # format YYYY-MM
    elif period == "quarterly":
        df["period"] = df["date"].dt.to_period("Q").astype(str)  # format YYYY-QX
    elif period == "yearly":
        df["period"] = df["date"].dt.to_period("Y").astype(str)  # format YYYY

    breakdown_df = df.groupby(["department", "period"])["amount"].sum().reset_index()

    anomalies = []
    for dept, group in df.groupby("department"):
        avg = group["amount"].mean()
        threshold = avg * 2
        flagged = group[group["amount"] > threshold]
        anomalies.extend(flagged.to_dict(orient="records"))

    trends = df.groupby("period")["amount"].sum().reset_index().to_dict(orient="records")

    return {
        "breakdown": breakdown_df.to_dict(orient="records"),
        "anomalies": anomalies,
        "trends": trends
    }


def safe_json(obj):
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json(i) for i in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0
        return obj
    elif hasattr(obj, "strftime"):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    else:
        return obj


def generate_pdf_report(analysis, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Expense Analysis Report", 0, 1, "C")

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Breakdown by Department & Period", 0, 1)

    # Table headers
    pdf.set_font("Arial", "B", 10)
    pdf.cell(60, 8, "Department", 1)
    pdf.cell(50, 8, "Period", 1)
    pdf.cell(40, 8, "Amount", 1)
    pdf.ln()

    pdf.set_font("Arial", "", 10)
    for row in analysis["breakdown"]:
        pdf.cell(60, 8, str(row["department"]), 1)
        pdf.cell(50, 8, str(row["period"]), 1)
        pdf.cell(40, 8, f"{row['amount']:.2f}", 1)
        pdf.ln()

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Anomalies Detected", 0, 1)
    pdf.set_font("Arial", "", 10)
    if not analysis["anomalies"]:
        pdf.cell(0, 8, "No anomalies detected.", 0, 1)
    else:
        for row in analysis["anomalies"]:
            pdf.multi_cell(0, 8, f"{row['date']} - {row['department']} - {row['amount']:.2f} - {row.get('description', '')}")
            pdf.ln(1)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Overall Trends", 0, 1)
    pdf.set_font("Arial", "", 10)
    for row in analysis["trends"]:
        pdf.cell(0, 8, f"Period {row['period']}: Total Amount {row['amount']:.2f}", 0, 1)

    pdf.output(filename)


# ---------- Routes ----------
@app.post("/analyze")
async def analyze_file(file: UploadFile, period: str = Form("monthly")):
    tmp_file_path = None
    tmp_pdf_file = None
    try:
        suffix = os.path.splitext(file.filename)[1]
        fd, tmp_file_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)

        content = await file.read()
        with open(tmp_file_path, "wb") as f:
            f.write(content)

        if suffix.lower() == ".pdf":
            tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
            file_path = extract_pdf_to_csv(tmp_file_path, tmp_csv)
        else:
            file_path = tmp_file_path

        analysis = expense_analysis(file_path, period)
        analysis = safe_json(analysis)

        tmp_pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        generate_pdf_report(analysis, tmp_pdf_file)

        return {"status": "success", "download_link": f"https://fast-api-server-x1ks.onrender.com/download/{os.path.basename(tmp_pdf_file)}"}

    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


@app.get("/download/{file_name}")
def download_file(file_name: str):
    file_path = os.path.join(tempfile.gettempdir(), file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename="Expense_Report.pdf", media_type="application/pdf")
    return JSONResponse(status_code=404, content={"status": "error", "message": "File not found"})
