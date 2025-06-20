import os
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
import openai
from pypdf import PdfReader, PdfWriter
from tempfile import NamedTemporaryFile

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/fill-pdf/")
async def fill_pdf(file: UploadFile):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF files only")

    temp_pdf = NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_pdf.write(await file.read())
    temp_pdf.close()

    reader = PdfReader(temp_pdf.name)
    fields = reader.get_fields()
    
    if not fields:
        raise HTTPException(status_code=400, detail="No fillable fields detected")

    field_names = list(fields.keys())

    # Example prompt for GPT:
    prompt = f"""
    Please provide realistic and appropriate example data for each of the following form fields:
    {field_names}
    
    Provide your response in JSON format like:
    {{"field_name": "value", ...}}
    """

    completion = openai.ChatCompletion.create(
        model="gpt-4-turbo",  # or "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        field_values = eval(completion.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Response Parsing Failed: {str(e)}")

    writer = PdfWriter()
    writer.add_page(reader.pages[0])

    # Fill PDF fields with AI-provided data
    writer.update_page_form_field_values(writer.pages[0], field_values)

    filled_pdf = NamedTemporaryFile(delete=False, suffix=".pdf")
    with open(filled_pdf.name, "wb") as output_pdf:
        writer.write(output_pdf)

    return FileResponse(filled_pdf.name, filename="filled_form.pdf")
