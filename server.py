#!/Users/aidanburrowes/miniconda3/envs/pdf-filler/bin/python
"""
MCP stdio server – Vision-Enhanced LLM-powered PDF form filler.

This script uses an LLM to fill in PDF forms.

Inputs:
    pdf_path : path/filename of the blank PDF to be filled.
    context  : A block of free-text from the user with the info to fill.

Outputs:
    If information is missing, it asks a clarifying question:
    {"status":"need_info", "message": "I couldn't find your..."}

    If the form is filled successfully:
    {"status":"done", "filename": "...", "filled_pdf": "<base64...>", "local_path": "..."}
"""

import io
import os
import json
import uuid
import base64
import logging
import sys
import textwrap

from mcp.server.fastmcp import FastMCP
from pypdf import PdfReader, PdfWriter
from pypdf.generic import BooleanObject, NameObject
import google.generativeai as genai
from pdf2image import convert_from_bytes # For vision capability

# ----- Configuration ----------------------------------------------------

# IMPORTANT: You must configure your Gemini API key.
# 1. Get your key from Google AI Studio: https://aistudio.google.com/app/apikey
# 2. Set it as an environment variable:
#    export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
try:
    genai.configure()
except Exception as e:
    print("FATAL: Could not configure Gemini API. Is GOOGLE_API_KEY set?", file=sys.stderr)
    sys.exit(1)

# Switched to Flash model for speed and to avoid rate limits.
LLM_MODEL = "gemini-1.5-flash-latest" 

log = logging.getLogger("smart-pdf-fill")
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ----- PDF Helper Function ----------------------------------------------

def fill_pdf_bytes(pdf_bytes: bytes, values: dict[str, str]) -> bytes:
    """Fills a PDF form with the given values across ALL pages."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()
    writer.append(reader)

    for page in writer.pages:
        try:
            writer.update_page_form_field_values(page, values)
        except Exception as e:
            log.warning(f"Could not fill some fields on a page: {e}")

    try:
        if writer.root_object and writer.root_object.get("/AcroForm"):
            writer.root_object["/AcroForm"][NameObject("/NeedAppearances")] = BooleanObject(True)
    except Exception as e:
        log.warning(f"Could not set NeedAppearances flag: {e}")

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()

# ----- Core LLM-powered Logic ---------------------------------

def map_fields_with_vision(pdf_bytes: bytes, fields: dict) -> dict:
    """Uses a vision model to map internal field names to human-readable labels."""
    log.info("Starting vision-based field mapping...")
    try:
        image = convert_from_bytes(pdf_bytes, dpi=200, first_page=1, last_page=1)[0]
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
    except Exception as e:
        log.error(f"Failed to convert PDF to image for vision analysis: {e}. Falling back to non-vision mode.")
        return {}
    
    model = genai.GenerativeModel(LLM_MODEL)
    field_ids = list(fields.keys())

    prompt = textwrap.dedent(f"""
        You are a highly accurate document analysis system.
        Analyze the provided image of a PDF form and a list of internal 'field_ids'.
        Your task is to identify the visible text label associated with each 'field_id'.

        List of internal field_ids:
        {json.dumps(field_ids, indent=2)}

        Return your findings as a single, valid JSON object where keys are the internal 'field_ids' and values are the corresponding human-readable labels you identified.
        If you cannot confidently determine the label for a field, omit it from the JSON or set its value to null.
        Output ONLY the JSON.
    """)
    
    try:
        response = model.generate_content([prompt, {"mime_type": "image/png", "data": img_bytes}])
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        vision_mapping = json.loads(json_text)
        log.info(f"Vision model successfully mapped {len(vision_mapping)} fields.")
        return vision_mapping
    except Exception as e:
        log.error(f"Failed to get or parse vision response from LLM: {e}")
        return {}


def extract_answers_with_llm(human_readable_fields: list[str], context: str) -> dict:
    """
    Uses an LLM to find answers for fields, enhancing subjective answers.
    """
    model = genai.GenerativeModel(LLM_MODEL)
    field_list_str = ", ".join([f'"{field}"' for field in human_readable_fields])

    # This prompt is now enhanced to act as a career coach for subjective fields.
    prompt = textwrap.dedent(f"""
        You are a sophisticated AI assistant and career coach that helps users fill out forms professionally.
        Your task is to analyze a user's 'CONTEXT' and provide the best possible answers for a list of 'FIELDS TO FILL'.

        CONTEXT:
        ---
        {context}
        ---

        FIELDS TO FILL:
        [{field_list_str}]

        Follow these critical rules:
        1.  **Differentiate between Factual and Subjective Fields:**
            * For **Factual Fields** (e.g., "First Name", "Email", "Date of Birth", "Phone Number", "UFID"): Extract the information *exactly* as it appears in the context. Do not change it.
            * For **Subjective/Open-Ended Fields** (e.g., "Why do you want this job?", "Personal Statement", "Describe your skills"): Use the user's answer from the context as the core idea, but rephrase and enhance it to sound more professional, articulate, and compelling. Expand on the user's points where appropriate to create a strong, positive impression.

        2.  **Output Format:** Your response MUST be a single, valid JSON object. The keys of the JSON object must be the exact field names from the list.

        3.  **Handling Missing Information:** If you cannot find the information for a specific field in the context, use the string "N/A" as the value for that field.

        4.  **Tone for Rephrasing:** For subjective answers, adopt a confident, competent, and professional tone.

        5.  **No Extra Text:** Do not add any explanations or introductory text. Only output the JSON.
    """)

    log.info(f"Sending {len(human_readable_fields)} human-readable fields to the LLM for extraction and enhancement.")
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_text)
    except Exception as e:
        log.error(f"Failed to get or parse extraction response from LLM: {e}")
        return {field: "N/A" for field in human_readable_fields}

def generate_detailed_question(missing_fields: list[str]) -> str:
    """Uses the LLM to generate a user-friendly question for a short list of missing info."""
    model = genai.GenerativeModel(LLM_MODEL)
    field_list = ", ".join(missing_fields)
    prompt = f"You are a friendly assistant. You couldn't find the following information: {field_list}. Formulate a single, polite question to ask the user for all of it."
    log.info(f"Asking LLM to generate a detailed clarifying question for: {missing_fields}")
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        log.error(f"Failed to generate detailed question from LLM: {e}")
        return f"I couldn't find the following information, can you please provide it? {field_list}"

# ----- MCP Server Tool Definition ----------------------------------------

mcp = FastMCP(
    "Smart-PDF-Form-Filler",
    instructions="Upload a blank PDF form and provide the context for filling it. I use AI vision to understand the form and will fill it out, or ask for any missing info."
)

@mcp.tool()
def fill_form(pdf_path: str, context: str) -> dict:
    """The main tool function that orchestrates the PDF filling process."""
    if not os.path.isfile(pdf_path):
        return {"status": "error", "message": f"File not found: {pdf_path}"}

    try:
        pdf_bytes = open(pdf_path, "rb").read()
        reader = PdfReader(io.BytesIO(pdf_bytes))
        fields = reader.get_fields() or {}
    except Exception as e:
        return {"status": "error", "message": f"Could not read PDF: {e}"}

    if not fields:
        return {"status": "error", "message": "No fillable form fields were detected in this PDF."}

    text_fields = {k: v for k, v in fields.items() if v.get("/FT") == "/Tx"}
    if not text_fields:
        return {"status": "error", "message": "No text fields detected."}

    vision_mapping = map_fields_with_vision(pdf_bytes, text_fields)
    
    field_mapping = {}
    for internal_name, field_obj in text_fields.items():
        human_name = vision_mapping.get(internal_name)
        if not human_name:
            human_name = field_obj.get('/T')
        if not human_name:
            human_name = internal_name
        
        if human_name not in field_mapping:
            field_mapping[human_name] = internal_name
    
    human_readable_names = list(field_mapping.keys())
    extracted_values_by_human_name = extract_answers_with_llm(human_readable_names, context)

    values_for_pdf = {}
    missing_human_names = []
    for human_name, internal_name in field_mapping.items():
        if not isinstance(human_name, str):
            continue
            
        value = extracted_values_by_human_name.get(human_name, "N/A").strip()
        if value and value != "N/A":
            values_for_pdf[internal_name] = value
        else:
            missing_human_names.append(human_name)

    if missing_human_names:
        unique_missing = sorted(list(set(missing_human_names)))
        log.warning(f"Information is missing for the following fields: {unique_missing}")

        # If too many fields are missing, create a simple message with examples
        # to avoid a slow, complex, and timeout-prone API call.
        if len(unique_missing) > 15:
            # Take the first 5 missing items as examples for the user
            examples = ", ".join(unique_missing[:5])
            question = f"I was able to fill out some of the form, but a lot of information is still missing. For example, I need details about: {examples}, and more. Could you please provide more context?"
            log.info("Too many fields missing. Generating a simple summary question to avoid timeout.")
        else:
            # If a manageable number are missing, ask the AI to formulate a friendly question.
            question = generate_detailed_question(unique_missing)
            
        return {"status": "need_info", "message": question}

    # --- Final PDF Generation and Saving ---
    log.info(f"Populating PDF with data: {values_for_pdf}")
    try:
        pdf_filled_bytes = fill_pdf_bytes(pdf_bytes, values_for_pdf)
        output_dir = "filled_pdfs"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"filled_{os.path.basename(pdf_path).replace('.pdf', '')}_{uuid.uuid4().hex[:6]}.pdf"
        output_path = os.path.join(output_dir, filename)

        with open(output_path, "wb") as f:
            f.write(pdf_filled_bytes)
        log.info(f"Successfully saved filled PDF to: {output_path}")

        encoded_pdf = base64.b64encode(pdf_filled_bytes).decode()
        return {"status": "done", "filename": filename, "filled_pdf": encoded_pdf, "local_path": output_path}
    except Exception as e:
        log.error(f"Critical error during PDF generation: {e}")
        return {"status": "error", "message": f"Failed to write or save the final PDF: {e}"}


if __name__ == "__main__":
    log.info("Starting Smart PDF Filler MCP server...")
    mcp.run(transport="stdio")
