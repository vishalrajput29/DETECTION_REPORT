import os
import uuid
import io
import tempfile
from io import BytesIO
from fpdf import FPDF
from fpdf import FPDF
from PIL import Image
from ultralytics import YOLO
from dotenv import load_dotenv
from fpdf import FPDF
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

# Load YOLO model
model = YOLO("best.pt")

# Load Groq Chat Model
llm = ChatGroq(
    temperature=0.7,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

def detect_tumor(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    results = model(image)
    result_img = Image.fromarray(results[0].plot())
    detections = results[0].boxes
    return result_img, detections

def generate_report(detections, name, age):
    tumor_count = len(detections)
    prompt_template = PromptTemplate(
        input_variables=["name", "age", "tumor_count"],
        template="""
        Generate a detailed medical report for a patient named {name}, age {age}, 
        based on brain MRI analysis showing {tumor_count} detected tumor regions.
        The report should be clinical, and include interpretation, potential diagnosis,
        and recommendations for next steps.
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)
    report = chain.run(name=name, age=age, tumor_count=tumor_count)
    return report


def generate_pdf_bytes(name, age, report, result_img):
    # Create a PDF instance
    pdf = FPDF()
    pdf.add_page()

    # Add title to the PDF
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Medical Report", ln=True, align="C")
    pdf.ln(10)

    # Add patient details
    pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
    pdf.ln(10)

    # Add the report
    pdf.multi_cell(200, 10, txt=f"Report: {report}")
    pdf.ln(10)

    # Create a temporary file to save the image
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        result_img.save(temp_file, format='PNG')
        temp_file_path = temp_file.name

    # Now pass the file path to pdf.image()
    pdf.image(temp_file_path, x=10, y=50, w=100)

    # Output the PDF as bytes
    pdf_output = pdf.output(dest='S').encode('latin1')

    return pdf_output