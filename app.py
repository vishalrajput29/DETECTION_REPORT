import streamlit as st
from utils import detect_tumor, generate_report, generate_pdf_bytes

st.title("Brain Tumor Detection & Report Generator")

# Input fields
name = st.text_input("Patient Name")
age = st.number_input("Age", min_value=1)
uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

if st.button("Analyze"):
    if uploaded_file and name and age:
        # Detect tumor in MRI image
        result_img, detections = detect_tumor(uploaded_file)
        
        # Generate report based on the detection
        report = generate_report(detections, name, age)
        
        # Display results
        st.image(result_img, caption="Detected Tumor Regions", use_column_width=True)
        st.subheader("Generated Medical Report")
        st.write(report)

        # Generate PDF and offer download
        pdf_bytes = generate_pdf_bytes(name, age, report, result_img)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{name}_brain_mri_report.pdf",
            mime="application/pdf"
        )
    else:
        st.error("Please upload an image and fill all patient details.")
