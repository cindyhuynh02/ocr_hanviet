from utils.__init__ import *
from utils.model import *
img_folder = "G:/Github/ocr_hanviet/data/test"
pdf_folder = "G:/Github/ocr_hanviet/data/pdf"
output_path = "G:/Github/ocr_hanviet/data/ocr_output.csv"

def main():
    img_folder = "G:/Github/ocr_hanviet/data/test"
    pdf_folder = "G:/Github/ocr_hanviet/data/pdf"
    output_path = "G:/Github/ocr_hanviet/data/ocr_output.csv"
    
    # Step 1: Convert PDFs to images (uncomment if needed)
    # print("🔄 Converting PDFs to images...")
    # pdf_converter = PDFToImageConverter(pdf_folder, img_folder)
    # pdf_converter.convert_pdf_range(6, 10)  # Convert taps 6-10
    
    # Step 2: OCR processing
    print("🔄 Starting OCR processing...")
    extractor = GPTOCRExtractor(api_key=OPENAI_API_KEY, img_folder=img_folder)
    df_all = extractor.ocr_folder_parallel(max_workers=10, output_file=output_path)
    
    if not df_all.empty:
        print(f"🎉 OCR completed successfully! {len(df_all)} total lines extracted.")
    else:
        print("❌ OCR process failed or no data extracted.")


if __name__ == "__main__":
    main()