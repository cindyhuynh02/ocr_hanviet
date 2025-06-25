from utils.__init__ import *

class PDFToImageConverter:
    """Convert PDF pages to JPG images"""
    
    def __init__(self, pdf_folder: str, img_folder: str):
        self.pdf_folder = pdf_folder
        self.img_folder = img_folder
        
        # Create output folder if it doesn't exist
        os.makedirs(self.img_folder, exist_ok=True)
    
    def convert_pdf_range(self, start_tap: int, end_tap: int, dpi: int = 300) -> None:
        """
        Convert PDF files to JPG images for a range of tap numbers
        
        Args:
            start_tap: Starting tap number (inclusive)
            end_tap: Ending tap number (inclusive)
            dpi: Resolution for image conversion
        """
        print(f"🔄 Converting PDFs from tap {start_tap} to {end_tap}...")
        
        for tap_number in range(start_tap, end_tap + 1):
            pdf_path = os.path.join(self.pdf_folder, f"tap_{tap_number}.PDF")
            self._convert_single_pdf(pdf_path, tap_number, dpi)
    
    def _convert_single_pdf(self, pdf_path: str, tap_number: int, dpi: int) -> None:
        """Convert a single PDF file to JPG images"""
        try:
            doc = fitz.open(pdf_path)
            print(f"📖 Processing tap {tap_number}: {len(doc)} pages")
        except Exception as e:
            print(f"❌ Cannot open file {pdf_path}: {e}")
            return

        for page_number in range(len(doc)):
            try:
                page = doc.load_page(page_number)
                pix = page.get_pixmap(dpi=dpi)
                output_path = os.path.join(
                    self.img_folder, 
                    f"page_{tap_number}_{page_number + 1}.jpg"
                )
                pix.save(output_path)
                print(f"✅ Saved: {output_path}")
            except Exception as e:
                print(f"❌ Error processing page {page_number + 1} of tap {tap_number}: {e}")

        doc.close()

class GPTOCRExtractor:
    """OCR text extraction using GPT-4o Vision API"""
    
    def __init__(self, api_key: str, img_folder: str):
        self.api_key = api_key
        self.img_folder = img_folder
        self.last_section_id = None
        openai.api_key = self.api_key

    def _load_image(self, image_path: str) -> Optional[str]:
        """Load and encode image as base64"""
        if not os.path.exists(image_path):
            print(f"❌ File does not exist: {image_path}")
            return None
        
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"❌ Cannot read image: {e}")
            return None

    def _call_gpt_ocr(self, image_data_url: str) -> Optional[str]:
        """Call GPT-4o API for OCR"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional OCR assistant. Extract each LINE of Vietnamese printed text from the image. "
                            "For each line, return a dictionary with: "
                            "'text': content of the line, and 'bbox': bounding box coordinates as [x0, y0, x1, y1]. "
                            "Return only a list of dictionaries, no additional explanation."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_url}
                            }
                        ]
                    }
                ],
                temperature=0,
                max_tokens=2000,
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"❌ Error calling GPT-4o API: {e}")
            return None

    def _parse_response(self, content: str) -> List[Dict]:
        """Parse GPT response to extract OCR lines"""
        if not content:
            return []

        content_clean = content.strip()
        if content_clean.startswith("```"):
            content_clean = re.sub(r"^```(?:json)?\s*", "", content_clean)
            content_clean = re.sub(r"\s*```$", "", content_clean)

        try:
            ocr_lines = json.loads(content_clean)
            if isinstance(ocr_lines, list):
                return ocr_lines
        except json.JSONDecodeError:
            try:
                print("⚠️ JSON error, trying ast parsing...")
                ocr_lines = ast.literal_eval(content_clean)
                if isinstance(ocr_lines, list):
                    return ocr_lines
            except Exception as e_ast:
                print("❌ Parse failed:", e_ast)
                print("🔍 Content preview:", repr(content_clean[:300]))
        
        return []

    def _build_dataframe(self, ocr_lines: List[Dict], image_name: str, 
                        paper_name: str, file_id: str) -> pd.DataFrame:
        """Build DataFrame from OCR lines"""
        data = []
        current_section = None
        sentence_buffer = []
        bbox_buffer = []

        def is_sentence_ending(text: str) -> bool:
            """Check if text represents end of sentence"""
            text = text.strip()
            return (
                text.endswith((".", ":", "!", "?"))
                or text.isupper()
                or text.startswith("-")
                or len(text.split()) <= 3
            )

        for idx, line in enumerate(ocr_lines):
            if not isinstance(line, dict) or "text" not in line:
                continue

            text = line["text"].strip()
            if not text:
                continue
            
            bbox = line.get("bbox", [0, 0, 0, 0])
            if not isinstance(bbox, list) or len(bbox) != 4:
                bbox = [0, 0, 0, 0]

            # Check for section markers
            match = re.search(r"(quyển\s+\w+)", text, re.IGNORECASE)
            if match:
                current_section = match.group(1)
                self.last_section_id = current_section

            sentence_buffer.append(text)
            bbox_buffer.append(bbox)

            # Determine sentence boundaries
            next_line = ocr_lines[idx + 1]["text"].strip() if idx + 1 < len(ocr_lines) else ""
            ends_now = is_sentence_ending(text)
            starts_new = next_line.startswith("-") or next_line.isupper()

            if ends_now or starts_new or idx == len(ocr_lines) - 1:
                full_text = " ".join(sentence_buffer).strip()
                sentence_buffer.clear()

                sentences = [s.strip() for s in full_text.split('.') if s.strip()]
                if not sentences:
                    sentences = [full_text]

                for i, sent in enumerate(sentences):
                    full_sentence = sent if sent.endswith(('.', '!', '?', ':')) else sent + '.'
                    data.append({
                        "File_id": file_id,
                        "Paper_id": paper_name,
                        "Section_id": current_section or self.last_section_id,
                        "Page_id": image_name,
                        "BBox": bbox_buffer[0] if bbox_buffer else [0, 0, 0, 0],
                        "Text": full_text if i == 0 else "",
                        "Sentence": full_sentence
                    })
                bbox_buffer.clear()
        
        return pd.DataFrame(data)

    def ocr_image(self, image_path: str, image_name: str, 
                  paper_name: str, file_id: str = "HVE_009") -> pd.DataFrame:
        """OCR a single image"""
        print(f"📤 OCR processing: {image_name}")
        
        image_b64 = self._load_image(image_path)
        if not image_b64:
            return pd.DataFrame()

        content = self._call_gpt_ocr(f"data:image/jpeg;base64,{image_b64}")
        if not content:
            print("❌ No response from GPT")
            return pd.DataFrame()

        ocr_lines = self._parse_response(content)
        if not ocr_lines:
            print("❌ Empty OCR lines")
            return pd.DataFrame()

        return self._build_dataframe(ocr_lines, image_name, paper_name, file_id)

    def ocr_folder_parallel(self, max_workers: int = 4, 
                           output_file: Optional[str] = None) -> pd.DataFrame:
        """OCR all images in folder using parallel processing"""
        if not os.path.exists(self.img_folder):
            print(f"❌ Folder does not exist: {self.img_folder}")
            return pd.DataFrame()

        files = sorted(
            f for f in os.listdir(self.img_folder) 
            if f.lower().startswith("page_") and f.lower().endswith(".jpg")
        )
        
        if not files:
            print("❌ No valid images found")
            return pd.DataFrame()

        print(f"🖼️ Found {len(files)} images, starting OCR with {max_workers} workers...")

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(
                    self.ocr_image, 
                    os.path.join(self.img_folder, f), 
                    f.split("_")[-1].replace(".jpg", ""), 
                    f.split("_")[1]
                ): f for f in files
            }

            for i, future in enumerate(as_completed(future_to_file), 1):
                file = future_to_file[future]
                try:
                    df = future.result()
                    if not df.empty:
                        results.append(df)

                        # Write to output file immediately
                        if output_file:
                            write_header = not os.path.exists(output_file)
                            df.to_csv(output_file, mode='a', index=False, 
                                    encoding='utf-8-sig', header=write_header)

                        print(f"✅ ({i}/{len(files)}) {file}: {len(df)} lines")
                    else:
                        print(f"⚠️ ({i}/{len(files)}) {file}: No lines")
                except Exception as e:
                    print(f"❌ ({i}/{len(files)}) {file}: {e}")

        if results:
            df_all = pd.concat(results, ignore_index=True)
            print(f"📊 Total {len(df_all)} lines from {len(results)} images")
            
            if output_file:
                try:
                    df_all.to_csv(output_file, index=False, encoding='utf-8-sig')
                    print(f"💾 Results saved to: {output_file}")
                except Exception as e:
                    print(f"⚠️ Cannot save file: {e}")
            
            return df_all
        else:
            print("❌ No data extracted")
            return pd.DataFrame()