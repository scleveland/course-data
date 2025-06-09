import os
import json
import time
import random
import concurrent.futures
from typing import List, Dict, Optional, Tuple, Type, Any, Callable
import PyPDF2
from vertexai.preview.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold
import vertexai
from google.api_core import exceptions as google_exceptions
from google.oauth2 import service_account
from concurrent.futures import ThreadPoolExecutor, as_completed

class GeminiCourseExtractor:
    def __init__(self, project_id: str, keyfile_path: str, location: str = "us-central1"):
        """Initialize the Gemini extractor with Vertex AI using a service account key file.
        
        Args:
            project_id: Your Google Cloud project ID
            keyfile_path: Path to the service account key JSON file
            location: Google Cloud region (default: us-central1)
        """
        self.project_id = project_id
        self.keyfile_path = keyfile_path
        self.location = location
        self.model = None
        self.credentials = None
        self.courses: List[Dict] = []
        self.current_institution: Optional[str] = None
        self.initialize_vertex_ai()
    
    def initialize_vertex_ai(self) -> None:
        """Initialize the Vertex AI with the Gemini model using service account credentials."""
        try:
            # Load the service account credentials
            self.credentials = service_account.Credentials.from_service_account_file(
                self.keyfile_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            
            # Initialize Vertex AI with the credentials
            vertexai.init(
                project=self.project_id,
                location=self.location,
                credentials=self.credentials
            )
            
            # Initialize the Gemini 2.5 Flash preview model
            self.model = GenerativeModel("gemini-2.5-flash-preview-05-20")
            print("Successfully initialized Vertex AI with service account credentials")
            
        except Exception as e:
            print(f"Error initializing Vertex AI: {e}")
            raise
    
    def retry_with_backoff(
        self,
        func: Callable,
        *args,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        exceptions: tuple[Type[Exception], ...] = (
            google_exceptions.ResourceExhausted,
            google_exceptions.ServiceUnavailable,
            google_exceptions.DeadlineExceeded,
            Exception  # Catch-all for other transient errors
        )
    ) -> Any:
        """Retry a function with exponential backoff."""
        delay = initial_delay
        for attempt in range(1, max_retries + 1):
            try:
                return func(*args)
            except exceptions as e:
                if attempt == max_retries:
                    print(f"Max retries ({max_retries}) reached. Last error: {e}")
                    raise
                
                # Add jitter to avoid thundering herd
                sleep_time = min(delay * (1 + random.random() * 0.1), max_delay)
                print(f"Attempt {attempt} failed with error: {e}. Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
                delay *= backoff_factor
                
        raise Exception("Max retries reached")

    def process_chunk(self, chunk_data: Tuple[int, str, str]) -> Tuple[int, List[Dict]]:
        """Process a single chunk of text and return its courses with retry logic."""
        chunk_num, chunk, filename = chunk_data
        print(f"Processing chunk {chunk_num}...")
        
        def _process() -> List[Dict]:
            return self.extract_courses_with_gemini(chunk, filename) or []
            
        try:
            chunk_courses = self.retry_with_backoff(
                _process,
                max_retries=3,
                initial_delay=2.0,
                backoff_factor=2.0
            )
            
            if chunk_courses:
                print(f"Extracted {len(chunk_courses)} courses from chunk {chunk_num}")
                return (chunk_num, chunk_courses)
            return (chunk_num, [])
            
        except Exception as e:
            print(f"Error in chunk {chunk_num} after retries: {e}")
            return (chunk_num, [])

    def find_course_section_start(self, reader: PyPDF2.PdfReader, max_toc_pages: int = 20) -> int:
        """Analyze the PDF to find the starting page of the course section using Gemini.
        
        Args:
            reader: PyPDF2 PdfReader instance
            max_toc_pages: Maximum number of pages to check for table of contents
            
        Returns:
            int: 1-based page number where course section starts, or 1 if not found
        """
        print("  - Searching for course section using Gemini...")
        
        # First, find a proper table of contents page
        toc_page_num = None
        max_pages_to_scan = min(30, len(reader.pages))  # Scan at most 30 pages
        
        # Phase 1: Find the table of contents using Gemini
        for page_num in range(max_pages_to_scan):
            try:
                page = reader.pages[page_num]
                text = page.extract_text()
                print(f"  - Analyzing page {page_num + 1} with Gemini... {text[:100]}")
                # Skip empty or very short pages
                if not text or len(text.strip()) < 20:
                    continue
                
                # Use Gemini to determine if this is a TOC page
                prompt = f"""Analyze if this page is a table of contents or contents page from a university course catalog.
                A table of contents or contents page typically contains a list of sections and page numbers. Sections may be names General Information, Courses and others that makes sense in the context of an academic institutions catalog.
                
                Page content:
                {text}...
                
                Is this a table of contents page? Answer with exactly 'yes' or 'no'."""
                
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": 65000,
                        "temperature": 0.0
                    }
                    # },
                    # safety_settings={
                    #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    # }
                )
                
                if response and hasattr(response, 'text'):
                    if response.text.strip().lower().startswith('yes'):
                        toc_page_num = page_num + 1  # Store 1-based page number
                        print(f"  - Gemini identified page {toc_page_num} as a table of contents")
                        break
                    
            except Exception as e:
                print(f"  - Error reading page {page_num + 1}: {e}")
        
        # Phase 2: Look for course listings starting from the TOC page
        if toc_page_num is not None:
            # Search up to 10 pages after the TOC
            end_page = min(toc_page_num + 10, len(reader.pages))
            for page_num in range(toc_page_num - 1, end_page):  # Convert back to 0-based
                try:
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    
                    print(f"  - Analyzing TOC on page {page_num} with Gemini...{text}")
                    # Prepare the prompt with clear instructions
                    prompt_text = f"""You are analyzing a university course catalog's table of contents.
                    Your task is to find the page number where the actual course listings begin.
                    Look for sections like 'Course Descriptions', 'Courses', 'Course Catalog', etc.  Then if there is no page number look if there are subsections like overview or description etc and use the first subsection's page number.
                    
                    Return ONLY the page number as an integer, or 'null' if you can't find any course listing information in this TOC.
                    
                    Table of Contents:
                    {text}
                    
                    Page number where course listings begin (or 'null' if not found):"""
                    
                    # Send to Gemini with safety settings
                    response = self.model.generate_content(
                        prompt_text,
                        generation_config={
                            "max_output_tokens": 65000,  # We only need a number
                            "temperature": 0.0,  # Be deterministic
                        },
                        safety_settings={
                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        }
                    )
                
                    # Safely extract the response text
                    if response and hasattr(response, 'text'):
                        result = response.text.strip()
                        # Clean the result to get just the first number
                        import re
                        match = re.search(r'\d+', result)
                        if match:
                            page_num = int(match.group())
                            if 1 <= page_num <= len(reader.pages):
                                print(f"  - Gemini suggests starting at page {page_num}")
                                return page_num
                            else:
                                print(f"  - Gemini returned out-of-range page number: {page_num}")
                        else:
                            print(f"  - No page number found in Gemini response")
                    else:
                        print("  - Empty or invalid response from Gemini")
                    
                except Exception as e:
                    print(f"  - Error processing Gemini response: {str(e)}")
            
            # If we get here, we didn't find course listings but have a TOC page
            return toc_page_num
                        
        

        
    def _extract_page_number(self, line: str) -> int:
        """Extract page number from a TOC line."""
        # Handle different TOC formats:
        # 1. "Courses ................ 123"
        # 2. "Courses 123"
        # 3. "123 Courses"
        # 4. "Courses" (on one line) followed by "123" (on next line)
        
        # Remove any non-alphanumeric characters except spaces and dots
        clean_line = ''.join(c if c.isalnum() or c in ' .' else ' ' for c in line)
        
        # Look for numbers in the line
        parts = clean_line.split()
        for part in reversed(parts):  # Check from right to left
            if part.isdigit():
                page_num = int(part)
                if 1 <= page_num <= 1000:  # Reasonable page number range
                    return page_num
        
        return None

    def process_pdf(self, filepath: str, max_workers: int = 4) -> List[Dict]:
        """Process a single PDF file and return extracted courses using parallel processing."""
        filename = os.path.basename(filepath)
        print(f"Processing: {filename}")
        
        # Read the entire PDF with retry and proper resource management
        def _read_pdf() -> str:
            try:
                print(f"  - Opening PDF file: {filepath}")
                with open(filepath, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    total_pages = len(reader.pages)
                    
                    # Find the starting page for course section
                    start_page = self.find_course_section_start(reader)
                    print(f"  - Processing pages {start_page} to {total_pages} of {total_pages}...")
                    
                    # Use a list to collect page texts and join at the end
                    page_texts = []
                    for i in range(start_page - 1, total_pages):  # Convert to 0-based index
                        page = reader.pages[i]
                        if (i - start_page + 1) % 10 == 0 or i == start_page - 1 or i == total_pages - 1:
                            print(f"  - Extracting text from page {i + 1}/{total_pages}")
                        page_text = page.extract_text()
                        if page_text:  # Only add non-empty texts
                            page_texts.append(page_text)
                    
                    total_text = '\n'.join(page_texts)
                    print(f"  - Extracted {len(total_text):,} characters from {total_pages - start_page + 1} pages")
                    return total_text
                    
            except Exception as e:
                print(f"Error reading PDF: {e}")
                return ""
            finally:
                # Ensure any resources are released
                if 'reader' in locals():
                    if hasattr(reader, '_stream') and reader._stream:
                        reader._stream.close()
                    print("  - Closed PDF file")
        
        text = self.retry_with_backoff(_read_pdf, max_retries=3, initial_delay=1.0)
            
        if not text:
            print(f"Warning: Could not extract text from {filepath}")
            return []
            
        # Process text in chunks to avoid memory issues
        chunk_size = 10000  # Target chunk size
        min_chunk_size = 1000  # Minimum chunk size to ensure progress
        overlap = 1000  # 1K character overlap to prevent splitting courses
        chunks = []
        start = 0
        chunk_num = 1
        total_chars = len(text)
        
        print(f"  - Splitting {total_chars:,} characters into chunks...")
        
        while start < total_chars:
            # Calculate end position for this chunk
            end = min(start + chunk_size, total_chars)
            
            # If this would be the last chunk and it's too small, just extend it to the end
            if end == total_chars and (end - start) < min_chunk_size and len(chunks) > 0:
                # Merge with previous chunk if possible
                prev_chunk_num, prev_chunk, _ = chunks[-1]
                chunks[-1] = (prev_chunk_num, prev_chunk + text[start:end], filename)
                print(f"  - Merged final small chunk ({end-start:,} chars) with previous chunk")
                break
            
            # Add the chunk if it's not empty
            if start < end:
                chunk_text = text[start:end]
                chunks.append((chunk_num, chunk_text, filename))
                
                # Calculate and log progress
                progress = min(100, int((end / total_chars) * 100))
                print(f"  - Created chunk {chunk_num}: positions {start:,}-{end:,} "
                      f"({len(chunk_text):,} chars, {progress}% of text)")
            
            # Calculate next start position (move forward by chunk_size - overlap)
            new_start = start + chunk_size - overlap
            
            # If we're at or beyond the end, we're done
            if new_start >= total_chars:
                break
                
            # Ensure we're making progress
            if new_start <= start:
                new_start = start + 1
                if new_start >= total_chars:
                    break
            
            # Update for next iteration
            start = new_start
            chunk_num += 1
            
            # Force garbage collection every 10 chunks to help with memory
            if chunk_num % 10 == 0:
                import gc
                gc.collect()
                
            # Safety check to prevent infinite loops
            if chunk_num > 1000:  # Arbitrary large number to prevent infinite loops
                print(f"  - Warning: Exceeded maximum number of chunks (1000). Stopping chunking.")
                break
                
        print(f"  - Split into {len(chunks)} chunks for processing")
        
        all_courses = []
        completed_chunks = set()
        
        # Process chunks in parallel with rate limiting
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {}
            for chunk in chunks:
                future = executor.submit(self.process_chunk, chunk)
                future_to_chunk[future] = chunk[0]
                
                # Small delay between submissions to avoid overwhelming the API
                time.sleep(0.5)
            
            # Process results as they complete
            start_time = time.time()
            last_update = start_time
            
            print(f"\n  --- Starting parallel processing of {len(chunks)} chunks with {max_workers} workers ---")
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_num = future_to_chunk[future]
                current_time = time.time()
                
                try:
                    chunk_num, chunk_courses = future.result()
                    if chunk_courses:
                        all_courses.extend(chunk_courses)
                    completed_chunks.add(chunk_num)
                    
                    # Calculate progress and ETA
                    elapsed = current_time - start_time
                    chunks_remaining = len(chunks) - len(completed_chunks)
                    chunks_per_sec = len(completed_chunks) / (elapsed + 1e-6)
                    eta = chunks_remaining / (chunks_per_sec + 1e-6)
                    
                    # Only log every 5 chunks or if it's been more than 5 seconds since last update
                    if (len(completed_chunks) % 5 == 0 or 
                        current_time - last_update > 5 or 
                        len(completed_chunks) == len(chunks)):
                        print(
                            f"  - Processed chunk {len(completed_chunks)}/{len(chunks)} | "
                            f"{len(chunk_courses)} courses | "
                            f"ETA: {int(eta//60)}m {int(eta%60)}s"
                        )
                        last_update = current_time
                        
                except Exception as e:
                    print(f"Error processing chunk {chunk_num} after submission: {e}")
        
        total_time = time.time() - start_time
        print(f"\n  --- Completed processing {filename} in {total_time:.1f} seconds ---")
        print(f"  - Extracted {len(all_courses)} courses from {len(chunks)} chunks")
        print(f"  - Average processing time: {total_time/len(chunks):.2f} seconds per chunk")
        return all_courses
    
    def read_entire_pdf(self, filepath: str) -> str:
        """Read the entire PDF file and return its text content with proper resource management."""
        try:
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                # Use a list to collect page texts and join at the end
                page_texts = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:  # Only add non-empty texts
                        page_texts.append(page_text)
                return '\n'.join(page_texts)
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
        finally:
            # Ensure any resources are released
            if 'reader' in locals():
                if hasattr(reader, '_stream') and reader._stream:
                    reader._stream.close()

    def extract_courses_with_gemini(self, text: str, filename: str) -> List[Dict]:
        """Use Gemini to extract course information from text with degree and certificate associations."""
        prompt = """Extract all courses from the following university catalog text. 
        For each course, return a JSON array of objects with these fields:
        - course_id (e.g., 'ENG 100')
        - subject (the subject area code, e.g., 'ENG' for English, 'MATH' for Mathematics)
        - title (course title)
        - description (course description)
        - credits (number of credits, if mentioned)
        - prerequisites (if any)
        - program (the major, degree, or certificate program this course belongs to, if mentioned)
        - degrees (an array of degree programs that require this course, e.g., ["Computer Science BS", "Data Science BA"])
        - certificates (an array of certificates that include this course, e.g., ["Cybersecurity Certificate", "Web Development Certificate"])
        - department (the academic department offering the course, e.g., 'Computer Science', 'English')
        - college (the college or school within the university that offers this course, e.g., 'College of Arts and Sciences')
        
        For the program field, identify the most relevant academic program.
        For the degrees field, include all degree programs that require this course.
        For the certificates field, include all certificate programs that include this course.
        For the department field, identify the academic department that offers the course.
        For the college field, identify the college or school within the university.
        
        Return ONLY valid JSON, nothing else. Example:
        [
            {
                "course_id": "CS 101",
                "title": "Introduction to Computer Science",
                "description": "Fundamentals of computer programming and problem solving...",
                "credits": 4,
                "prerequisites": "MATH 100 or placement in MATH 115",
                "program": "Computer Science",
                "degrees": ["Computer Science BS", "Computer Engineering BS", "Data Science BA"],
                "certificates": ["Cybersecurity Certificate", "Software Development Certificate"],
                "department": "Computer Science Department",
                "college": "College of Natural Sciences"
            },
            {
                "course_id": "MATH 201",
                "title": "Calculus I",
                "description": "Limits, derivatives, and applications...",
                "credits": 4,
                "prerequisites": "MATH 115 or equivalent",
                "program": "Mathematics",
                "degrees": ["Mathematics BS", "Physics BS", "Engineering BS"],
                "certificates": ["Applied Mathematics Certificate"],
                "department": "Mathematics Department",
                "college": "College of Natural Sciences"
            }
        ]
        
        Here's the text to process:
        """
        
        # Set the institution based on filename
        self.set_institution_from_filename(filename)
        
        try:
            # Limit text size to avoid context window issues
            content = text[:8000]
            
            # Create the full prompt
            full_prompt = prompt + content
            
            # Generate content with error handling
            try:
                response = self.model.generate_content(full_prompt)
            except Exception as e:
                print(f"Error generating content: {e}")
                return []
            
            # Debug: Print the raw response structure
            print("Response structure:", dir(response))
            
            # Extract text from response
            response_text = None
            try:
                # Try different ways to get the response text
                if hasattr(response, 'text'):
                    response_text = response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        if candidate.content.parts:
                            response_text = candidate.content.parts[0].text
                
                if not response_text:
                    print("Could not extract text from response")
                    print("Full response:", response)
                    return []
                
                print("Raw response (first 500 chars):", response_text[:500])
                
                # Clean the response to extract just the JSON
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0]
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].split('```')[0]
                
                # Parse the JSON
                courses = json.loads(response_text)
                
                # Ensure we have a list
                if not isinstance(courses, list):
                    courses = [courses]
                
                # Process each course to ensure required fields are present
                processed_courses = []
                for course in courses:
                    # Ensure required fields exist with defaults
                    if not isinstance(course, dict):
                        continue
                        
                    # Initialize missing fields with defaults
                    course.setdefault('institution', self.current_institution)
                    course.setdefault('degrees', [])
                    course.setdefault('certificates', [])
                    
                    # Ensure degrees and certificates are lists
                    if 'degrees' not in course or not isinstance(course['degrees'], list):
                        course['degrees'] = []
                    if 'certificates' not in course or not isinstance(course['certificates'], list):
                        course['certificates'] = []
                    
                    # Clean up any string values that should be lists
                    if isinstance(course.get('degrees'), str):
                        course['degrees'] = [d.strip() for d in course['degrees'].split(',') if d.strip()]
                    if isinstance(course.get('certificates'), str):
                        course['certificates'] = [c.strip() for c in course['certificates'].split(',') if c.strip()]
                    
                    processed_courses.append(course)
                
                print(f"Successfully extracted {len(processed_courses)} courses")
                return processed_courses
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Problematic response: {response_text}")
                return []
                
            except Exception as e:
                print(f"Error processing response: {e}")
                print(f"Response type: {type(response)}")
                print(f"Response dir: {dir(response)}")
                if hasattr(response, 'prompt_feedback'):
                    print(f"Prompt feedback: {response.prompt_feedback}")
                return []
                
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def set_institution_from_filename(self, filename: str) -> None:
        """
        Extract and set the institution name from the filename.
        
        Args:
            filename: The name of the PDF file (e.g., 'University-of-Hawaii-Hilo.pdf')
        """
        # Remove file extension and any path
        base_name = os.path.splitext(os.path.basename(filename))[0]
        
        # Common patterns to clean up
        patterns_to_remove = [
            'catalog', 'courses', 'university', 'of', 'at', 'and',
            'course', 'listings', 'listing', 'catalogue', 'programs',
            'academic', 'bulletin', 'handbook', 'guide', 'pdf',
            'catalog-', 'catalog_', '-catalog', '_catalog',
            '202', '201', '20', '21', '22', '23', '24', '25'  # Common year numbers
        ]
        
        # Split into words and clean up
        words = []
        for word in base_name.replace('-', ' ').replace('_', ' ').split():
            word_lower = word.lower()
            if (word_lower not in patterns_to_remove and 
                not word_lower.isdigit() and 
                len(word_lower) > 1):
                words.append(word)
        
        # Join words and clean up any remaining artifacts
        institution = ' '.join(words).strip()
        
        # Special handling for known institutions
        if not institution:
            institution = 'Unknown Institution'
        else:
            # Clean up common issues
            institution = ' '.join(institution.split())  # Remove extra spaces
            
            # Handle specific cases
            if 'hawaii' in institution.lower() and 'hilo' in institution.lower():
                institution = 'University of Hawaii at Hilo'
            elif 'hawaii' in institution.lower() and ('west' in institution.lower() or 'oahu' in institution.lower()):
                institution = 'University of Hawaii West Oʻahu'
            elif 'hawaii' in institution.lower() and 'manoa' in institution.lower():
                institution = 'University of Hawaii at Manoa'
            elif 'hawaii' in institution.lower() and 'community' in institution.lower():
                institution = 'Hawaii Community College'
            elif 'hawaii' in institution.lower() and 'honolulu' in institution.lower():
                institution = 'Honolulu Community College'
            elif 'hawaii' in institution.lower() and 'kapiolani' in institution.lower():
                institution = 'Kapiʻolani Community College'
            elif 'hawaii' in institution.lower() and 'kauai' in institution.lower():
                institution = 'Kauai Community College'
            elif 'hawaii' in institution.lower() and 'leeward' in institution.lower():
                institution = 'Leeward Community College'
            elif 'hawaii' in institution.lower() and 'maui' in institution.lower():
                institution = 'University of Hawaii Maui College'
            elif 'hawaii' in institution.lower() and 'windward' in institution.lower():
                institution = 'Windward Community College'
            
            # Format the institution name properly
            if 'university' in institution.lower() and 'of' not in institution.lower():
                # Convert "University Hawaii" to "University of Hawaii"
                institution = institution.replace('University', 'University of', 1)
            
            # Handle common abbreviations
            abbreviation_map = {
                'UHH': 'University of Hawaii at Hilo',
                'UH Hilo': 'University of Hawaii at Hilo',
                'UHWO': 'University of Hawaii West Oʻahu',
                'UH West Oahu': 'University of Hawaii West Oʻahu',
                'UH Manoa': 'University of Hawaii at Manoa',
                'UHM': 'University of Hawaii at Manoa',
                'HCC': 'Hawaii Community College',
                'HonCC': 'Honolulu Community College',
                'KCC': 'Kapiʻolani Community College',
                'Kauai CC': 'Kauai Community College',
                'LCC': 'Leeward Community College',
                'UHMC': 'University of Hawaii Maui College',
                'WCC': 'Windward Community College'
            }
            
            # Check for abbreviations
            for abbr, full_name in abbreviation_map.items():
                if abbr.lower() in institution.lower():
                    institution = full_name
                    break
        
        self.current_institution = institution
        print(f"  - Extracted institution name: {institution}")

def process_pdfs_with_gemini(project_id: str, keyfile_path: str, catalogs_dir: str, output_dir: str, max_workers: int = 20):
    """Process all PDFs in the catalogs directory using Gemini."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the extractor with service account credentials
    extractor = GeminiCourseExtractor(project_id, keyfile_path)
    
    all_courses = []
    processed_files = 0
    
    # Process each PDF in the catalogs directory
    for filename in sorted(os.listdir(catalogs_dir)):
        if not filename.lower().endswith('.pdf'):
            continue
            
        filepath = os.path.join(catalogs_dir, filename)
        print(f"\nProcessing {filename}...")
        
        try:
            # Process the PDF and extract courses
            courses = extractor.process_pdf(filepath, max_workers=max_workers)
            if courses:
                # Create a clean base name for the output file
                base_name = os.path.splitext(filename)[0]
                output_filename = f"{base_name}_courses.json"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save courses for this file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(courses, f, indent=2, ensure_ascii=False)
                
                all_courses.extend(courses)
                processed_files += 1
                print(f"Extracted {len(courses)} courses from {filename}")
                print(f"Saved to: {output_path}")
            else:
                print(f"No courses extracted from {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save combined results
    if all_courses:
        combined_output = os.path.join(output_dir, 'all_courses_combined.json')
        with open(combined_output, 'w', encoding='utf-8') as f:
            json.dump(all_courses, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessing complete!")
    print(f"Processed {processed_files} files")
    print(f"Extracted a total of {len(all_courses)} courses")
    if all_courses:
        print(f"Individual course files saved in: {output_dir}")
        print(f"Combined results saved to: {combined_output}")
    
    return all_courses

def process_single_file(filepath: str, project_id: str, keyfile_path: str, output_dir: str, max_workers: int = 20):
    """Process a single PDF file and save the extracted courses."""
    if not os.path.isfile(filepath):
        print(f"Error: File not found: {filepath}")
        return []
        
    print(f"\nProcessing single file: {os.path.basename(filepath)}")
    
    # Initialize the extractor with service account credentials
    extractor = GeminiCourseExtractor(project_id, keyfile_path)
    
    try:
        # Process the PDF and extract courses
        courses = extractor.process_pdf(filepath, max_workers=max_workers)
        if courses:
            # Create a clean base name for the output file
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            output_filename = f"{base_name}_courses.json"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save courses for this file
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(courses, f, indent=2, ensure_ascii=False)
            
            print(f"Extracted {len(courses)} courses from {os.path.basename(filepath)}")
            print(f"Saved to: {output_path}")
            return courses
        else:
            print(f"No courses extracted from {os.path.basename(filepath)}")
            return []
            
    except Exception as e:
        print(f"Error processing {os.path.basename(filepath)}: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract course information from PDF catalogs using Gemini.')
    parser.add_argument('--catalog', type=str, help='Path to a directory containing catalog PDFs to process')
    parser.add_argument('--file', type=str, help='Path to a specific catalog PDF file to process')
    args = parser.parse_args()
    
    # Configuration
    GOOGLE_CLOUD_PROJECT = "librechat-its-rci"  # Google Cloud project ID
    SERVICE_ACCOUNT_KEYFILE = "keyfile.json"  # Path to your service account key file
    
    # Define directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    
    # Ensure keyfile exists
    keyfile_path = os.path.join(BASE_DIR, SERVICE_ACCOUNT_KEYFILE)
    if not os.path.exists(keyfile_path):
        print(f"Error: Service account key file not found at {keyfile_path}")
        print("Please place your service account key file in the project directory and update the filename if needed.")
        exit(1)
    
    # Process either a single file or a directory of files
    try:
        if args.file:
            # Process just the specified file
            filepath = os.path.abspath(args.file)
            if not filepath.lower().endswith('.pdf'):
                print("Error: The specified file is not a PDF")
                exit(1)
                
            process_single_file(
                filepath=filepath,
                project_id=GOOGLE_CLOUD_PROJECT,
                keyfile_path=keyfile_path,
                output_dir=OUTPUT_DIR,
                max_workers=100
            )
        else:
            # Process a directory of files (or default to catalogs directory)
            if args.catalog:
                if os.path.isdir(args.catalog):
                    CATALOGS_DIR = os.path.abspath(args.catalog)
                    print(f"Processing all PDFs in directory: {CATALOGS_DIR}")
                else:
                    print(f"Error: The provided path is not a directory: {args.catalog}")
                    exit(1)
            else:
                # Default to the catalogs directory if no path is provided
                CATALOGS_DIR = os.path.join(BASE_DIR, "catalogs")
                print(f"No catalog path provided. Defaulting to: {CATALOGS_DIR}")
            
            process_pdfs_with_gemini(
                project_id=GOOGLE_CLOUD_PROJECT,
                keyfile_path=keyfile_path,
                catalogs_dir=CATALOGS_DIR,
                output_dir=OUTPUT_DIR,
                max_workers=100
            )
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
    
    # Ensure keyfile exists
    keyfile_path = os.path.join(BASE_DIR, SERVICE_ACCOUNT_KEYFILE)
    if not os.path.exists(keyfile_path):
        print(f"Error: Service account key file not found at {keyfile_path}")
        print("Please place your service account key file in the project directory and update the filename if needed.")
        exit(1)
    
    # Process the PDFs
    try:
        process_pdfs_with_gemini(
            project_id=GOOGLE_CLOUD_PROJECT,
            keyfile_path=keyfile_path,
            catalogs_dir=CATALOGS_DIR,
            output_dir=OUTPUT_DIR,
            max_workers=100
        )
    except Exception as e:
        print(f"An error occurred: {e}")
