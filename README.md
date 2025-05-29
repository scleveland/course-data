# Gemini Course Extractor

A Python tool that extracts course information from university catalog PDFs using Google's Gemini AI model.

## Prerequisites

- Python 3.8 or higher
- Google Cloud account with Vertex AI API enabled
- Service account key file with Vertex AI permissions

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd course-data
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   # OR
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install the required packages**
   ```bash
   pip install -r requirements.txt
   ```
   
   If you don't have a `requirements.txt`, install the dependencies manually:
   ```bash
   pip install PyPDF2 google-cloud-aiplatform google-auth
   ```

4. **Set up Google Cloud credentials**
   - Create a service account in Google Cloud Console
   - Download the JSON key file
   - Place the key file in the project directory (default: `keyfile.json`)
   - Make sure the service account has the necessary permissions for Vertex AI

## Configuration

1. **Directory Structure**
   ```
   project-root/
   ├── catalogs/           # Place your PDF catalogs here
   ├── output/             # Output directory for extracted courses
   ├── logs/               # Log files
   ├── keyfile.json        # Your service account key
   └── gemini_extractor.py # Main script
   ```

2. **Environment Variables** (optional)
   You can set these environment variables or modify them in the script:
   ```bash
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   export SERVICE_ACCOUNT_KEYFILE="path/to/keyfile.json"
   ```

## Usage

### Basic Usage

```bash
python gemini_extractor.py
```

This will process all PDFs in the `catalogs` directory and save the results to the `output` directory.

### Command Line Arguments

You can customize the script's behavior using command line arguments:

```bash
python gemini_extractor.py \
    --project_id "your-project-id" \
    --keyfile "path/to/keyfile.json" \
    --catalogs_dir "./catalogs" \
    --output_dir "./output" \
    --max_workers 10
```

### Output

The script will create:
1. `output/gemini_extracted_courses.json` - Combined JSON file with all extracted courses
2. Log files in the `logs` directory with timestamps

## Troubleshooting

1. **Authentication Errors**
   - Verify your service account key file is valid
   - Ensure the service account has the necessary permissions
   - Check that the `GOOGLE_APPLICATION_CREDENTIALS` environment variable is set if not using the default keyfile location

2. **API Quota/Usage Limits**
   - The script includes retry logic for rate limiting
   - Check your Google Cloud project's quota and usage in the Google Cloud Console

3. **Logs**
   - Detailed logs are saved in the `logs` directory
   - The log level can be adjusted in the script if more verbose output is needed

## Performance Notes

- The script processes PDFs in parallel using multiple workers (default: 20)
- Each PDF is split into chunks for processing
- Progress is logged to both console and log files

## License

[Specify your license here]

## Contributing

[Your contribution guidelines here]
