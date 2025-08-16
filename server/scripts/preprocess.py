import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Any

# Import the processing functions from each module
from pdf import process_pdf_file
from docu import process_docx_file
from image import process_image_file

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Universal document processor that can handle PDF, DOCX, and image files.
    Automatically detects file type and uses the appropriate extraction method.
    """
    
    def __init__(self, output_dir: str = "extracted_texts", use_easyocr: bool = True):
        """
        Initialize the document processor.
        
        Args:
            output_dir: Directory to save extracted text files
            use_easyocr: Whether to use EasyOCR for image text extraction
        """
        self.output_dir = output_dir
        self.use_easyocr = use_easyocr
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Ensure output directory exists
        Path(output_dir).mkdir(exist_ok=True)
        
        # Supported file extensions
        self.supported_extensions = {
            # PDF files
            '.pdf': 'pdf',
            
            # DOCX files
            '.docx': 'docx',
            '.doc': 'docx',  # Note: .doc files may need conversion first
            
            # Image files
            '.png': 'image',
            '.jpg': 'image',
            '.jpeg': 'image',
            '.gif': 'image',
            '.bmp': 'image',
            '.tiff': 'image',
            '.tif': 'image',
            '.webp': 'image'
        }
    
    def detect_file_type(self, file_path: str) -> str:
        """
        Detect the file type based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File type string ('pdf', 'docx', 'image', or 'unknown')
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = Path(file_path).suffix.lower()
        
        return self.supported_extensions.get(file_extension, 'unknown')
    
    def process_document(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Process a document file and extract text content.
        
        Args:
            file_path: Path to the document file
            **kwargs: Additional arguments passed to specific processors
            
        Returns:
            Dictionary containing extraction results
        """
        self.logger.info(f"Processing document: {file_path}")
        
        # Detect file type
        file_type = self.detect_file_type(file_path)
        self.logger.info(f"Detected file type: {file_type}")
        
        if file_type == 'unknown':
            supported_exts = ', '.join(self.supported_extensions.keys())
            raise ValueError(f"Unsupported file type. Supported extensions: {supported_exts}")
        
        # Process based on file type
        try:
            if file_type == 'pdf':
                self.logger.info("Using PDF processor...")
                return process_pdf_file(
                    pdf_path=file_path,
                    output_dir=self.output_dir,
                    use_easyocr=self.use_easyocr,
                    use_fallback_ocr=kwargs.get('use_fallback_ocr', True)
                )
            
            elif file_type == 'docx':
                self.logger.info("Using DOCX processor...")
                return process_docx_file(
                    docx_path=file_path,
                    output_dir=self.output_dir,
                    use_easyocr=self.use_easyocr
                )
            
            elif file_type == 'image':
                self.logger.info("Using image processor...")
                return process_image_file(
                    image_path=file_path,
                    output_dir=self.output_dir,
                    save_result=kwargs.get('save_result', True)
                )
            
        except Exception as e:
            self.logger.error(f"Error processing {file_type} file: {e}")
            raise
    
    def process_multiple_documents(self, file_paths: list) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple documents in batch.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Dictionary mapping file paths to their extraction results
        """
        results = {}
        
        self.logger.info(f"Processing {len(file_paths)} documents...")
        
        for i, file_path in enumerate(file_paths, 1):
            self.logger.info(f"[{i}/{len(file_paths)}] Processing: {Path(file_path).name}")
            
            try:
                result = self.process_document(file_path)
                results[file_path] = result
                self.logger.info(f"Successfully processed: {Path(file_path).name}")
                
            except Exception as e:
                self.logger.error(f"Failed to process {Path(file_path).name}: {e}")
                results[file_path] = {
                    'error': str(e),
                    'file_type': self.detect_file_type(file_path) if os.path.exists(file_path) else 'unknown'
                }
        
        return results
    
    def get_supported_extensions(self) -> list:
        """
        Get list of supported file extensions.
        
        Returns:
            List of supported file extensions
        """
        return list(self.supported_extensions.keys())
    
    def print_summary(self, results: Dict[str, Any]):
        """
        Print a summary of the extraction results.
        
        Args:
            results: Results dictionary from document processing
        """
        self.logger.info("=" * 60)
        self.logger.info("DOCUMENT PROCESSING SUMMARY")
        self.logger.info("=" * 60)
        
        if 'error' in results:
            self.logger.error(f"Processing failed: {results['error']}")
            return
        
        # Common fields across all processors
        if 'output_file' in results:
            self.logger.info(f"Output file: {results['output_file']}")
        
        if 'combined_text' in results:
            char_count = len(results['combined_text'])
            self.logger.info(f"Total characters extracted: {char_count:,}")
        
        # PDF-specific summary
        if 'total_pages' in results:
            self.logger.info(f"Pages processed: {results['total_pages']}")
            self.logger.info(f"Images found: {results.get('images_found', 0)}")
        
        # DOCX-specific summary
        if 'native_text' in results and isinstance(results['native_text'], dict):
            native_text = results['native_text']
            self.logger.info(f"Paragraphs extracted: {native_text.get('total_paragraphs', 0)}")
            self.logger.info(f"Tables extracted: {native_text.get('total_tables', 0)}")
            if 'images' in results:
                self.logger.info(f"Images processed: {len(results['images'])}")
        
        # Image-specific summary
        if 'confidence' in results:
            self.logger.info(f"OCR confidence: {results['confidence']:.2f}")
            self.logger.info(f"Words extracted: {results.get('word_count', 0)}")
        
        self.logger.info("=" * 60)


def main():
    """
    Command-line interface for the universal document processor.
    """
    parser = argparse.ArgumentParser(
        description='Universal Document Text Extractor - Process PDF, DOCX, and image files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py document.pdf
  python main.py report.docx --output-dir results
  python main.py image.png --no-ocr
  python main.py file1.pdf file2.docx file3.png
        '''
    )
    
    parser.add_argument(
        'files',
        nargs='+',
        help='Path(s) to document file(s) to process'
    )
    
    parser.add_argument(
        '--output-dir',
        default='extracted_texts',
        help='Directory to save extracted text files (default: extracted_texts)'
    )
    
    parser.add_argument(
        '--no-ocr',
        action='store_true',
        help='Disable EasyOCR for image text extraction'
    )
    
    parser.add_argument(
        '--no-fallback-ocr',
        action='store_true',
        help='Disable fallback OCR for PDFs (applies only to PDF files)'
    )
    
    parser.add_argument(
        '--list-supported',
        action='store_true',
        help='List supported file extensions and exit'
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = DocumentProcessor(
        output_dir=args.output_dir,
        use_easyocr=not args.no_ocr
    )
    
    # List supported extensions if requested
    if args.list_supported:
        logger.info("Supported file extensions:")
        for ext in sorted(processor.get_supported_extensions()):
            file_type = processor.supported_extensions[ext]
            logger.info(f"  {ext} -> {file_type} processor")
        return 0
    
    # Process files
    try:
        if len(args.files) == 1:
            # Single file processing
            file_path = args.files[0]
            result = processor.process_document(
                file_path,
                use_fallback_ocr=not args.no_fallback_ocr
            )
            processor.print_summary(result)
            
        else:
            # Multiple file processing
            results = processor.process_multiple_documents(args.files)
            
            # Print summary for each file
            successful_count = 0
            for file_path, result in results.items():
                logger.info(f"File: {Path(file_path).name}")
                if 'error' not in result:
                    successful_count += 1
                processor.print_summary(result)
            
            logger.info(f"Successfully processed {successful_count}/{len(args.files)} files")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
