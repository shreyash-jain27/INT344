import PyPDF2
import pdfplumber
import re
from typing import Dict, List, Tuple
from pathlib import Path
import json
import pandas as pd
from src.layout_analyzer import LayoutAnalyzer


class PDFProcessor:
    """
    Advanced PDF processor for annual reports
    Handles text extraction, section identification, and structure preservation
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.text_content = ""
        self.sections = {}
        self.metadata = {}
        self.pages_text = []
        self.layout_analyzer = LayoutAnalyzer(pdf_path)
        self.layouts = {}

    def extract_text(self) -> str:
        """Extract text from entire PDF with page information"""
        print("📄 Extracting text from PDF...")
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                self.metadata = {
                    'total_pages': len(pdf.pages),
                    'file_name': Path(self.pdf_path).name
                }

                all_text = []
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        self.pages_text.append({
                            'page_num': i + 1,
                            'text': page_text
                        })
                        all_text.append(f"\n=== PAGE {i+1} ===\n{page_text}")

                self.text_content = '\n'.join(all_text)
                
                # Perform layout analysis
                print("🔍 Analyzing layout structure...")
                raw_blocks = self.layout_analyzer.analyze_layout()
                
                # Transform list of blocks into dictionary by page
                self.layouts = {}
                for block in raw_blocks:
                    if block.page not in self.layouts:
                        self.layouts[block.page] = {
                            'blocks': [],
                            'width': 0, # Placeholder
                            'height': 0, # Placeholder
                            'image_count': 0 # Placeholder
                        }
                    
                    # Convert LayoutBlock object to dictionary for compatibility
                    block_dict = {
                        'text': block.text,
                        'type': block.block_type.title() if block.block_type else 'Text',
                        'avg_font_size': block.font_size if block.font_size else 0,
                        'top': block.bbox[1],
                        'bottom': block.bbox[3],
                        'x0': block.bbox[0],
                        'x1': block.bbox[2],
                        'page': block.page
                    }
                    self.layouts[block.page]['blocks'].append(block_dict)
                
                print(f"✓ Extracted text from {len(pdf.pages)} pages")
                print(f"✓ Total characters: {len(self.text_content):,}")
                return self.text_content

        except Exception as e:
            print(f"❌ Error extracting text: {e}")
            return ""

    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers (common pattern: Page 1, Page 2, etc.)
        text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)
        # Remove extra newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def identify_sections(self) -> Dict[str, str]:
        """
        Identify major sections in annual report using Layout Analysis.
        Returns a dictionary with section names and their content.
        """
        if not self.text_content:
            self.extract_text()

        print("\n🔍 Identifying sections (Context-Aware)...")

        # Enhanced section patterns
        section_patterns = {
            "Chairman's Message": [
                r"(?i)^(chairman'?s?\s+(message|statement|letter|address|report))$",
                r"(?i)^(letter\s+to\s+shareholders)$",
                r"(?i)^(from\s+the\s+chairman)$"
            ],
            "Executive Summary": [
                r"(?i)^(executive\s+summary)$",
                r"(?i)^(highlights?\s+of\s+the\s+year)$",
                r"(?i)^(key\s+highlights?)$"
            ],
            "Financial Performance": [
                r"(?i)^(financial\s+(performance|results|highlights|review))$",
                r"(?i)^(financial\s+year\s+\d{4})$",
                r"(?i)^(performance\s+overview)$"
            ],
            "Management Discussion": [
                r"(?i)^(management'?s?\s+discussion\s+and\s+analysis)$",
                r"(?i)^(md&a|mda)$",
                r"(?i)^(management\s+review)$"
            ],
            "Business Overview": [
                r"(?i)^(business\s+(overview|review|operations|segments?))$",
                r"(?i)^(about\s+(us|the\s+company))$",
                r"(?i)^(company\s+profile)$",
                r"(?i)^(our\s+business)$"
            ],
            "Corporate Governance": [
                r"(?i)^(corporate\s+governance)$",
                r"(?i)^(governance\s+report)$",
                r"(?i)^(board\s+of\s+directors)$"
            ],
            "Risk Management": [
                r"(?i)^(risk\s+(management|factors|assessment))$",
                r"(?i)^(risks?\s+and\s+concerns?)$"
            ],
            "Future Outlook": [
                r"(?i)^(future\s+(outlook|prospects?))$",
                r"(?i)^(forward\s+looking)$",
                r"(?i)^(way\s+forward)$",
                r"(?i)^(future\s+plans?)$"
            ],
            "Financial Statements": [
                r"(?i)^(financial\s+statements?)$",
                r"(?i)^(balance\s+sheet)$",
                r"(?i)^(income\s+statement)$",
                r"(?i)^(profit\s+and\s+loss)$",
                r"(?i)^(cash\s+flow\s+statement)$"
            ],
            "Auditor's Report": [
                r"(?i)^(auditor'?s?\s+report)$",
                r"(?i)^(independent\s+auditor)$",
                r"(?i)^(audit\s+report)$"
            ]
        }

        sections = {}
        current_section = "Introduction"
        current_content = []
        
        # Helper to check if text matches a section pattern
        def get_matched_section(text):
            text = text.strip()
            # Strict length check: Headers shouldn't be long sentences
            if len(text.split()) > 10: 
                return None
                
            for section_name, patterns in section_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text):
                        return section_name
            return None

        # Iterate through pages and blocks from layout analysis
        if self.layouts:
            sorted_pages = sorted(self.layouts.keys())
            for page_num in sorted_pages:
                layout = self.layouts[page_num]
                blocks = layout.get('blocks', [])
                
                # Sort blocks by vertical position (top to bottom)
                # LayoutAnalyzer already sorts them roughly, but let's be sure
                # Note: LayoutAnalyzer returns blocks in reading order usually
                
                for block in blocks:
                    text = block['text'].strip()
                    if not text:
                        continue
                        
                    # Check if this block is a header
                    matched_section = get_matched_section(text)
                    
                    # Context validation:
                    # 1. Must match regex
                    # 2. Must be classified as Heading/Title OR have large font
                    # 3. Must NOT be part of a sentence (already checked by length)
                    
                    is_visual_header = (
                        block.get('type') in ['Heading', 'Title', 'Header'] or 
                        block.get('avg_font_size', 0) > 12
                    )
                    
                    if matched_section and is_visual_header:
                        # Found a new section
                        if current_content:
                            sections[current_section] = self.clean_text('\n'.join(current_content))
                            print(f"  ✓ Found: {current_section} ({len(current_content)} blocks)")
                        
                        current_section = matched_section
                        current_content = []
                    else:
                        # Append to current section
                        current_content.append(text)
                        
        else:
            # Fallback to line-based if no layout info
            print("⚠️ No layout info available, falling back to line-based extraction")
            lines = self.text_content.split('\n')
            for line in lines:
                line = line.strip()
                if not line or line.startswith('==='): continue
                
                matched_section = get_matched_section(line)
                if matched_section:
                    if current_content:
                        sections[current_section] = self.clean_text('\n'.join(current_content))
                    current_section = matched_section
                    current_content = []
                else:
                    current_content.append(line)

        # Save last section
        if current_content:
            sections[current_section] = self.clean_text('\n'.join(current_content))
            print(f"  ✓ Found: {current_section} ({len(current_content)} blocks)")

        self.sections = sections
        print(f"\n✅ Total sections identified: {len(sections)}")
        return sections

    def extract_tables(self, min_rows=2) -> List[Dict]:
        """
        Extract tables from PDF with improved settings
        Args:
            min_rows: Minimum rows to consider as a valid table
        """
        print("\n📊 Extracting tables with improved settings...")
        tables = []

        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Use custom table settings for better extraction
                    table_settings = {
                        "vertical_strategy": "lines_strict",
                        "horizontal_strategy": "lines_strict",
                        "explicit_vertical_lines": [],
                        "explicit_horizontal_lines": [],
                        "snap_tolerance": 3,
                        "join_tolerance": 3,
                        "edge_min_length": 3,
                        "min_words_vertical": 3,
                        "min_words_horizontal": 1,
                        "intersection_tolerance": 3,
                        "text_tolerance": 3,
                        "text_x_tolerance": 3,
                        "text_y_tolerance": 3,
                    }

                    # Try multiple extraction strategies
                    page_tables = []

                    # Strategy 1: Default extraction
                    try:
                        default_tables = page.extract_tables()
                        if default_tables:
                            page_tables.extend(default_tables)
                    except:
                        pass

                    # Strategy 2: With custom settings
                    try:
                        custom_tables = page.extract_tables(
                            table_settings=table_settings)
                        if custom_tables:
                            page_tables.extend(custom_tables)
                    except:
                        pass

                    # Process extracted tables
                    for table_idx, table in enumerate(page_tables):
                        if table and len(table) >= min_rows:
                            # Clean the table
                            cleaned_table = self._clean_table(table)

                            if cleaned_table and len(cleaned_table) >= min_rows:
                                # Try to convert to DataFrame for better display
                                try:
                                    # Handle duplicate column names
                                    columns = cleaned_table[0]
                                    # Make column names unique
                                    seen = {}
                                    unique_columns = []
                                    for col in columns:
                                        if col in seen:
                                            seen[col] += 1
                                            unique_columns.append(
                                                f"{col}_{seen[col]}")
                                        else:
                                            seen[col] = 0
                                            unique_columns.append(col)

                                    df = pd.DataFrame(
                                        cleaned_table[1:], columns=unique_columns)

                                    tables.append({
                                        'page': page_num + 1,
                                        'table_number': len(tables) + 1,
                                        'rows': len(cleaned_table),
                                        'columns': len(cleaned_table[0]) if cleaned_table else 0,
                                        'data': cleaned_table,
                                        'dataframe': df,
                                        'has_headers': True
                                    })
                                except Exception as e:
                                    # If DataFrame conversion fails, store raw data
                                    tables.append({
                                        'page': page_num + 1,
                                        'table_number': len(tables) + 1,
                                        'rows': len(cleaned_table),
                                        'columns': len(cleaned_table[0]) if cleaned_table else 0,
                                        'data': cleaned_table,
                                        'dataframe': None,
                                        'has_headers': False
                                    })

            # Remove duplicate tables (sometimes both strategies extract the same table)
            tables = self._remove_duplicate_tables(tables)

            print(f"✓ Extracted {len(tables)} valid tables")
            return tables

        except Exception as e:
            print(f"❌ Error extracting tables: {e}")
            return []

    def _clean_table(self, table: List[List]) -> List[List]:
        """Clean and validate table data"""
        if not table:
            return []

        cleaned = []
        for row in table:
            if row:  # Skip completely empty rows
                # Clean each cell
                cleaned_row = []
                for cell in row:
                    if cell is None or cell == '':
                        cleaned_row.append('')
                    else:
                        # Clean the cell value
                        cleaned_cell = str(cell).strip()
                        cleaned_row.append(cleaned_cell)

                # Only add row if it has at least one non-empty cell
                if any(cell for cell in cleaned_row):
                    cleaned.append(cleaned_row)

        return cleaned

    def _remove_duplicate_tables(self, tables: List[Dict]) -> List[Dict]:
        """Remove duplicate tables based on content similarity"""
        if len(tables) <= 1:
            return tables

        unique_tables = []
        seen_signatures = set()

        for table in tables:
            # Create a signature based on dimensions and first few cells
            data = table['data']
            if data:
                signature = (
                    table['page'],
                    table['rows'],
                    table['columns'],
                    str(data[0][:3]) if len(data[0]) >= 3 else str(data[0])
                )

                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    unique_tables.append(table)

        return unique_tables

    def extract_tables_detailed(self) -> Dict[str, any]:
        """
        Extract tables with detailed analysis
        Returns statistics and the tables
        """
        tables = self.extract_tables()

        stats = {
            'total_tables': len(tables),
            'tables_by_page': {},
            'total_rows': 0,
            'total_columns': 0,
            'largest_table': None,
            'tables_with_headers': 0
        }

        for table in tables:
            page = table['page']
            stats['tables_by_page'][page] = stats['tables_by_page'].get(
                page, 0) + 1
            stats['total_rows'] += table['rows']
            stats['total_columns'] += table['columns']

            if table.get('has_headers'):
                stats['tables_with_headers'] += 1

            if not stats['largest_table'] or table['rows'] > stats['largest_table']['rows']:
                stats['largest_table'] = table

        return {
            'tables': tables,
            'stats': stats
        }

    def get_section_stats(self) -> Dict[str, int]:
        """Get statistics about extracted sections"""
        if not self.sections:
            self.identify_sections()

        stats = {}
        for section_name, content in self.sections.items():
            stats[section_name] = {
                'word_count': len(content.split()),
                'char_count': len(content),
                'line_count': len(content.split('\n'))
            }
        return stats

    def save_sections(self, output_dir: str = "data/outputs"):
        """Save extracted sections to separate files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create a subfolder for this PDF
        pdf_name = Path(self.pdf_path).stem
        pdf_output_dir = output_path / pdf_name
        pdf_output_dir.mkdir(exist_ok=True)

        # Save each section
        for section_name, content in self.sections.items():
            file_name = section_name.replace(
                " ", "_").replace("'", "") + ".txt"
            file_path = pdf_output_dir / file_name
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Section: {section_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(content)

        # Save metadata
        metadata_path = pdf_output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'file_name': self.metadata.get('file_name', ''),
                'total_pages': self.metadata.get('total_pages', 0),
                'sections': list(self.sections.keys()),
                'stats': self.get_section_stats()
            }, f, indent=2)

        print(f"\n💾 Saved sections to: {pdf_output_dir}")
        return str(pdf_output_dir)
