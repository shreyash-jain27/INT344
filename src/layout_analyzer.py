# ============================================
# src/layout_analyzer.py
# Advanced layout detection and analysis
# ============================================

import pdfplumber
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import re

@dataclass
class LayoutBlock:
    """Represents a detected layout block"""
    block_type: str  # 'heading', 'paragraph', 'table', 'image', 'footer'
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    page: int
    confidence: float
    font_size: Optional[float] = None
    font_name: Optional[str] = None
    
    def __repr__(self):
        return f"LayoutBlock(type={self.block_type}, page={self.page}, text={self.text[:50]}...)"


class LayoutAnalyzer:
    """
    Advanced layout analysis for PDF documents
    Detects headings, paragraphs, tables, images, and reading order
    """
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.blocks = []
        self.pages = []
        
    def analyze_layout(self) -> List[LayoutBlock]:
        """
        Main method to analyze document layout
        Returns list of detected layout blocks
        """
        print("\n🔍 Analyzing document layout...")
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    print(f"  Processing page {page_num}/{len(pdf.pages)}...")
                    
                    # Extract text with detailed information
                    words = page.extract_words(
                        x_tolerance=3,
                        y_tolerance=3,
                        keep_blank_chars=False,
                        use_text_flow=True
                    )
                    
                    # Group words into blocks
                    page_blocks = self._group_into_blocks(words, page_num, page)
                    
                    # Classify blocks
                    classified_blocks = self._classify_blocks(page_blocks, page)
                    
                    self.blocks.extend(classified_blocks)
                    
            print(f"✓ Detected {len(self.blocks)} layout blocks")
            
            # Sort blocks by reading order
            self.blocks = self._determine_reading_order(self.blocks)
            
            return self.blocks
            
        except Exception as e:
            print(f"❌ Error analyzing layout: {e}")
            return []
    
    def _group_into_blocks(self, words: List[Dict], page_num: int, page) -> List[LayoutBlock]:
        """Group words into coherent text blocks"""
        if not words:
            return []
        
        blocks = []
        current_block = {
            'words': [],
            'bbox': None
        }
        
        # Sort words by vertical position first, then horizontal
        sorted_words = sorted(words, key=lambda w: (w['top'], w['x0']))
        
        for i, word in enumerate(sorted_words):
            if not current_block['words']:
                # Start new block
                current_block['words'] = [word]
                current_block['bbox'] = [word['x0'], word['top'], word['x1'], word['bottom']]
            else:
                # Check if word belongs to current block
                prev_word = current_block['words'][-1]
                
                # Vertical distance
                v_dist = abs(word['top'] - prev_word['top'])
                
                # Horizontal distance
                h_dist = word['x0'] - prev_word['x1']
                
                # Same line if vertical distance is small
                if v_dist < 5 and h_dist < 50:
                    current_block['words'].append(word)
                    # Expand bbox
                    current_block['bbox'][2] = max(current_block['bbox'][2], word['x1'])
                    current_block['bbox'][3] = max(current_block['bbox'][3], word['bottom'])
                # New line in same block
                elif v_dist < 20 and abs(word['x0'] - current_block['bbox'][0]) < 30:
                    current_block['words'].append(word)
                    current_block['bbox'][2] = max(current_block['bbox'][2], word['x1'])
                    current_block['bbox'][3] = word['bottom']
                else:
                    # Finish current block and start new one
                    text = ' '.join(w['text'] for w in current_block['words'])
                    
                    blocks.append(LayoutBlock(
                        block_type='text',
                        text=text,
                        bbox=tuple(current_block['bbox']),
                        page=page_num,
                        confidence=1.0,
                        font_size=self._estimate_font_size(current_block['words'])
                    ))
                    
                    # Start new block
                    current_block = {
                        'words': [word],
                        'bbox': [word['x0'], word['top'], word['x1'], word['bottom']]
                    }
        
        # Add last block
        if current_block['words']:
            text = ' '.join(w['text'] for w in current_block['words'])
            blocks.append(LayoutBlock(
                block_type='text',
                text=text,
                bbox=tuple(current_block['bbox']),
                page=page_num,
                confidence=1.0,
                font_size=self._estimate_font_size(current_block['words'])
            ))
        
        return blocks
    
    def _classify_blocks(self, blocks: List[LayoutBlock], page) -> List[LayoutBlock]:
        """Classify blocks into headings, paragraphs, etc."""
        classified = []
        
        for block in blocks:
            # Classify based on multiple features
            block_type = self._determine_block_type(block, page)
            block.block_type = block_type
            classified.append(block)
        
        return classified
    
    def _determine_block_type(self, block: LayoutBlock, page) -> str:
        """Determine the type of a text block"""
        text = block.text.strip()
        
        # Empty or very short text
        if len(text) < 5:
            return 'fragment'
        
        # Check for heading characteristics
        is_heading = self._is_heading(block, page)
        if is_heading:
            return 'heading'
        
        # Check for footer/header
        if self._is_footer_or_header(block, page):
            return 'footer'
        
        # Check if it's a table caption
        if re.match(r'^Table\s+\d+', text, re.IGNORECASE):
            return 'table_caption'
        
        # Check if it's a figure caption
        if re.match(r'^(Figure|Fig\.)\s+\d+', text, re.IGNORECASE):
            return 'figure_caption'
        
        # Default to paragraph
        return 'paragraph'
    
    def _is_heading(self, block: LayoutBlock, page) -> bool:
        """Detect if block is a heading"""
        text = block.text.strip()
        
        # Features that indicate heading
        features = []
        
        # 1. Font size (if available)
        if block.font_size and block.font_size > 12:
            features.append(True)
        
        # 2. All caps
        if text.isupper() and len(text) > 5:
            features.append(True)
        
        # 3. Short length
        if len(text.split()) < 10:
            features.append(True)
        
        # 4. Title case
        if text.istitle():
            features.append(True)
        
        # 5. Ends without punctuation
        if not text.endswith(('.', ',', ';', ':', '!', '?')):
            features.append(True)
        
        # 6. Contains numbers (section numbering)
        if re.match(r'^\d+\.?\s+', text):
            features.append(True)
        
        # 7. Position (top of page)
        if block.bbox[1] < page.height * 0.15:
            features.append(True)
        
        # Need at least 3 features to be a heading
        return sum(features) >= 3
    
    def _is_footer_or_header(self, block: LayoutBlock, page) -> bool:
        """Detect if block is a header or footer"""
        # Check position
        is_top = block.bbox[1] < page.height * 0.1
        is_bottom = block.bbox[3] > page.height * 0.9
        
        # Check content
        text = block.text.strip().lower()
        has_page_num = bool(re.search(r'\b\d{1,4}\b', text))
        is_short = len(text.split()) < 8
        
        return (is_top or is_bottom) and (has_page_num or is_short)
    
    def _estimate_font_size(self, words: List[Dict]) -> float:
        """Estimate font size from word height"""
        if not words:
            return 10.0
        
        heights = [w['bottom'] - w['top'] for w in words]
        return sum(heights) / len(heights)
    
    def _determine_reading_order(self, blocks: List[LayoutBlock]) -> List[LayoutBlock]:
        """Sort blocks by reading order (top to bottom, left to right)"""
        return sorted(blocks, key=lambda b: (b.page, b.bbox[1], b.bbox[0]))
    
    def get_headings(self) -> List[LayoutBlock]:
        """Extract all headings"""
        return [b for b in self.blocks if b.block_type == 'heading']
    
    def get_paragraphs(self) -> List[LayoutBlock]:
        """Extract all paragraphs"""
        return [b for b in self.blocks if b.block_type == 'paragraph']
    
    def extract_document_structure(self) -> Dict:
        """Extract hierarchical document structure"""
        structure = {
            'headings': [],
            'paragraphs': [],
            'tables': [],
            'footers': [],
            'total_blocks': len(self.blocks)
        }
        
        for block in self.blocks:
            if block.block_type == 'heading':
                structure['headings'].append({
                    'text': block.text,
                    'page': block.page,
                    'position': block.bbox
                })
            elif block.block_type == 'paragraph':
                structure['paragraphs'].append({
                    'text': block.text[:100] + '...',
                    'page': block.page,
                    'word_count': len(block.text.split())
                })
            elif block.block_type in ['table_caption', 'figure_caption']:
                structure['tables'].append({
                    'caption': block.text,
                    'page': block.page
                })
            elif block.block_type == 'footer':
                structure['footers'].append({
                    'text': block.text,
                    'page': block.page
                })
        
        return structure
    
    def visualize_layout(self, page_num: int = 1, output_path: str = "layout_visualization.png"):
        """
        Create visualization of detected layout
        """
        try:
            from pdf2image import convert_from_path
            
            print(f"\n🎨 Creating layout visualization for page {page_num}...")
            
            # Convert PDF page to image
            images = convert_from_path(
                self.pdf_path,
                first_page=page_num,
                last_page=page_num,
                dpi=150
            )
            
            if not images:
                print("❌ Could not convert PDF to image")
                return None
            
            img = images[0]
            draw = ImageDraw.Draw(img)
            
            # Get blocks for this page
            page_blocks = [b for b in self.blocks if b.page == page_num]
            
            # Color coding
            colors = {
                'heading': 'red',
                'paragraph': 'blue',
                'table_caption': 'green',
                'figure_caption': 'purple',
                'footer': 'gray',
                'text': 'orange'
            }
            
            # Draw rectangles around blocks
            print(f"   Image size: {img.width}x{img.height}")
            print(f"   PDF page size (from metadata): {page.width if 'page' in locals() else 'unknown'}x{page.height if 'page' in locals() else 'unknown'}")
            
            # Simple scaling factor: ratio of pixels to points
            # Standard PDF points are 72 DPI. 
            # If DPI=150 in convert_from_path, scale is 150/72
            scale_x = img.width / 612.0  # Default A4 width
            scale_y = img.height / 792.0 # Default A4 height (Letter size actually, safer to use points)
            
            # Use actual page cropbox if possible
            try:
                with pdfplumber.open(self.pdf_path) as pdf:
                    p = pdf.pages[page_num-1]
                    scale_x = img.width / float(p.width)
                    scale_y = img.height / float(p.height)
            except:
                pass

            for block in page_blocks:
                color = colors.get(block.block_type, 'black')
                
                # Scale bbox (PDF coordinates to image coordinates)
                bbox = [
                    block.bbox[0] * scale_x,
                    block.bbox[1] * scale_y,
                    block.bbox[2] * scale_x,
                    block.bbox[3] * scale_y
                ]
                
                draw.rectangle(bbox, outline=color, width=3)
                
                # Add label
                try:
                    draw.text(
                        (bbox[0], bbox[1] - 15),
                        block.block_type,
                        fill=color
                    )
                except:
                    pass
            
            # Save
            img.save(output_path)
            print(f"✓ Visualization saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Visualization error: {e}")
            return None
    
    def get_statistics(self) -> Dict:
        """Get statistics about the layout"""
        stats = {
            'total_blocks': len(self.blocks),
            'by_type': {},
            'by_page': {}
        }
        
        for block in self.blocks:
            # Count by type
            stats['by_type'][block.block_type] = stats['by_type'].get(block.block_type, 0) + 1
            
            # Count by page
            stats['by_page'][block.page] = stats['by_page'].get(block.page, 0) + 1
        
        return stats


# ============================================
# Helper function for integration
# ============================================

def analyze_pdf_layout(pdf_path: str) -> Dict:
    """
    Convenience function to analyze PDF layout
    Returns structure and blocks
    """
    analyzer = LayoutAnalyzer(pdf_path)
    blocks = analyzer.analyze_layout()
    structure = analyzer.extract_document_structure()
    stats = analyzer.get_statistics()
    
    return {
        'blocks': blocks,
        'structure': structure,
        'stats': stats,
        'analyzer': analyzer
    }