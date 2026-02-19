"""Parse Pittsfield Township Record Card PDFs using OCR (pymupdf + tesseract)."""

from dataclasses import dataclass, field
import io
import re

import pymupdf
import pytesseract
from PIL import Image


@dataclass
class AssessmentYear:
    """One row from the assessment history table."""
    year: int
    land_value: int | None = None
    building_value: int | None = None
    assessed_value: int | None = None
    taxable_value: int | None = None


@dataclass
class PropertyData:
    """Parsed property data from a Record Card PDF."""
    parcel_number: str = ""
    address: str = ""
    area_code: str = ""
    subdivision: str = ""
    sev_2026: int = 0          # 2026 Assessed Value (= SEV)
    tcv_2026: int = 0          # 2026 Est TCV from page 1
    land_value: int = 0        # Total Est. Land value
    ecf: float | None = None
    floor_area: int = 0
    ground_area: int = 0
    basement_sf: int = 0
    year_built: int = 0
    condition: str = ""
    style: str = ""
    effective_age: int = 0
    total_base_new: int = 0
    total_depr_cost: int = 0
    estimated_tcv_cost: int = 0  # Estimated T.C.V. from cost approach
    assessment_history: list[AssessmentYear] = field(default_factory=list)
    raw_text_page1: str = ""
    raw_text_page2: str = ""


def _clean_number(s: str) -> int | None:
    """Parse an integer from OCR text, handling spaces within numbers like '192 , 337'."""
    if not s:
        return None
    # Remove all spaces, commas, dollar signs, letters at end (c/C/S)
    cleaned = re.sub(r"[,\s$]", "", s)
    cleaned = re.sub(r"[a-zA-Z]+$", "", cleaned)
    try:
        return int(cleaned)
    except ValueError:
        return None


def _ocr_page(doc, page_num: int, dpi: int = 300) -> str:
    """Render a PDF page to image and OCR it."""
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def _normalize_ocr_numbers(text: str) -> str:
    """Fix OCR artifacts where spaces appear within numbers: '192 , 337' → '192,337'."""
    # Remove spaces around commas between digits: "192 , 337" → "192,337"
    text = re.sub(r"(\d)\s*,\s*(\d)", r"\1,\2", text)
    return text


def _parse_assessment_history(text: str, prop: PropertyData) -> None:
    """Parse the assessment history table from page 1 OCR text."""
    # Normalize OCR number artifacts first
    norm = _normalize_ocr_numbers(text)

    # Now numbers are clean: "48,900 192,337 241,237 231,890c"
    history_pattern = re.compile(
        r"[|\(]?\s*(202[3-6])\s+"     # year
        r"([\d,]+)\s+"                # land value
        r"([\d,]+)\s+"                # building value
        r"([\d,]+)\s+"                # assessed value (= SEV)
        r"([\d,]+)[cCsS]"            # taxable value (ends with c/C/s/S)
    )
    for m in history_pattern.finditer(norm):
        year = int(m.group(1))
        ay = AssessmentYear(
            year=year,
            land_value=_clean_number(m.group(2)),
            building_value=_clean_number(m.group(3)),
            assessed_value=_clean_number(m.group(4)),
            taxable_value=_clean_number(m.group(5)),
        )
        prop.assessment_history.append(ay)
        if year == 2026:
            prop.sev_2026 = ay.assessed_value or 0

    # Sort history by year
    prop.assessment_history.sort(key=lambda a: a.year)


def _parse_page1(text: str, prop: PropertyData) -> None:
    """Extract fields from page 1 OCR text."""
    # Parcel Number
    m = re.search(r"Parcel Number:\s*(L\s*-[\d\s-]+)", text)
    if m:
        prop.parcel_number = re.sub(r"\s+", "", m.group(1))

    # Property Address — line starting with a street number before "School:"
    m = re.search(r"(?:Property Address|4\d{3}|[A-Z]\d{3,4})\s*\n?\s*(\d+\s+[A-Z][A-Z\s]+(?:DR|LN|CT|RD|AVE|WAY|BLVD|CIR|ST|PL|TRL))", text)
    if m:
        prop.address = " ".join(m.group(1).split())
    else:
        # Fallback: look for a street address pattern
        m = re.search(r"\b(\d{3,5}\s+[A-Z][A-Z\s]+(?:DR|LN|CT|RD|AVE|WAY|BLVD|CIR|ST|PL|TRL))\b", text)
        if m:
            prop.address = " ".join(m.group(1).split())

    # ECF Area Code and Subdivision from Land Table
    # e.g. "Land Table AR-4.AR4-MEADOWS OF ARBOR RIDGE" or "Land Table ARF.ARBOR FARMS"
    m = re.search(r"Land\s+(?:Value\s+Estimates\s+for\s+)?Land\s+Table\s+(\S+?)\.(\S.*?)(?:\n|$)", text)
    if m:
        prop.area_code = m.group(1).strip()
        raw_sub = m.group(2).strip()
        # Strip area-code prefix before the first space/dash that starts the real name
        # e.g. "AR4-MEADOWS OF ARBOR RIDGE" → "MEADOWS OF ARBOR RIDGE"
        # e.g. "ARBOR FARMS" → "ARBOR FARMS"
        sub_match = re.match(r"^[A-Z0-9]+-(.+)", raw_sub)
        if sub_match:
            raw_sub = sub_match.group(1).strip()
        prop.subdivision = raw_sub.replace("-", " ").replace("_", " ").strip()

    # 2026 Est TCV
    m = re.search(r"2026\s+Est\s+TCV\s+([\d,\s]+)", text)
    if m:
        prop.tcv_2026 = _clean_number(m.group(1)) or 0

    # Total Est. Land value
    m = re.search(r"Total\s+Est\.\s+Land\s+value\s*=\s*([\d,\s]+)", text)
    if m:
        prop.land_value = _clean_number(m.group(1)) or 0

    # Assessment History Table
    # Rows look like: |2026 49,600 193,132 242,732 213,794c
    # or: (2026 56,200 206,581 262,781 245,533C
    # OCR introduces spaces within numbers: "192 , 337" or "251, 380"
    # Strategy: normalize OCR digit-space artifacts, then parse with simple regex
    _parse_assessment_history(text, prop)


def _parse_page2(text: str, prop: PropertyData) -> None:
    """Extract fields from page 2 OCR text."""
    # Building Style — may appear as "Building Style:\n\nTWO-STORY" (double newline possible)
    m = re.search(r"Building\s+Style:\s*((?:TWO|ONE|TRI|BI|SPLIT|RANCH|CAPE|COLONIAL|BUNGALOW)[\w-]*)", text)
    if not m:
        # Fallback: look for style in cost estimate section "Single Family TWO-STORY"
        m = re.search(r"Single\s+Family\s+((?:TWO|ONE|TRI|BI|SPLIT|RANCH|CAPE|COLONIAL|BUNGALOW)[\w-]*)", text, re.IGNORECASE)
    if m:
        prop.style = m.group(1).strip().upper()

    # Year Built — look for "Blt YYYY" or "Yr Built" pattern
    m = re.search(r"(?:Blt|B[Il]t)\s+(\d{4})", text)
    if not m:
        m = re.search(r"(\d{4})\s*(?:Actua|Actual)", text)
    if not m:
        m = re.search(r"Yr\s+Built.*?(\d{4})", text)
    if m:
        prop.year_built = int(m.group(1))

    # Condition
    m = re.search(r"Condition:\s*(\w+)", text)
    if m:
        prop.condition = m.group(1).capitalize()

    # Effective Age
    m = re.search(r"Effec\.\s*Age:\s*(\d+)", text)
    if m:
        prop.effective_age = int(m.group(1))

    # Floor Area
    m = re.search(r"Floor\s+Area:\s*(\d[\d,]+)", text)
    if m:
        prop.floor_area = _clean_number(m.group(1)) or 0

    # Ground Area
    m = re.search(r"Ground\s+Area\s*=\s*(\d[\d,]+)\s*SF", text)
    if m:
        prop.ground_area = _clean_number(m.group(1)) or 0

    # Basement SF
    m = re.search(r"Basement:\s*(\d[\d,]+)\s*S\.?F\.?", text)
    if m:
        prop.basement_sf = _clean_number(m.group(1)) or 0

    # Total Base New
    m = re.search(r"Total\s+Base\s+New:\s*([\d,\s]+)", text)
    if m:
        prop.total_base_new = _clean_number(m.group(1)) or 0

    # Total Depr Cost
    m = re.search(r"Total\s+Depr\s+Cost:\s*([\d,\s]+)", text)
    if m:
        prop.total_depr_cost = _clean_number(m.group(1)) or 0

    # ECF — pattern: "X 0.802" or "X 0.872" (near Total Depr Cost)
    m = re.search(r"(?:E\.?C\.?F\.?|X)\s+(0\.\d{2,4})", text)
    if m:
        try:
            prop.ecf = float(m.group(1))
        except ValueError:
            pass

    # Estimated T.C.V. (cost approach) — OCR may produce T.c.V. or T.C.V.
    m = re.search(r"Estimated\s+T\.?[cC]\.?V\.?:?\s*([\d,\s]+)", text)
    if m:
        prop.estimated_tcv_cost = _clean_number(m.group(1)) or 0


def parse_rc_pdf(pdf_bytes: bytes) -> PropertyData:
    """Parse a Record Card PDF and return structured property data."""
    prop = PropertyData()
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    if len(doc) >= 1:
        text1 = _ocr_page(doc, 0)
        prop.raw_text_page1 = text1
        _parse_page1(text1, prop)

    if len(doc) >= 2:
        text2 = _ocr_page(doc, 1)
        prop.raw_text_page2 = text2
        _parse_page2(text2, prop)

    doc.close()

    # If SEV wasn't found from history table, derive from TCV
    if prop.sev_2026 == 0 and prop.tcv_2026 > 0:
        prop.sev_2026 = prop.tcv_2026 // 2

    return prop
