import sys
import pathlib
import argparse
import tempfile

try:
  import markdown
except ImportError:
  markdown = None

try:
  from weasyprint import HTML
except ImportError:
  HTML = None

try:
  from xhtml2pdf import pisa
except ImportError:
  pisa = None

#!/usr/bin/env python3
"""
Simple CLI tool to convert Markdown files to PDF.

Requirements (install one of these backends):
  pip install markdown weasyprint
or
  pip install markdown xhtml2pdf

Usage:
  python report_to_pdf.py input.md output.pdf
"""
# Try WeasyPrint first (better quality), fall back to xhtml2pdf if not available
if HTML is not None:
  _BACKEND = "weasyprint"
elif pisa is not None:
  _BACKEND = "xhtml2pdf"
else:
  _BACKEND = None


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
  @page {{
    size: A4;
    margin: 2cm;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.6;
    color: #333;
  }}
  h1 {{
    font-size: 24pt;
    font-weight: bold;
    margin-top: 0;
    margin-bottom: 0.5em;
    page-break-after: avoid;
    border-bottom: 2px solid #333;
    padding-bottom: 0.3em;
  }}
  h2 {{
    font-size: 16pt;
    font-weight: bold;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    page-break-after: avoid;
    border-bottom: 1px solid #ddd;
    padding-bottom: 0.2em;
  }}
  h3 {{
    font-size: 13pt;
    font-weight: bold;
    margin-top: 1em;
    margin-bottom: 0.5em;
    page-break-after: avoid;
  }}
  h4 {{
    font-size: 11pt;
    font-weight: bold;
    margin-top: 0.8em;
    margin-bottom: 0.4em;
  }}
  p {{
    margin: 0.5em 0;
  }}
  ul, ol {{
    margin: 0.5em 0;
    padding-left: 2em;
  }}
  li {{
    margin: 0.3em 0;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    page-break-inside: avoid;
  }}
  th, td {{
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
    vertical-align: top;
  }}
  th {{
    background-color: #f5f5f5;
    font-weight: bold;
  }}
  img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0.5em 0;
    page-break-inside: avoid;
  }}
  pre, code {{
    font-family: "Courier New", Consolas, monospace;
    font-size: 9pt;
    background: #f5f5f5;
  }}
  pre {{
    padding: 0.7em;
    border-radius: 3px;
    overflow-x: auto;
    page-break-inside: avoid;
    border-left: 3px solid #ccc;
  }}
  code {{
    padding: 0.1em 0.3em;
    border-radius: 3px;
  }}
  .page-break {{
    page-break-before: always;
  }}
  </style>
</head>
<body>
{content}
</body>
</html>
"""


def md_to_html(md_text: str) -> str:
  if markdown is None:
    raise RuntimeError("Missing dependency: install 'markdown' (pip install markdown)")
  html = markdown.markdown(
    md_text,
    extensions=[
      "extra",
      "codehilite",
      "tables",
      "toc",
      "sane_lists",
    ],
    output_format="html5",
  )
  return HTML_TEMPLATE.format(content=html)


def html_to_pdf_weasyprint(html: str, output_path: pathlib.Path, base_url: str = ".") -> None:
  if HTML is None:
    raise RuntimeError("WeasyPrint not available")
  HTML(string=html, base_url=base_url).write_pdf(str(output_path))


def html_to_pdf_xhtml2pdf(html: str, output_path: pathlib.Path) -> None:
  if pisa is None:
    raise RuntimeError("xhtml2pdf not available")
  with open(output_path, "wb") as f:
    pisa.CreatePDF(html, dest=f)


def convert_md_to_pdf(input_md: pathlib.Path, output_pdf: pathlib.Path) -> None:
  if _BACKEND is None:
    raise RuntimeError(
      "No PDF backend available. Install one of:\n"
      "  pip install weasyprint markdown\n"
      "or\n"
      "  pip install xhtml2pdf markdown"
    )

  text = input_md.read_text(encoding="utf-8")
  html = md_to_html(text)

  # Use the directory of the input Markdown as the base URL for resolving relative paths
  base_url = str(input_md.parent.resolve())

  if _BACKEND == "weasyprint":
    html_to_pdf_weasyprint(html, output_pdf, base_url=base_url)
  else:
    # xhtml2pdf sometimes prefers a temp file
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as tmp:
      tmp.write(html)
      tmp.flush()
      html_to_pdf_xhtml2pdf(html, output_pdf)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description="Convert Markdown to PDF.")
  parser.add_argument("input", help="Input Markdown file")
  parser.add_argument("output", nargs="?", help="Output PDF file (default: same name with .pdf)")
  return parser.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)

  in_path = pathlib.Path(args.input).expanduser().resolve()
  if not in_path.is_file():
    print(f"Input file not found: {in_path}", file=sys.stderr)
    return 1

  if args.output:
    out_path = pathlib.Path(args.output).expanduser().resolve()
  else:
    out_path = in_path.with_suffix(".pdf")

  try:
    convert_md_to_pdf(in_path, out_path)
  except Exception as e:
    print(f"Conversion failed: {e}", file=sys.stderr)
    return 1

  print(f"Written PDF: {out_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())