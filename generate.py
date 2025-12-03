import os
import shutil
import subprocess
import sys
from typing import Dict, List, Any


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters in text content."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = []
    for ch in text:
        escaped.append(replacements.get(ch, ch))
    return "".join(escaped)


def _escape_url(text: str) -> str:
    """Escape URL for use in \href{url}{text}. 
    URLs should generally work as-is, but we escape special LaTeX chars that might break parsing."""
    # URLs should be URL-encoded, so most special chars are fine
    # Only escape chars that would break LaTeX command structure
    # Note: We avoid escaping braces here since URLs shouldn't have them, but if they do, it breaks the \href{url}{text} structure
    # For now, just escape backslashes and let LaTeX handle the rest
    return text.replace("\\", r"\textbackslash{}")


def _render_contact_line(contact: Dict[str, str]) -> str:
    parts: List[str] = []
    email = contact.get("email")
    if email:
        # Escape email for both URL and display text
        email_escaped = _escape_latex(email)
        parts.append(f"\\href{{mailto:{email_escaped}}}{{{email_escaped}}}")
    portfolio = contact.get("portfolio")
    if portfolio:
        # URLs: escape URL but keep display text simple
        url_escaped = _escape_url(portfolio)
        parts.append(f"\\href{{{url_escaped}}}{{{_escape_latex('portfolio')}}}")
    github = contact.get("github")
    if github:
        url_escaped = _escape_url(github)
        parts.append(f"\\href{{{url_escaped}}}{{{_escape_latex('github')}}}")
    linkedin = contact.get("linkedin")
    if linkedin:
        url_escaped = _escape_url(linkedin)
        parts.append(f"\\href{{{url_escaped}}}{{{_escape_latex('linkedin')}}}")
    return " | ".join(parts) if parts else ""


def _render_skills(skills: Dict[str, List[str]]) -> str:
    if not skills:
        return ""
    lines: List[str] = []
    for category, items in skills.items():
        if not items:
            continue
        lines.append(f"\\textbf{{{_escape_latex(category)}}}: {_escape_latex(', '.join(items))} \\")
    return "\n".join(lines)


def _render_experience(experience_blocks: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for exp in experience_blocks:
        title = _escape_latex(exp.get("title", ""))
        company = _escape_latex(exp.get("company", ""))
        location = _escape_latex(exp.get("location", ""))
        start = _escape_latex(exp.get("start_date", ""))
        end = _escape_latex(exp.get("end_date", ""))
        
        # Format: Title | Company (City, State) \hfill Start -- End \\
        location_part = f"({location})" if location else ""
        header = f"\\textbf{{{title}}} | {company}"
        if location_part:
            header += f" {location_part}"
        if start and end:
            header += f" \\hfill {start} -- {end}"
        elif start:
            header += f" \\hfill {start}"
        header += " \\\\"
        
        bullets = exp.get("bullets", [])
        bullet_lines = [f"  \\item {_escape_latex(b)}" for b in bullets if b]
        block = "\n".join([
            header,
            "\\begin{itemize}",
            *bullet_lines,
            "\\end{itemize}",
            "",
        ])
        blocks.append(block)
    return "\n".join(blocks)


def _render_projects(project_blocks: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for pr in project_blocks:
        name = _escape_latex(pr.get("name", "Project"))
        start_date = _escape_latex(pr.get("start_date", ""))
        end_date = _escape_latex(pr.get("end_date", ""))
        
        # Format: \textbf{Project Name} | \href{link1}{name1} | \href{link2}{name2} \hfill Start -- End \\
        header = f"\\textbf{{{name}}}"
        
        # Support both old "link" format (backward compatibility) and new "links" array format
        link_parts: List[str] = []
        
        # Check for new "links" array format
        links = pr.get("links", [])
        if links:
            for link_obj in links:
                if isinstance(link_obj, dict):
                    link_name = link_obj.get("name", "link")
                    link_url = link_obj.get("url") or link_obj.get("link")
                    if link_url:
                        url_escaped = _escape_url(link_url)
                        link_parts.append(f"\\href{{{url_escaped}}}{{{_escape_latex(link_name)}}}")
        
        # Fall back to old "link" format for backward compatibility
        if not link_parts:
            link = pr.get("link")
            if link:
                url_escaped = _escape_url(link)
                link_parts.append(f"\\href{{{url_escaped}}}{{{_escape_latex('github')}}}")
        
        # Add links to header
        if link_parts:
            header += " | " + " | ".join(link_parts)
        
        # Add timeline on the right if dates are provided
        date_part = ""
        if start_date and end_date:
            date_part = f"{start_date} -- {end_date}"
        elif start_date:
            date_part = start_date
        
        if date_part:
            header += f" \\hfill {date_part}"
        
        header += " \\\\"
        
        bullets = pr.get("bullets", [])
        bullet_lines = [f"  \\item {_escape_latex(b)}" for b in bullets if b]
        block = "\n".join([
            header,
            "\\begin{itemize}",
            *bullet_lines,
            "\\end{itemize}",
            "",
        ])
        blocks.append(block)
    return "\n".join(blocks)


def _render_education(education_blocks: List[Dict[str, Any]]) -> str:
    if not education_blocks:
        return ""
    lines: List[str] = []
    for ed in education_blocks:
        institution = _escape_latex(ed.get("institution", ""))
        degree = _escape_latex(ed.get("degree", ""))
        start_date = _escape_latex(ed.get("start_date", ""))
        end_date = _escape_latex(ed.get("end_date", ""))
        
        # Format: \textbf{Degree} {Institution} \hfill Start -- End \\
        degree_part = f"\\textbf{{{degree}}}" if degree else ""
        institution_part = f"{{{institution}}}" if institution else ""
        date_part = f"{start_date} -- {end_date}" if start_date and end_date else (start_date if start_date else "")
        
        header_parts = []
        if degree_part:
            header_parts.append(degree_part)
        if institution_part:
            header_parts.append(institution_part)
        
        header = " ".join(header_parts)
        if date_part:
            header += f" \\hfill {date_part}"
        header += " \\\\"
        lines.append(header)
        
        # Add highlights as bullet points
        extra = ed.get("highlights", [])
        if extra:
            lines.append("\\begin{itemize}")
            for x in extra:
                lines.append(f"  \\item {_escape_latex(x)}")
            lines.append("\\end{itemize}")
    
    return "\n".join(lines)


def render_resume_latex(template_text: str, payload: Dict[str, Any]) -> str:
    name = payload.get("name", "")
    contact_line = _render_contact_line(payload.get("contact", {}))
    skills_block = _render_skills(payload.get("skills", {}))
    experience_block = _render_experience(payload.get("experience", []))
    projects_block = _render_projects(payload.get("projects", []))
    education_block = _render_education(payload.get("education", []))

    latex = template_text
    latex = latex.replace("%%NAME%%", _escape_latex(name))
    latex = latex.replace("%%CONTACT_LINE%%", contact_line)
    latex = latex.replace("%%SKILLS_BLOCKS%%", skills_block)
    latex = latex.replace("%%EXPERIENCE_BLOCKS%%", experience_block)
    latex = latex.replace("%%PROJECT_BLOCKS%%", projects_block)
    latex = latex.replace("%%EDUCATION_BLOCKS%%", education_block)
    return latex


def write_and_compile_latex(latex_text: str, output_path: str) -> str:
    out_dir = os.path.dirname(os.path.abspath(output_path)) or os.getcwd()
    base_name, ext = os.path.splitext(os.path.basename(output_path))
    if not ext:
        ext = ".pdf"
    pdf_output = os.path.join(out_dir, base_name + ".pdf")
    tex_output = os.path.join(out_dir, base_name + ".tex")

    os.makedirs(out_dir, exist_ok=True)
    
    # Try tectonic first (lightweight, auto-fetches packages)
    tectonic_path = shutil.which("tectonic")
    if tectonic_path:
        # For tectonic (XeLaTeX), XCharter uses fontspec which conflicts with fontenc/inputenc
        # Remove fontenc and inputenc when XCharter is present (regardless of order)
        lines = latex_text.split("\n")
        has_xcharter = any("\\usepackage{XCharter}" in line for line in lines)
        cleaned_lines = []
        
        for line in lines:
            # Remove glyphtounicode (not needed for XeLaTeX)
            if "\\input{glyphtounicode}" in line or "\\pdfgentounicode=1" in line:
                continue
            
            # Skip fontenc and inputenc when XCharter is present (XCharter uses fontspec)
            if has_xcharter and ("\\usepackage[T1]{fontenc}" in line or "\\usepackage[utf8]{inputenc}" in line):
                continue
            
            cleaned_lines.append(line)
        
        cleaned_latex = "\n".join(cleaned_lines)
        with open(tex_output, "w", encoding="utf-8") as f:
            f.write(cleaned_latex)
        
        try:
            result = subprocess.run(
                [tectonic_path, tex_output],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=out_dir,
            )
            # Tectonic outputs to the same directory with .pdf extension
            tectonic_pdf = os.path.join(out_dir, base_name + ".pdf")
            if os.path.exists(tectonic_pdf):
                # Rename to match expected output path if different
                if tectonic_pdf != pdf_output:
                    os.rename(tectonic_pdf, pdf_output)
                return pdf_output
            else:
                print(f"Warning: PDF generation failed. LaTeX source saved to: {tex_output}", file=sys.stderr)
                return tex_output
        except subprocess.CalledProcessError as e:
            print(f"Error: tectonic compilation failed. LaTeX source saved to: {tex_output}", file=sys.stderr)
            if e.stderr:
                error_msg = e.stderr.decode('utf-8', errors='ignore')[:500]
                print(f"Error details: {error_msg}", file=sys.stderr)
            return tex_output
        except Exception as e:
            print(f"Error: {str(e)}. LaTeX source saved to: {tex_output}", file=sys.stderr)
            return tex_output
    else:
        # For pdflatex, write the original latex
        with open(tex_output, "w", encoding="utf-8") as f:
            f.write(latex_text)

    # Fall back to pdflatex
    pdflatex_path = shutil.which("pdflatex")
    if not pdflatex_path:
        print("Warning: No LaTeX engine found (tectonic or pdflatex).", file=sys.stderr)
        print(f"LaTeX source saved to: {tex_output}", file=sys.stderr)
        print("To install: brew install tectonic (lightweight) or brew install --cask mactex (full)", file=sys.stderr)
        return tex_output

    try:
        # Run pdflatex twice for proper TOC/refs handling
        for _ in range(2):
            result = subprocess.run(
                [pdflatex_path, "-interaction=nonstop", f"-output-directory={out_dir}", tex_output],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=out_dir,
            )
        if os.path.exists(pdf_output):
            return pdf_output
        else:
            print(f"Warning: PDF generation failed. LaTeX source saved to: {tex_output}", file=sys.stderr)
            return tex_output
    except subprocess.CalledProcessError as e:
        print(f"Error: pdflatex compilation failed. LaTeX source saved to: {tex_output}", file=sys.stderr)
        if e.stderr:
            print(f"Error details: {e.stderr.decode('utf-8', errors='ignore')[:500]}", file=sys.stderr)
        return tex_output
    except Exception as e:
        print(f"Error: {str(e)}. LaTeX source saved to: {tex_output}", file=sys.stderr)
        return tex_output



