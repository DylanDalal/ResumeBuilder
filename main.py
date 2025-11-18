import argparse
import json
import os
import re
from typing import Any, Dict, List

from generate import render_resume_latex, write_and_compile_latex


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def call_openai_selection(api_key: str, job_description: str, jobs: List[Dict[str, Any]], projects: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        # Prefer new OpenAI SDK if available
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=api_key)
            system = (
                "You are an assistant that selects resume content strictly from provided data. "
                "Return JSON with keys: selected_jobs (list of {id, bullet_indices}), "
                "selected_projects (list of {id, bullet_indices}), and optional skills (mapping). "
                "Only use bullets by index from the given entries. "
                "IMPORTANT: Fill the resume. Select 3-5 most relevant jobs and 2-3 most relevant projects. "
                "For each selected job/project, include 2-4 bullets. Prioritize relevance but ensure the resume is well-populated. "
                "If needed, include less-relevant experience to fill the page. Total bullets should be around 35 to create a full resume. "
                "NOTE: Some bullets may have a 'group' field. If multiple bullets share the same group, only one will be included in the final resume. "
                "You can still select multiple bullets from the same group, but the system will automatically filter to keep only one."
            )
            user = json.dumps({
                "job_description": job_description,
                "jobs": jobs,
                "projects": projects,
            })
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content or "{}"
            return json.loads(content)
        except ImportError:
            # Fallback to legacy SDK
            import openai  # type: ignore

            openai.api_key = api_key
            system = (
                "You are an assistant that selects resume content strictly from provided data. "
                "Return JSON with keys: selected_jobs (list of {id, bullet_indices}), "
                "selected_projects (list of {id, bullet_indices}), and optional skills (mapping). "
                "Only use bullets by index from the given entries. "
                "IMPORTANT: Fill the resume generously. Select 3-5 most relevant jobs and 2-3 most relevant projects. "
                "For each selected job/project, include 2-4 bullets. Prioritize relevance but ensure the resume is well-populated. "
                "If needed, include less-relevant experience to fill the page. Total bullets should be 15-25 to create a full resume. "
                "NOTE: Some bullets may have a 'group' field. If multiple bullets share the same group, only one will be included in the final resume. "
                "You can still select multiple bullets from the same group, but the system will automatically filter to keep only one."
            )
            user = json.dumps({
                "job_description": job_description,
                "jobs": jobs,
                "projects": projects,
            })
            completion = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
            )
            content = completion["choices"][0]["message"]["content"] or "{}"
            return json.loads(content)
    except Exception:
        # If API fails, return empty selection to avoid crashing
        return {"selected_jobs": [], "selected_projects": [], "skills": {}}


def _normalize_bullet(bullet: Any) -> Dict[str, Any]:
    """Normalize a bullet to a dict with 'text' and optional 'group'.
    Supports backward compatibility: strings become {'text': string}.
    """
    if isinstance(bullet, str):
        return {"text": bullet}
    elif isinstance(bullet, dict):
        return {"text": bullet.get("text", ""), "group": bullet.get("group")}
    else:
        return {"text": str(bullet)}


def _filter_conflicting_bullets(bullets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter bullets to ensure only one bullet per group is included.
    If multiple bullets share the same group, keep only the first one.
    Bullets without a group are always included.
    """
    seen_groups: set[str] = set()
    filtered: List[Dict[str, Any]] = []
    
    for bullet in bullets:
        group = bullet.get("group")
        if group:
            if group not in seen_groups:
                seen_groups.add(group)
                filtered.append(bullet)
            # Skip if we've already seen this group
        else:
            # Bullets without a group are always included
            filtered.append(bullet)
    
    return filtered


def _parse_date_for_sorting(date_str: str) -> tuple:
    """Parse date string like 'Jan 2024' or 'Present' into (year, month) tuple for sorting.
    Returns (9999, 12) for 'Present' to make it sort first (most recent).
    Returns (0, 0) for empty/invalid dates to sort them last.
    """
    if not date_str or date_str.strip() == "":
        return (0, 0)
    if date_str.lower() in ["present", "current", "now"]:
        return (9999, 12)  # Make "Present" sort as most recent
    
    # Parse formats like "Jan 2024", "January 2024", "2024", "Jan 2024 -- Present"
    months = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
        "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6,
        "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "september": 9,
        "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12
    }
    
    # Try to extract year and month
    parts = date_str.lower().split()
    year = 0
    month = 0
    
    for part in parts:
        # Check for year (4 digits)
        if re.match(r'^\d{4}$', part):
            year = int(part)
        # Check for month name
        elif part in months:
            month = months[part]
    
    # If we found a year but no month, default to December (end of year)
    if year > 0 and month == 0:
        month = 12
    
    return (year, month)


def build_payload(
    name: str,
    contact: Dict[str, str],
    jobs: List[Dict[str, Any]],
    projects: List[Dict[str, Any]],
    selection: Dict[str, Any],
    education: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    id_to_job = {j["id"]: j for j in jobs}
    id_to_project = {p["id"]: p for p in projects}

    # Collect all bullets with metadata about their source
    all_bullets_with_metadata: List[Dict[str, Any]] = []
    
    # Collect bullets from jobs
    for item in selection.get("selected_jobs", []):
        job = id_to_job.get(item.get("id"))
        if not job:
            continue
        indices = item.get("bullet_indices", [])
        raw_bullets = [job.get("bullets", [])[i] for i in indices if 0 <= i < len(job.get("bullets", []))]
        for bullet in raw_bullets:
            normalized = _normalize_bullet(bullet)
            all_bullets_with_metadata.append({
                **normalized,
                "source_type": "job",
                "source_id": item.get("id"),
                "job_data": job,
            })
    
    # Collect bullets from projects
    for item in selection.get("selected_projects", []):
        pr = id_to_project.get(item.get("id"))
        if not pr:
            continue
        indices = item.get("bullet_indices", [])
        raw_bullets = [pr.get("bullets", [])[i] for i in indices if 0 <= i < len(pr.get("bullets", []))]
        for bullet in raw_bullets:
            normalized = _normalize_bullet(bullet)
            all_bullets_with_metadata.append({
                **normalized,
                "source_type": "project",
                "source_id": item.get("id"),
                "project_data": pr,
            })
    
    # Filter conflicting bullets globally (across all jobs and projects)
    filtered_bullets = _filter_conflicting_bullets(all_bullets_with_metadata)
    
    # Reconstruct jobs and projects from filtered bullets
    job_bullets_map: Dict[str, List[str]] = {}  # job_id -> list of bullet texts
    project_bullets_map: Dict[str, List[str]] = {}  # project_id -> list of bullet texts
    
    for bullet in filtered_bullets:
        source_id = bullet.get("source_id")
        text = bullet.get("text", "")
        if bullet.get("source_type") == "job" and source_id:
            if source_id not in job_bullets_map:
                job_bullets_map[source_id] = []
            job_bullets_map[source_id].append(text)
        elif bullet.get("source_type") == "project" and source_id:
            if source_id not in project_bullets_map:
                project_bullets_map[source_id] = []
            project_bullets_map[source_id].append(text)
    
    # Build jobs payload
    selected_jobs_payload: List[Dict[str, Any]] = []
    for item in selection.get("selected_jobs", []):
        job = id_to_job.get(item.get("id"))
        if not job:
            continue
        bullets = job_bullets_map.get(item.get("id"), [])
        if bullets:  # Only include jobs that have bullets after filtering
            selected_jobs_payload.append({
                "title": job.get("title", ""),
                "company": job.get("company", ""),
                "location": job.get("location", ""),
                "start_date": job.get("start_date", ""),
                "end_date": job.get("end_date", "Present"),
                "bullets": bullets,
            })
    
    # Sort jobs by most recent first (by end_date, then start_date)
    selected_jobs_payload.sort(
        key=lambda x: (
            _parse_date_for_sorting(x.get("end_date", "")),
            _parse_date_for_sorting(x.get("start_date", ""))
        ),
        reverse=True
    )

    # Build projects payload
    selected_projects_payload: List[Dict[str, Any]] = []
    for item in selection.get("selected_projects", []):
        pr = id_to_project.get(item.get("id"))
        if not pr:
            continue
        bullets = project_bullets_map.get(item.get("id"), [])
        if bullets:  # Only include projects that have bullets after filtering
            selected_projects_payload.append({
                "name": pr.get("name", ""),
                "link": pr.get("link"),
                "start_date": pr.get("start_date", ""),
                "end_date": pr.get("end_date", ""),
                "bullets": bullets,
            })
    
    # Sort projects by most recent first (by end_date if available, else start_date)
    selected_projects_payload.sort(
        key=lambda x: (
            _parse_date_for_sorting(x.get("end_date", "") or x.get("start_date", "")),
            _parse_date_for_sorting(x.get("start_date", ""))
        ),
        reverse=True
    )

    payload: Dict[str, Any] = {
        "name": name,
        "contact": contact,
        "skills": selection.get("skills", {}),
        "experience": selected_jobs_payload,
        "projects": selected_projects_payload,
        "education": education or [],
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume Builder")
    parser.add_argument("--output", required=True, help="Output path for PDF or TEX")
    parser.add_argument("--input", required=True, help="Path to txt file with job description")
    parser.add_argument("--key", required=True, help="OpenAI API key")
    parser.add_argument("--name", required=False, default=None, help="Your name (overrides personal.json)")
    parser.add_argument("--contact", required=False, default=None, help="JSON string for contact info (overrides personal.json)")
    parser.add_argument("--education", required=False, default=None, help="Path to education JSON file (overrides personal.json)")
    parser.add_argument("--template", required=False, default="latex.txt", help="Path to LaTeX template")
    parser.add_argument("--data_dir", required=False, default="data", help="Directory containing jobs.json, projects.json, and personal.json")
    args = parser.parse_args()

    job_description = read_text(args.input)
    jobs = load_json(os.path.join(args.data_dir, "jobs.json"))
    projects = load_json(os.path.join(args.data_dir, "projects.json"))
    
    # Load personal info from data/personal.json if it exists
    personal_file = os.path.join(args.data_dir, "personal.json")
    personal_data: Dict[str, Any] = {}
    if os.path.exists(personal_file):
        personal_data = load_json(personal_file)
    
    # Use command line args if provided, otherwise use personal.json
    name = args.name if args.name else personal_data.get("name", "John Doe")
    
    contact: Dict[str, str] = {}
    if args.contact:
        try:
            contact = json.loads(args.contact)
            if not isinstance(contact, dict):
                contact = {}
        except Exception:
            contact = {}
    else:
        contact = personal_data.get("contact", {})
    
    education: List[Dict[str, Any]] = []
    if args.education:
        if os.path.exists(args.education):
            education = load_json(args.education)
    else:
        education = personal_data.get("education", [])

    selection = call_openai_selection(args.key, job_description, jobs, projects)

    payload = build_payload(
        name=name,
        contact=contact,
        jobs=jobs,
        projects=projects,
        selection=selection,
        education=education,
    )

    template_text = read_text(args.template)
    latex_text = render_resume_latex(template_text, payload)
    out = write_and_compile_latex(latex_text, args.output)
    print(out)


if __name__ == "__main__":
    main()



