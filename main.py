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


def call_openai_experiences(api_key: str, job_description: str, jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Call OpenAI to select 4-6 most relevant experiences with top 4 bullets each."""
    try:
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=api_key)
            
            system = (
                "You are an assistant that selects relevant resume experiences from provided data. "
                "Return JSON with key: selected_jobs (list of {id, bullet_indices}). "
                "Select 4-6 most relevant experiences to the job description. "
                "For each relevant job, rank the top 4 bullets in terms of relevance to the role, with the most relevant bullets at the top. "
                "The bullet_indices array should contain the indices in RANKED ORDER (most relevant first), NOT in original order. "
                "For example, if bullets [0, 1, 2, 3, 4] exist and bullets 2 and 4 are most relevant, return [2, 4, 0, 1] not [0, 1, 2, 3]. "
                "Generously keyword match these bullets - they should be clearly relevant to the role. "
                "If there's nothing relevant for a job, don't include it. "
                "Only use bullets by index from the given entries."
            )
            
            user_data = {
                "job_description": job_description,
                "jobs": jobs,
            }
            user = json.dumps(user_data)
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
            import openai  # type: ignore
            openai.api_key = api_key
            
            system = (
                "You are an assistant that selects relevant resume experiences from provided data. "
                "Return JSON with key: selected_jobs (list of {id, bullet_indices}). "
                "Select 4-6 most relevant experiences to the job description. "
                "For each relevant job, rank the top 4 bullets in terms of relevance to the role, with the most relevant bullets at the top. "
                "The bullet_indices array should contain the indices in RANKED ORDER (most relevant first), NOT in original order. "
                "For example, if bullets [0, 1, 2, 3, 4] exist and bullets 2 and 4 are most relevant, return [2, 4, 0, 1] not [0, 1, 2, 3]. "
                "Generously keyword match these bullets - they should be clearly relevant to the role. "
                "If there's nothing relevant for a job, don't include it. "
                "Only use bullets by index from the given entries."
            )
            
            user_data = {
                "job_description": job_description,
                "jobs": jobs,
            }
            user = json.dumps(user_data)
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
        return {"selected_jobs": []}


def call_openai_rank_bullets(api_key: str, job_description: str, job: Dict[str, Any], max_bullets: int = 4) -> List[int]:
    """Call OpenAI to rank bullets for a single job. Returns ranked bullet indices."""
    try:
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=api_key)
            
            system = (
                "You are an assistant that ranks resume bullets by relevance to a job description. "
                "Return JSON with key: bullet_indices (array of integers). "
                f"Rank the top {max_bullets} bullets from the provided job in terms of relevance to the role, with the most relevant bullets at the top. "
                "The bullet_indices array should contain the indices in RANKED ORDER (most relevant first), NOT in original order. "
                "For example, if bullets [0, 1, 2, 3, 4] exist and bullets 2 and 4 are most relevant, return [2, 4, 0, 1] not [0, 1, 2, 3]. "
                "Generously keyword match these bullets - they should be clearly relevant to the role. "
                "Only use bullets by index from the given entries."
            )
            
            user_data = {
                "job_description": job_description,
                "job": job,
            }
            user = json.dumps(user_data)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content or "{}"
            result = json.loads(content)
            return result.get("bullet_indices", [])
        except ImportError:
            import openai  # type: ignore
            openai.api_key = api_key
            
            system = (
                "You are an assistant that ranks resume bullets by relevance to a job description. "
                "Return JSON with key: bullet_indices (array of integers). "
                f"Rank the top {max_bullets} bullets from the provided job in terms of relevance to the role, with the most relevant bullets at the top. "
                "The bullet_indices array should contain the indices in RANKED ORDER (most relevant first), NOT in original order. "
                "For example, if bullets [0, 1, 2, 3, 4] exist and bullets 2 and 4 are most relevant, return [2, 4, 0, 1] not [0, 1, 2, 3]. "
                "Generously keyword match these bullets - they should be clearly relevant to the role. "
                "Only use bullets by index from the given entries."
            )
            
            user_data = {
                "job_description": job_description,
                "job": job,
            }
            user = json.dumps(user_data)
            completion = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
            )
            content = completion["choices"][0]["message"]["content"] or "{}"
            result = json.loads(content)
            return result.get("bullet_indices", [])
    except Exception:
        # Fallback to sequential if API fails
        bullets = job.get("bullets", [])
        if bullets:
            num_bullets = min(max_bullets, len(bullets))
            return list(range(num_bullets))
        return []


def call_openai_projects(api_key: str, job_description: str, projects: List[Dict[str, Any]], num_projects: int) -> Dict[str, Any]:
    """Call OpenAI to select N most relevant projects with ranked bullets."""
    try:
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=api_key)
            
            system = (
                f"You are an assistant that selects relevant resume projects from provided data. "
                f"Return JSON with key: selected_projects (list of {{id, bullet_indices}}). "
                f"Select the {num_projects} most relevant projects to the job description. "
                f"For each relevant project, rank the bullets in terms of relevance to the role, with the most relevant bullets at the top. "
                f"Completed projects should be slightly higher priority than ongoing projects if they're relevant. "
                f"Generously keyword match these bullets - they should be clearly relevant to the role. "
                f"If there's nothing relevant for a project, don't include it. "
                f"Only use bullets by index from the given entries."
            )
            
            user_data = {
                "job_description": job_description,
                "projects": projects,
            }
            user = json.dumps(user_data)
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
            import openai  # type: ignore
            openai.api_key = api_key
            
            system = (
                f"You are an assistant that selects relevant resume projects from provided data. "
                f"Return JSON with key: selected_projects (list of {{id, bullet_indices}}). "
                f"Select the {num_projects} most relevant projects to the job description. "
                f"For each relevant project, rank the bullets in terms of relevance to the role, with the most relevant bullets at the top. "
                f"Completed projects should be slightly higher priority than ongoing projects if they're relevant. "
                f"Generously keyword match these bullets - they should be clearly relevant to the role. "
                f"If there's nothing relevant for a project, don't include it. "
                f"Only use bullets by index from the given entries."
            )
            
            user_data = {
                "job_description": job_description,
                "projects": projects,
            }
            user = json.dumps(user_data)
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
        return {"selected_projects": []}


# Old function removed - now using call_openai_experiences and call_openai_projects directly


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


DEFAULT_CONTACT: Dict[str, str] = {
    "email": "dylanmax@gmail.com",
    "github": "https://github.com/DylanDalal",
    "portfolio": "https://www.dylandalal.com",
    "linkedin": "https://www.linkedin.com/in/dylandalal",
}
DEFAULT_FALLBACK_JOBS = 4
DEFAULT_FALLBACK_PROJECTS = 3
DEFAULT_FALLBACK_JOB_BULLETS = 4
DEFAULT_FALLBACK_PROJECT_BULLETS = 2


def _build_default_selection(
    jobs: List[Dict[str, Any]],
    projects: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Construct a reasonable resume selection entirely locally.

    Used when the job description is sparse or the model returns no picks,
    ensuring we can still render a strong generalized resume.
    """
    def _sorted_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort entries by priority (if present, higher first), then by date (most recent first)."""
        return sorted(
            entries,
            key=lambda x: (
                -x.get("priority", 0),  # Negative for descending order (higher priority first)
                _parse_date_for_sorting(x.get("end_date", "")),
                _parse_date_for_sorting(x.get("start_date", "")),
            ),
            reverse=True,
        )

    selection: Dict[str, List[Dict[str, Any]]] = {
        "selected_jobs": [],
        "selected_projects": [],
    }

    for job in _sorted_entries(jobs):
        if len(selection["selected_jobs"]) >= DEFAULT_FALLBACK_JOBS:
            break
        bullets = job.get("bullets", [])
        if not bullets:
            continue
        bullet_indices = list(range(min(len(bullets), DEFAULT_FALLBACK_JOB_BULLETS)))
        selection["selected_jobs"].append({"id": job.get("id"), "bullet_indices": bullet_indices})

    for project in _sorted_entries(projects):
        if len(selection["selected_projects"]) >= DEFAULT_FALLBACK_PROJECTS:
            break
        bullets = project.get("bullets", [])
        if not bullets:
            continue
        bullet_indices = list(range(min(len(bullets), DEFAULT_FALLBACK_PROJECT_BULLETS)))
        selection["selected_projects"].append({"id": project.get("id"), "bullet_indices": bullet_indices})

    return selection


def build_payload(
    name: str,
    contact: Dict[str, str],
    jobs: List[Dict[str, Any]],
    projects: List[Dict[str, Any]],
    selection: Dict[str, Any],
    education: List[Dict[str, Any]] | None = None,
    force_additional_experience: bool = False,
    additional_experience_jobs: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    id_to_job = {j["id"]: j for j in jobs}
    id_to_project = {p["id"]: p for p in projects}
    id_to_additional_job = {j["id"]: j for j in (additional_experience_jobs or [])}

    # Collect all bullets with metadata about their source
    all_bullets_with_metadata: List[Dict[str, Any]] = []
    
    # Collect bullets from jobs
    for item in selection.get("selected_jobs", []):
        job = id_to_job.get(item.get("id"))
        if not job:
            continue
        indices = item.get("bullet_indices", [])
        raw_bullets = [job.get("bullets", [])[i] for i in indices if 0 <= i < len(job.get("bullets", []))]
        for rank, bullet in enumerate(raw_bullets):
            normalized = _normalize_bullet(bullet)
            all_bullets_with_metadata.append({
                **normalized,
                "source_type": "job",
                "source_id": item.get("id"),
                "job_data": job,
                "original_rank": rank,  # Preserve ranked order
            })
    
    # Collect bullets from projects
    for item in selection.get("selected_projects", []):
        pr = id_to_project.get(item.get("id"))
        if not pr:
            continue
        indices = item.get("bullet_indices", [])
        raw_bullets = [pr.get("bullets", [])[i] for i in indices if 0 <= i < len(pr.get("bullets", []))]
        for rank, bullet in enumerate(raw_bullets):
            normalized = _normalize_bullet(bullet)
            all_bullets_with_metadata.append({
                **normalized,
                "source_type": "project",
                "source_id": item.get("id"),
                "project_data": pr,
                "original_rank": rank,  # Preserve ranked order
            })
    
    # Filter conflicting bullets globally (across all jobs and projects)
    filtered_bullets = _filter_conflicting_bullets(all_bullets_with_metadata)
    
    # Reconstruct jobs and projects from filtered bullets, preserving ranked order
    job_bullets_map: Dict[str, List[tuple]] = {}  # job_id -> list of (rank, text) tuples
    project_bullets_map: Dict[str, List[tuple]] = {}  # project_id -> list of (rank, text) tuples
    
    for bullet in filtered_bullets:
        source_id = bullet.get("source_id")
        text = bullet.get("text", "")
        original_rank = bullet.get("original_rank", 999)  # Default to end if missing
        if bullet.get("source_type") == "job" and source_id:
            if source_id not in job_bullets_map:
                job_bullets_map[source_id] = []
            job_bullets_map[source_id].append((original_rank, text))
        elif bullet.get("source_type") == "project" and source_id:
            if source_id not in project_bullets_map:
                project_bullets_map[source_id] = []
            project_bullets_map[source_id].append((original_rank, text))
    
    # Sort bullets by original rank to preserve LLM ranking
    for job_id in job_bullets_map:
        job_bullets_map[job_id].sort(key=lambda x: x[0])
        job_bullets_map[job_id] = [text for _, text in job_bullets_map[job_id]]
    
    for project_id in project_bullets_map:
        project_bullets_map[project_id].sort(key=lambda x: x[0])
        project_bullets_map[project_id] = [text for _, text in project_bullets_map[project_id]]
    
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
            project_payload: Dict[str, Any] = {
                "name": pr.get("name", ""),
                "start_date": pr.get("start_date", ""),
                "end_date": pr.get("end_date", ""),
                "bullets": bullets,
            }
            # Support both old "link" format and new "links" array format
            if "links" in pr:
                project_payload["links"] = pr.get("links", [])
            elif "link" in pr:
                project_payload["link"] = pr.get("link")
            selected_projects_payload.append(project_payload)
    
    # Sort projects by most recent first (by end_date if available, else start_date)
    selected_projects_payload.sort(
        key=lambda x: (
            _parse_date_for_sorting(x.get("end_date", "") or x.get("start_date", "")),
            _parse_date_for_sorting(x.get("start_date", ""))
        ),
        reverse=True
    )

    # Build additional_experience payload (similar to jobs)
    additional_experience_payload: List[Dict[str, Any]] = []
    if selection.get("additional_experience") or force_additional_experience:
        # Collect bullets from additional_experience
        additional_bullets_with_metadata: List[Dict[str, Any]] = []
        for item in selection.get("additional_experience", []):
            # Try to find job in additional_experience_jobs first, then fall back to regular jobs
            job = id_to_additional_job.get(item.get("id")) or id_to_job.get(item.get("id"))
            if not job:
                continue
            indices = item.get("bullet_indices", [])
            raw_bullets = [job.get("bullets", [])[i] for i in indices if 0 <= i < len(job.get("bullets", []))]
            for rank, bullet in enumerate(raw_bullets):
                normalized = _normalize_bullet(bullet)
                additional_bullets_with_metadata.append({
                    **normalized,
                    "source_type": "additional_job",
                    "source_id": item.get("id"),
                    "job_data": job,
                    "original_rank": rank,  # Preserve ranked order
                })
        
        # Filter conflicting bullets for additional experience
        filtered_additional_bullets = _filter_conflicting_bullets(additional_bullets_with_metadata)
        
        # Reconstruct additional experience from filtered bullets, preserving ranked order
        additional_job_bullets_map: Dict[str, List[tuple]] = {}
        for bullet in filtered_additional_bullets:
            source_id = bullet.get("source_id")
            text = bullet.get("text", "")
            original_rank = bullet.get("original_rank", 999)  # Default to end if missing
            if source_id:
                if source_id not in additional_job_bullets_map:
                    additional_job_bullets_map[source_id] = []
                additional_job_bullets_map[source_id].append((original_rank, text))
        
        # Sort bullets by original rank to preserve LLM ranking
        for job_id in additional_job_bullets_map:
            additional_job_bullets_map[job_id].sort(key=lambda x: x[0])
            additional_job_bullets_map[job_id] = [text for _, text in additional_job_bullets_map[job_id]]
        
        # Build additional experience payload
        for item in selection.get("additional_experience", []):
            # Try to find job in additional_experience_jobs first, then fall back to regular jobs
            job = id_to_additional_job.get(item.get("id")) or id_to_job.get(item.get("id"))
            if not job:
                continue
            bullets = additional_job_bullets_map.get(item.get("id"), [])
            if bullets:
                additional_experience_payload.append({
                    "title": job.get("title", ""),
                    "company": job.get("company", ""),
                    "location": job.get("location", ""),
                    "start_date": job.get("start_date", ""),
                    "end_date": job.get("end_date", "Present"),
                    "bullets": bullets,
                })
        
        # Sort additional experience by most recent first
        additional_experience_payload.sort(
            key=lambda x: (
                _parse_date_for_sorting(x.get("end_date", "")),
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
        "additional_experience": additional_experience_payload,
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
    parser.add_argument("--required-jobs", required=False, default=None, help="Comma-separated list of job IDs that MUST be included in the resume")
    parser.add_argument("--force-additional-experience", action="store_true", help="Force the Additional Experience section to appear")
    parser.add_argument("--additional_experience_note", action="store_true", help="Append additional experience to jobs before passing to LLM")
    parser.add_argument("--additional-experience-jobs", required=False, default=None, help="Comma-separated list of job IDs (from jobs.json or additional.json) to include in Additional Experience section")
    args = parser.parse_args()

    job_description = read_text(args.input)
    jobs = load_json(os.path.join(args.data_dir, "jobs.json"))
    projects = load_json(os.path.join(args.data_dir, "projects.json"))
    
    # Load additional experience jobs from additional.json if it exists
    additional_experience_file = os.path.join(args.data_dir, "additional.json")
    additional_experience_jobs: List[Dict[str, Any]] | None = None
    if os.path.exists(additional_experience_file):
        additional_experience_jobs = load_json(additional_experience_file)
    
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

    merged_contact: Dict[str, str] = {**DEFAULT_CONTACT, **{k: v for k, v in contact.items() if v}}
    
    education: List[Dict[str, Any]] = []
    if args.education:
        if os.path.exists(args.education):
            education = load_json(args.education)
    else:
        education = personal_data.get("education", [])

    # Parse required job IDs from command line argument
    required_job_ids: List[str] | None = None
    if args.required_jobs:
        required_job_ids = [job_id.strip() for job_id in args.required_jobs.split(",") if job_id.strip()]

    # Parse additional experience job IDs from command line argument
    manual_additional_experience_ids: List[str] | None = None
    if args.additional_experience_jobs:
        manual_additional_experience_ids = [job_id.strip() for job_id in args.additional_experience_jobs.split(",") if job_id.strip()]

    # Filter jobs by required_job_ids if provided
    jobs_to_send = jobs
    if required_job_ids:
        # Include jobs from both regular jobs and additional_experience_jobs
        jobs_to_send = [j for j in jobs if j.get("id") in required_job_ids]
        # Also include required jobs from additional_experience_jobs
        if additional_experience_jobs:
            additional_required = [j for j in additional_experience_jobs if j.get("id") in required_job_ids]
            jobs_to_send = jobs_to_send + additional_required
    
    # Append additional experience to jobs if --additional_experience_note is provided
    if args.additional_experience_note and additional_experience_jobs:
        jobs_to_send = jobs_to_send + additional_experience_jobs
    
    # Also include manually specified additional experience jobs so LLM can see them
    if manual_additional_experience_ids:
        id_to_additional_job = {j["id"]: j for j in (additional_experience_jobs or [])}
        jobs_to_send_ids = {j.get("id") for j in jobs_to_send}
        
        for manual_id in manual_additional_experience_ids:
            if manual_id not in jobs_to_send_ids:
                # Find in additional_experience_jobs (they won't be in regular jobs)
                job = id_to_additional_job.get(manual_id)
                if job:
                    jobs_to_send.append(job)
    
    # Call LLM to select experiences
    experiences_result = call_openai_experiences(args.key, job_description, jobs_to_send) or {}
    selected_jobs = experiences_result.get("selected_jobs", [])
    
    # Debug: Print what ChatGPT returned for jobs
    print("\n=== ChatGPT returned the following jobs ===")
    id_to_job_debug = {j["id"]: j for j in jobs}
    id_to_additional_job_debug = {j["id"]: j for j in (additional_experience_jobs or [])}
    for job_selection in selected_jobs:
        job_id = job_selection.get("id")
        job_data = id_to_additional_job_debug.get(job_id) or id_to_job_debug.get(job_id)
        if job_data:
            print(f"  - {job_data.get('company', 'Unknown')} ({job_id})")
        else:
            print(f"  - Unknown job ({job_id})")
    print("==========================================\n")
    
    # Separate additional experience from regular jobs if it was included
    # We'll identify additional experience by checking if the job ID is in additional_experience_jobs
    additional_experience_ids = {j.get("id") for j in (additional_experience_jobs or [])}
    # Also include manually specified additional experience IDs
    if manual_additional_experience_ids:
        additional_experience_ids.update(manual_additional_experience_ids)
    
    regular_jobs = []
    additional_experience_list = []
    
    for job_selection in selected_jobs:
        job_id = job_selection.get("id")
        if job_id in additional_experience_ids:
            additional_experience_list.append(job_selection)
        else:
            regular_jobs.append(job_selection)
    
    # Debug: Print separated jobs
    print("\n=== After separation ===")
    print("Regular jobs:")
    for job_selection in regular_jobs:
        job_id = job_selection.get("id")
        job_data = id_to_job_debug.get(job_id)
        if job_data:
            print(f"  - {job_data.get('company', 'Unknown')} ({job_id})")
    print("Additional experience:")
    for job_selection in additional_experience_list:
        job_id = job_selection.get("id")
        job_data = id_to_additional_job_debug.get(job_id) or id_to_job_debug.get(job_id)
        if job_data:
            print(f"  - {job_data.get('company', 'Unknown')} ({job_id})")
    print("=======================\n")
    
    # If manually specified additional experience jobs weren't selected by LLM, add them
    if manual_additional_experience_ids:
        id_to_additional_job = {j["id"]: j for j in (additional_experience_jobs or [])}
        selected_job_ids = {job.get("id") for job in selected_jobs}
        
        for manual_id in manual_additional_experience_ids:
            if manual_id not in selected_job_ids:
                # Find in additional_experience_jobs (they won't be in regular jobs)
                job = id_to_additional_job.get(manual_id)
                if job:
                    bullets = job.get("bullets", [])
                    if bullets:
                        # Select top 4 bullets (or all if less than 4)
                        num_bullets = min(4, len(bullets))
                        bullet_indices = list(range(num_bullets))
                        additional_experience_list.append({
                            "id": manual_id,
                            "bullet_indices": bullet_indices
                        })
    
    # Ensure all required jobs are included
    if required_job_ids:
        id_to_job = {j["id"]: j for j in jobs}
        id_to_additional_job = {j["id"]: j for j in (additional_experience_jobs or [])}
        selected_job_ids = {job.get("id") for job in regular_jobs + additional_experience_list}
        
        for required_id in required_job_ids:
            if required_id not in selected_job_ids:
                # Check if it's an additional experience job or regular job
                if required_id in additional_experience_ids:
                    # It's an additional experience job
                    job = id_to_additional_job.get(required_id)
                    if job:
                        bullets = job.get("bullets", [])
                        if bullets:
                            # Rank bullets using LLM
                            bullet_indices = call_openai_rank_bullets(args.key, job_description, job, max_bullets=4)
                            if not bullet_indices:
                                # Fallback to sequential if ranking fails
                                num_bullets = min(4, len(bullets))
                                bullet_indices = list(range(num_bullets))
                            additional_experience_list.append({
                                "id": required_id,
                                "bullet_indices": bullet_indices
                            })
                else:
                    # It's a regular job
                    job = id_to_job.get(required_id)
                    if job:
                        bullets = job.get("bullets", [])
                        if bullets:
                            # Rank bullets using LLM
                            bullet_indices = call_openai_rank_bullets(args.key, job_description, job, max_bullets=4)
                            if not bullet_indices:
                                # Fallback to sequential if ranking fails
                                num_bullets = min(4, len(bullets))
                                bullet_indices = list(range(num_bullets))
                            regular_jobs.append({
                                "id": required_id,
                                "bullet_indices": bullet_indices
                            })
    
    # Determine number of experiences (regular + additional)
    num_experiences = len(regular_jobs) + len(additional_experience_list)
    
    # Ensure we have at least 4 experiences
    if num_experiences < 4:
        # Add more jobs from the default selection to reach at least 4
        id_to_job = {j["id"]: j for j in jobs}
        id_to_additional_job = {j["id"]: j for j in (additional_experience_jobs or [])}
        selected_job_ids = {job.get("id") for job in regular_jobs + additional_experience_list}
        
        # Get remaining jobs from both jobs.json and additional.json
        remaining_jobs = [j for j in jobs if j.get("id") not in selected_job_ids]
        if additional_experience_jobs:
            remaining_jobs.extend([j for j in additional_experience_jobs if j.get("id") not in selected_job_ids])
        
        # Sort by priority and date
        remaining_jobs = sorted(
            remaining_jobs,
            key=lambda x: (
                -x.get("priority", 0),
                _parse_date_for_sorting(x.get("end_date", "")),
                _parse_date_for_sorting(x.get("start_date", "")),
            ),
            reverse=True,
        )
        
        # Add jobs until we have at least 4 experiences
        needed = 4 - num_experiences
        for job in remaining_jobs[:needed]:
            bullets = job.get("bullets", [])
            if bullets:
                num_bullets = min(4, len(bullets))
                bullet_indices = list(range(num_bullets))
                # Check if this job should go in additional experience or regular
                job_id = job.get("id")
                if job_id in additional_experience_ids:
                    additional_experience_list.append({
                        "id": job_id,
                        "bullet_indices": bullet_indices
                    })
                else:
                    regular_jobs.append({
                        "id": job_id,
                        "bullet_indices": bullet_indices
                    })
        
        num_experiences = len(regular_jobs) + len(additional_experience_list)
    
    # Select projects based on number of experiences
    num_projects = 2 if num_experiences == 6 else 3
    projects_result = call_openai_projects(args.key, job_description, projects, num_projects) or {}
    selected_projects = projects_result.get("selected_projects", [])
    
    # Debug: Print what ChatGPT returned for projects
    print("\n=== ChatGPT returned the following projects ===")
    id_to_project_debug = {p["id"]: p for p in projects}
    for project_selection in selected_projects:
        project_id = project_selection.get("id")
        project_data = id_to_project_debug.get(project_id)
        if project_data:
            print(f"  - {project_data.get('name', 'Unknown')} ({project_id})")
        else:
            print(f"  - Unknown project ({project_id})")
    print("==============================================\n")
    
    # Build selection dict
    selection: Dict[str, Any] = {
        "selected_jobs": regular_jobs,
        "selected_projects": selected_projects,
        "skills": {},
    }
    if additional_experience_list:
        selection["additional_experience"] = additional_experience_list
    
    # Post-process to limit bullets based on experience count
    # Use the LLM-ranked bullets (take first N from the ranked list)
    id_to_job = {j["id"]: j for j in jobs}
    id_to_additional_job = {j["id"]: j for j in (additional_experience_jobs or [])}
    
    if num_experiences == 6:
        # 6 experiences: 3 bullets per experience, 2 bullets per project
        for job_selection in selection.get("selected_jobs", []):
            bullet_indices = job_selection.get("bullet_indices", [])
            job_selection["bullet_indices"] = bullet_indices[:3]
        
        if selection.get("additional_experience"):
            for job_selection in selection["additional_experience"]:
                bullet_indices = job_selection.get("bullet_indices", [])
                job_selection["bullet_indices"] = bullet_indices[:3]
        
        for project_selection in selection.get("selected_projects", []):
            bullet_indices = project_selection.get("bullet_indices", [])
            project_selection["bullet_indices"] = bullet_indices[:2]
    
    elif num_experiences == 5:
        # 5 experiences: 3-4 bullets for Professional, 3 bullets for Additional, 2 bullets for Projects
        for job_selection in selection.get("selected_jobs", []):
            bullet_indices = job_selection.get("bullet_indices", [])
            # Use 3-4 bullets (prefer 4)
            job_selection["bullet_indices"] = bullet_indices[:4]
        
        if selection.get("additional_experience"):
            for job_selection in selection["additional_experience"]:
                bullet_indices = job_selection.get("bullet_indices", [])
                job_selection["bullet_indices"] = bullet_indices[:3]
        
        for project_selection in selection.get("selected_projects", []):
            bullet_indices = project_selection.get("bullet_indices", [])
            project_selection["bullet_indices"] = bullet_indices[:2]
    
    elif num_experiences == 4:
        # 4 experiences: 4 bullets for Professional, 3 bullets for Additional, 3 bullets for Projects
        for job_selection in selection.get("selected_jobs", []):
            bullet_indices = job_selection.get("bullet_indices", [])
            job_selection["bullet_indices"] = bullet_indices[:4]
        
        if selection.get("additional_experience"):
            for job_selection in selection["additional_experience"]:
                bullet_indices = job_selection.get("bullet_indices", [])
                job_selection["bullet_indices"] = bullet_indices[:3]
        
        for project_selection in selection.get("selected_projects", []):
            bullet_indices = project_selection.get("bullet_indices", [])
            project_selection["bullet_indices"] = bullet_indices[:3]
    
    # Ensure projects never have more than 3 bullets
    for project_selection in selection.get("selected_projects", []):
        bullet_indices = project_selection.get("bullet_indices", [])
        project_selection["bullet_indices"] = bullet_indices[:3]
    
    # Fallback to default selection if no jobs were selected
    default_selection = _build_default_selection(jobs, projects)

    # Fallback to default selection if no jobs were selected
    if not selection.get("selected_jobs") and not selection.get("additional_experience"):
        selection["selected_jobs"] = default_selection["selected_jobs"]
        # Re-apply bullet limits based on count (for fallback, use first N bullets)
        num_experiences = len(selection["selected_jobs"])
        if num_experiences == 6:
            for job_selection in selection["selected_jobs"]:
                bullet_indices = job_selection.get("bullet_indices", [])
                job_selection["bullet_indices"] = bullet_indices[:3]
        elif num_experiences == 5:
            for job_selection in selection["selected_jobs"]:
                bullet_indices = job_selection.get("bullet_indices", [])
                job_selection["bullet_indices"] = bullet_indices[:4]
        elif num_experiences == 4:
            for job_selection in selection["selected_jobs"]:
                bullet_indices = job_selection.get("bullet_indices", [])
                job_selection["bullet_indices"] = bullet_indices[:4]
    
    if not selection.get("selected_projects"):
        selection["selected_projects"] = default_selection["selected_projects"]
        # Re-apply bullet limits
        num_experiences = len(selection.get("selected_jobs", [])) + len(selection.get("additional_experience", []))
        if num_experiences == 6:
            for project_selection in selection["selected_projects"]:
                bullet_indices = project_selection.get("bullet_indices", [])
                project_selection["bullet_indices"] = bullet_indices[:2]
        elif num_experiences == 5:
            for project_selection in selection["selected_projects"]:
                bullet_indices = project_selection.get("bullet_indices", [])
                project_selection["bullet_indices"] = bullet_indices[:2]
        elif num_experiences == 4:
            for project_selection in selection["selected_projects"]:
                bullet_indices = project_selection.get("bullet_indices", [])
                project_selection["bullet_indices"] = bullet_indices[:3]
    
    # Ensure projects never have more than 3 bullets (fallback case)
    for project_selection in selection.get("selected_projects", []):
        bullet_indices = project_selection.get("bullet_indices", [])
        project_selection["bullet_indices"] = bullet_indices[:3]
    
    selection.setdefault("skills", {})

    payload = build_payload(
        name=name,
        contact=merged_contact,
        jobs=jobs,
        projects=projects,
        selection=selection,
        education=education,
        force_additional_experience=args.force_additional_experience,
        additional_experience_jobs=additional_experience_jobs,
    )

    template_text = read_text(args.template)
    latex_text = render_resume_latex(template_text, payload)
    out = write_and_compile_latex(latex_text, args.output)
    print(out)


if __name__ == "__main__":
    main()



