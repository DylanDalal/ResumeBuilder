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
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        
        system = (
            "You are an assistant that selects relevant resume experiences from provided data. "
            "Return JSON with key: selected_jobs (list of {id, bullet_indices}). "
            "CRITICAL: You MUST select at least 4 experiences (4-6 total) from the provided jobs. "
            "If fewer than 4 jobs are provided, select all of them. "
            "For each relevant job, rank the top 4 bullets in terms of relevance to the role, with the most relevant bullets at the top. "
            "The bullet_indices array should contain the indices in RANKED ORDER (most relevant first), NOT in original order. "
            "For example, if bullets [0, 1, 2, 3, 4] exist and bullets 2 and 4 are most relevant, return [2, 4, 0, 1] not [0, 1, 2, 3]. "
            "Generously keyword match these bullets - they should be clearly relevant to the role. "
            "Only use bullets by index from the given entries."
        )
        
        user_data = {
            "job_description": job_description,
            "jobs": jobs,
        }
        user = json.dumps(user_data)
        completion = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception:
        return {"selected_jobs": []}


def call_openai_rank_bullets(api_key: str, job_description: str, job: Dict[str, Any], max_bullets: int = 4) -> List[int]:
    """Call OpenAI to rank bullets for a single job or project. Returns ranked bullet indices."""
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        
        system = (
            "You are an assistant that ranks resume bullets by relevance to a job description. "
            "Return JSON with key: bullet_indices (array of integers). "
            f"Rank the top {max_bullets} bullets from the provided entry (job or project) in terms of relevance to the role, with the most relevant bullets at the top. "
            "The bullet_indices array should contain the indices in RANKED ORDER (most relevant first), NOT in original order. "
            "For example, if bullets [0, 1, 2, 3, 4] exist and bullets 2 and 4 are most relevant, return [2, 4, 0, 1] not [0, 1, 2, 3]. "
            "Generously keyword match these bullets - they should be clearly relevant to the role. "
            "Only use bullets by index from the given entries."
        )
        
        user_data = {
            "job_description": job_description,
            "entry": job,
        }
        user = json.dumps(user_data)
        completion = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or "{}"
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
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or "{}"
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


def _fill_bullets_to_minimum(entry_data: Dict[str, Any], current_bullet_indices: List[int], min_bullets: int = 4) -> List[int]:
    """Fill bullets to reach minimum count, avoiding duplicate groups.
    
    Args:
        entry_data: The job or project data from JSON
        current_bullet_indices: Currently selected bullet indices (ranked)
        min_bullets: Minimum number of bullets desired (default 4)
    
    Returns:
        Updated list of bullet indices with additional bullets added if needed
    """
    all_bullets = entry_data.get("bullets", [])
    if len(all_bullets) == 0:
        return current_bullet_indices
    
    # Don't try to fill beyond what's available
    actual_min = min(min_bullets, len(all_bullets))
    
    # If we already have enough bullets, return as-is
    if len(current_bullet_indices) >= actual_min:
        return current_bullet_indices
    
    # Get groups already used by current bullets
    used_groups: set[str] = set()
    for idx in current_bullet_indices:
        if 0 <= idx < len(all_bullets):
            bullet = all_bullets[idx]
            normalized = _normalize_bullet(bullet)
            group = normalized.get("group")
            if group:
                used_groups.add(group)
    
    # Find additional bullets that aren't already selected and don't conflict with groups
    new_indices = list(current_bullet_indices)
    for idx in range(len(all_bullets)):
        if idx in current_bullet_indices:
            continue  # Already selected
        
        if len(new_indices) >= actual_min:
            break  # We have enough
        
        bullet = all_bullets[idx]
        normalized = _normalize_bullet(bullet)
        group = normalized.get("group")
        
        # Skip if this bullet is from a group we've already used
        if group and group in used_groups:
            continue
        
        # Add this bullet
        new_indices.append(idx)
        if group:
            used_groups.add(group)
    
    return new_indices


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
        for rank, idx in enumerate(indices):
            if 0 <= idx < len(job.get("bullets", [])):
                bullet = job.get("bullets", [])[idx]
                normalized = _normalize_bullet(bullet)
                all_bullets_with_metadata.append({
                    **normalized,
                    "source_type": "job",
                    "source_id": item.get("id"),
                    "job_data": job,
                    "original_rank": rank,  # Preserve ranked order
                    "original_index": idx,  # Store original bullet index
                })
    
    # Collect bullets from projects
    for item in selection.get("selected_projects", []):
        pr = id_to_project.get(item.get("id"))
        if not pr:
            continue
        indices = item.get("bullet_indices", [])
        for rank, idx in enumerate(indices):
            if 0 <= idx < len(pr.get("bullets", [])):
                bullet = pr.get("bullets", [])[idx]
                normalized = _normalize_bullet(bullet)
                all_bullets_with_metadata.append({
                    **normalized,
                    "source_type": "project",
                    "source_id": item.get("id"),
                    "project_data": pr,
                    "original_rank": rank,  # Preserve ranked order
                    "original_index": idx,  # Store original bullet index
                })
    
    # Filter conflicting bullets per-job (not globally)
    # Groups should only conflict within the same job, not across different jobs
    print("\n=== build_payload: Before group filtering ===")
    for item in selection.get("selected_jobs", []):
        job_id = item.get("id")
        job = id_to_job.get(job_id)
        company = job.get("company", "Unknown") if job else "Unknown"
        indices = item.get("bullet_indices", [])
        print(f"  Regular: {company} ({job_id}): {len(indices)} bullet indices")
    if selection.get("additional_experience"):
        for item in selection.get("additional_experience", []):
            job_id = item.get("id")
            job = id_to_additional_job.get(job_id) or id_to_job.get(job_id)
            company = job.get("company", "Unknown") if job else "Unknown"
            indices = item.get("bullet_indices", [])
            print(f"  Additional: {company} ({job_id}): {len(indices)} bullet indices")
    print(f"  Total bullets before filtering: {len(all_bullets_with_metadata)}")
    
    # Filter per-job and per-project (groups only conflict within the same entry)
    job_bullets_map: Dict[str, List[tuple]] = {}  # job_id -> list of (rank, text) tuples
    project_bullets_map: Dict[str, List[tuple]] = {}  # project_id -> list of (rank, text) tuples
    job_kept_indices_map: Dict[str, List[int]] = {}  # job_id -> list of kept indices after filtering
    
    # Group bullets by source_id first
    bullets_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for bullet in all_bullets_with_metadata:
        source_id = bullet.get("source_id")
        source_type = bullet.get("source_type")
        if source_id and (source_type == "job" or source_type == "project"):
            if source_id not in bullets_by_source:
                bullets_by_source[source_id] = []
            bullets_by_source[source_id].append(bullet)
    
    # Filter each job/project separately
    for source_id, bullets in bullets_by_source.items():
        filtered = _filter_conflicting_bullets(bullets)
        source_type = bullets[0].get("source_type") if bullets else None
        
        # Track kept indices for jobs
        if source_type == "job":
            kept_indices = [bullet.get("original_index") for bullet in filtered if bullet.get("original_index") is not None]
            job_kept_indices_map[source_id] = kept_indices
        
        for bullet in filtered:
            text = bullet.get("text", "")
            original_rank = bullet.get("original_rank", 999)
            if source_type == "job":
                if source_id not in job_bullets_map:
                    job_bullets_map[source_id] = []
                job_bullets_map[source_id].append((original_rank, text))
            elif source_type == "project":
                if source_id not in project_bullets_map:
                    project_bullets_map[source_id] = []
                project_bullets_map[source_id].append((original_rank, text))
    
    print(f"  Total bullets after per-job filtering: {sum(len(bullets) for bullets in job_bullets_map.values()) + sum(len(bullets) for bullets in project_bullets_map.values())}")
    
    # Sort bullets by original rank to preserve LLM ranking
    for job_id in job_bullets_map:
        job_bullets_map[job_id].sort(key=lambda x: x[0])
        job_bullets_map[job_id] = [text for _, text in job_bullets_map[job_id]]
    
    for project_id in project_bullets_map:
        project_bullets_map[project_id].sort(key=lambda x: x[0])
        project_bullets_map[project_id] = [text for _, text in project_bullets_map[project_id]]
    
    # Refill jobs to 4 bullets if group filtering reduced them below 4
    for item in selection.get("selected_jobs", []):
        job_id = item.get("id")
        job = id_to_job.get(job_id)
        if not job:
            continue
        current_bullets = job_bullets_map.get(job_id, [])
        if len(current_bullets) < 4:
            # Get indices that were kept after filtering
            kept_indices = job_kept_indices_map.get(job_id, [])
            # Refill using the kept indices as a starting point
            refilled_indices = _fill_bullets_to_minimum(job, kept_indices, min_bullets=4)
            
            # Rebuild the bullet list from refilled indices
            all_job_bullets = job.get("bullets", [])
            refilled_bullets = []
            for idx in refilled_indices:
                if 0 <= idx < len(all_job_bullets):
                    bullet = all_job_bullets[idx]
                    normalized = _normalize_bullet(bullet)
                    refilled_bullets.append(normalized.get("text", ""))
            
            job_bullets_map[job_id] = refilled_bullets
    
    print("\n=== build_payload: After group filtering and reconstruction ===")
    for job_id in job_bullets_map:
        job = id_to_job.get(job_id)
        company = job.get("company", "Unknown") if job else "Unknown"
        bullets = job_bullets_map[job_id]
        print(f"  Regular: {company} ({job_id}): {len(bullets)} bullets")
    if selection.get("additional_experience"):
        for item in selection.get("additional_experience", []):
            job_id = item.get("id")
            job = id_to_additional_job.get(job_id) or id_to_job.get(job_id)
            company = job.get("company", "Unknown") if job else "Unknown"
            bullets = job_bullets_map.get(job_id, [])
            print(f"  Additional: {company} ({job_id}): {len(bullets)} bullets")
    print("===========================================================\n")
    
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
            for rank, idx in enumerate(indices):
                if 0 <= idx < len(job.get("bullets", [])):
                    bullet = job.get("bullets", [])[idx]
                    normalized = _normalize_bullet(bullet)
                    additional_bullets_with_metadata.append({
                        **normalized,
                        "source_type": "additional_job",
                        "source_id": item.get("id"),
                        "job_data": job,
                        "original_rank": rank,  # Preserve ranked order
                        "original_index": idx,  # Store original bullet index
                    })
        
        # Filter conflicting bullets per-job for additional experience
        # Group bullets by source_id first, then filter each job separately
        additional_bullets_by_source: Dict[str, List[Dict[str, Any]]] = {}
        for bullet in additional_bullets_with_metadata:
            source_id = bullet.get("source_id")
            if source_id:
                if source_id not in additional_bullets_by_source:
                    additional_bullets_by_source[source_id] = []
                additional_bullets_by_source[source_id].append(bullet)
        
        # Filter each job separately
        additional_job_bullets_map: Dict[str, List[tuple]] = {}
        additional_kept_indices_map: Dict[str, List[int]] = {}
        for source_id, bullets in additional_bullets_by_source.items():
            filtered = _filter_conflicting_bullets(bullets)
            kept_indices = [bullet.get("original_index") for bullet in filtered if bullet.get("original_index") is not None]
            additional_kept_indices_map[source_id] = kept_indices
            for bullet in filtered:
                text = bullet.get("text", "")
                original_rank = bullet.get("original_rank", 999)
                if source_id not in additional_job_bullets_map:
                    additional_job_bullets_map[source_id] = []
                additional_job_bullets_map[source_id].append((original_rank, text))
        
        # Sort bullets by original rank to preserve LLM ranking
        for job_id in additional_job_bullets_map:
            additional_job_bullets_map[job_id].sort(key=lambda x: x[0])
            additional_job_bullets_map[job_id] = [text for _, text in additional_job_bullets_map[job_id]]
        
        # Refill additional experience jobs to 3 bullets if group filtering reduced them below 3
        # Additional experiences always have 3 bullets max
        for item in selection.get("additional_experience", []):
            job_id = item.get("id")
            job = id_to_additional_job.get(job_id) or id_to_job.get(job_id)
            if not job:
                continue
            current_bullets = additional_job_bullets_map.get(job_id, [])
            if len(current_bullets) < 3:
                # Get indices that were kept after filtering
                kept_indices = additional_kept_indices_map.get(job_id, [])
                # Refill using the kept indices as a starting point (to 3 bullets, not 4)
                refilled_indices = _fill_bullets_to_minimum(job, kept_indices, min_bullets=3)
                
                # Rebuild the bullet list from refilled indices
                all_job_bullets = job.get("bullets", [])
                refilled_bullets = []
                for idx in refilled_indices:
                    if 0 <= idx < len(all_job_bullets):
                        bullet = all_job_bullets[idx]
                        normalized = _normalize_bullet(bullet)
                        refilled_bullets.append(normalized.get("text", ""))
                
                additional_job_bullets_map[job_id] = refilled_bullets
        
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
    parser.add_argument("--required-projects", required=False, default=None, help="Comma-separated list of project IDs that MUST be included in the resume")
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

    # Parse required project IDs from command line argument
    required_project_ids: List[str] | None = None
    if args.required_projects:
        required_project_ids = [project_id.strip() for project_id in args.required_projects.split(",") if project_id.strip()]

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
                        # Select top 3 bullets for additional experience (or all if less than 3)
                        num_bullets = min(3, len(bullets))
                        bullet_indices = list(range(num_bullets))
                        additional_experience_list.append({
                            "id": manual_id,
                            "bullet_indices": bullet_indices
                        })
    
    # Fill bullets to 4 per experience immediately after ChatGPT returns (before truncation)
    # This ensures each experience looks complete even if ChatGPT only selected 2-3 relevant bullets
    id_to_job = {j["id"]: j for j in jobs}
    id_to_additional_job = {j["id"]: j for j in (additional_experience_jobs or [])}
    
    print("\n=== Bullet counts: After ChatGPT returns ===")
    for job_selection in regular_jobs:
        job_id = job_selection.get("id")
        job_data = id_to_job.get(job_id)
        if job_data:
            company = job_data.get("company", "Unknown")
            current_indices = job_selection.get("bullet_indices", [])
            print(f"  Regular: {company} ({job_id}): {len(current_indices)} bullets")
            filled_indices = _fill_bullets_to_minimum(job_data, current_indices, min_bullets=4)
            job_selection["bullet_indices"] = filled_indices
            print(f"    -> After fill: {len(filled_indices)} bullets (available: {len(job_data.get('bullets', []))})")
    
    for job_selection in additional_experience_list:
        job_id = job_selection.get("id")
        job_data = id_to_additional_job.get(job_id) or id_to_job.get(job_id)
        if job_data:
            company = job_data.get("company", "Unknown")
            current_indices = job_selection.get("bullet_indices", [])
            print(f"  Additional: {company} ({job_id}): {len(current_indices)} bullets")
            # Additional experiences should be filled to 3 bullets (not 4)
            filled_indices = _fill_bullets_to_minimum(job_data, current_indices, min_bullets=3)
            job_selection["bullet_indices"] = filled_indices
            print(f"    -> After fill: {len(filled_indices)} bullets (available: {len(job_data.get('bullets', []))})")
    print("==========================================\n")
    
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
                            # Rank bullets using LLM (additional experiences have 3 bullets max)
                            bullet_indices = call_openai_rank_bullets(args.key, job_description, job, max_bullets=3)
                            if not bullet_indices:
                                # Fallback to sequential if ranking fails
                                num_bullets = min(3, len(bullets))
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
    
    # Filter projects by required_project_ids if provided
    projects_to_send = projects
    if required_project_ids:
        projects_to_send = [p for p in projects if p.get("id") in required_project_ids]
    
    # Select projects based on number of experiences
    num_projects = 2 if num_experiences == 6 else 3
    projects_result = call_openai_projects(args.key, job_description, projects_to_send, num_projects) or {}
    selected_projects = projects_result.get("selected_projects", [])
    
    # Ensure all required projects are included
    if required_project_ids:
        id_to_project = {p["id"]: p for p in projects}
        selected_project_ids = {project.get("id") for project in selected_projects}
        
        for required_id in required_project_ids:
            if required_id not in selected_project_ids:
                project = id_to_project.get(required_id)
                if project:
                    bullets = project.get("bullets", [])
                    if bullets:
                        # Rank bullets using LLM
                        bullet_indices = call_openai_rank_bullets(args.key, job_description, project, max_bullets=3)
                        if not bullet_indices:
                            # Fallback to sequential if ranking fails
                            num_bullets = min(3, len(bullets))
                            bullet_indices = list(range(num_bullets))
                        selected_projects.append({
                            "id": required_id,
                            "bullet_indices": bullet_indices
                        })
    
    # Fill bullets to 3 per project immediately after ChatGPT returns (before truncation)
    id_to_project = {p["id"]: p for p in projects}
    print("\n=== Project bullet counts: After ChatGPT returns ===")
    for project_selection in selected_projects:
        project_id = project_selection.get("id")
        project_data = id_to_project.get(project_id)
        if project_data:
            name = project_data.get("name", "Unknown")
            current_indices = project_selection.get("bullet_indices", [])
            print(f"  {name} ({project_id}): {len(current_indices)} bullets")
            filled_indices = _fill_bullets_to_minimum(project_data, current_indices, min_bullets=3)
            project_selection["bullet_indices"] = filled_indices
            print(f"    -> After fill: {len(filled_indices)} bullets (available: {len(project_data.get('bullets', []))})")
    print("==================================================\n")
    
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
    
    # Post-process: Apply truncation rules based on number of professional experiences
    # Rules:
    # - Additional experiences: Always 3 bullets each
    # - 5 professional experiences: 3 bullets each, projects: 2 bullets
    # - 4 professional experiences: up to 4 bullets each, projects: 2 bullets if additional exists, else 3
    # - 3 professional experiences: up to 4 bullets each, projects: 3 bullets
    id_to_job = {j["id"]: j for j in jobs}
    id_to_additional_job = {j["id"]: j for j in (additional_experience_jobs or [])}
    id_to_project = {p["id"]: p for p in projects}
    
    num_professional = len(selection.get("selected_jobs", []))
    has_additional = bool(selection.get("additional_experience"))
    
    print(f"\n=== Truncation: {num_professional} professional experiences, {'with' if has_additional else 'without'} additional ===")
    
    # Apply truncation rules based on number of professional experiences
    if num_professional == 5:
        # 5 professional experiences: 3 bullets each, projects: 2 bullets
        print("  Target: 3 bullets per professional job, 3 per additional, 2 per project")
        for job_selection in selection.get("selected_jobs", []):
            job_id = job_selection.get("id")
            job_data = id_to_job.get(job_id)
            company = job_data.get("company", "Unknown") if job_data else "Unknown"
            bullet_indices = job_selection.get("bullet_indices", [])
            before = len(bullet_indices)
            job_selection["bullet_indices"] = bullet_indices[:3]
            after = len(job_selection["bullet_indices"])
            print(f"    Regular: {company} ({job_id}): {before} -> {after} bullets")
        
        if selection.get("additional_experience"):
            for job_selection in selection["additional_experience"]:
                job_id = job_selection.get("id")
                job_data = id_to_additional_job.get(job_id) or id_to_job.get(job_id)
                company = job_data.get("company", "Unknown") if job_data else "Unknown"
                bullet_indices = job_selection.get("bullet_indices", [])
                before = len(bullet_indices)
                job_selection["bullet_indices"] = bullet_indices[:3]
                after = len(job_selection["bullet_indices"])
                print(f"    Additional: {company} ({job_id}): {before} -> {after} bullets")
        
        for project_selection in selection.get("selected_projects", []):
            project_id = project_selection.get("id")
            project_data = id_to_project.get(project_id)
            name = project_data.get("name", "Unknown") if project_data else "Unknown"
            bullet_indices = project_selection.get("bullet_indices", [])
            before = len(bullet_indices)
            project_selection["bullet_indices"] = bullet_indices[:2]
            after = len(project_selection["bullet_indices"])
            print(f"    Project: {name} ({project_id}): {before} -> {after} bullets")
    
    elif num_professional == 4:
        # 4 professional experiences: up to 4 bullets each
        # Projects: 2 bullets if additional exists, else 3 bullets
        project_max = 2 if has_additional else 3
        print(f"  Target: 4 bullets per professional job, 3 per additional, {project_max} per project")
        for job_selection in selection.get("selected_jobs", []):
            job_id = job_selection.get("id")
            job_data = id_to_job.get(job_id)
            company = job_data.get("company", "Unknown") if job_data else "Unknown"
            bullet_indices = job_selection.get("bullet_indices", [])
            before = len(bullet_indices)
            # Limit to 4 bullets max
            job_selection["bullet_indices"] = bullet_indices[:4]
            after = len(job_selection["bullet_indices"])
            print(f"    Regular: {company} ({job_id}): {before} -> {after} bullets")
        
        if selection.get("additional_experience"):
            for job_selection in selection["additional_experience"]:
                job_id = job_selection.get("id")
                job_data = id_to_additional_job.get(job_id) or id_to_job.get(job_id)
                company = job_data.get("company", "Unknown") if job_data else "Unknown"
                bullet_indices = job_selection.get("bullet_indices", [])
                before = len(bullet_indices)
                job_selection["bullet_indices"] = bullet_indices[:3]
                after = len(job_selection["bullet_indices"])
                print(f"    Additional: {company} ({job_id}): {before} -> {after} bullets")
        
        for project_selection in selection.get("selected_projects", []):
            project_id = project_selection.get("id")
            project_data = id_to_project.get(project_id)
            name = project_data.get("name", "Unknown") if project_data else "Unknown"
            bullet_indices = project_selection.get("bullet_indices", [])
            before = len(bullet_indices)
            project_selection["bullet_indices"] = bullet_indices[:project_max]
            after = len(project_selection["bullet_indices"])
            print(f"    Project: {name} ({project_id}): {before} -> {after} bullets")
    
    elif num_professional == 3:
        # 3 professional experiences: up to 4 bullets each, projects: 3 bullets
        print("  Target: 4 bullets per professional job, 3 per additional, 3 per project")
        for job_selection in selection.get("selected_jobs", []):
            job_id = job_selection.get("id")
            job_data = id_to_job.get(job_id)
            company = job_data.get("company", "Unknown") if job_data else "Unknown"
            bullet_indices = job_selection.get("bullet_indices", [])
            before = len(bullet_indices)
            # Limit to 4 bullets max
            job_selection["bullet_indices"] = bullet_indices[:4]
            after = len(job_selection["bullet_indices"])
            print(f"    Regular: {company} ({job_id}): {before} -> {after} bullets")
        
        if selection.get("additional_experience"):
            for job_selection in selection["additional_experience"]:
                job_id = job_selection.get("id")
                job_data = id_to_additional_job.get(job_id) or id_to_job.get(job_id)
                company = job_data.get("company", "Unknown") if job_data else "Unknown"
                bullet_indices = job_selection.get("bullet_indices", [])
                before = len(bullet_indices)
                job_selection["bullet_indices"] = bullet_indices[:3]
                after = len(job_selection["bullet_indices"])
                print(f"    Additional: {company} ({job_id}): {before} -> {after} bullets")
        
        for project_selection in selection.get("selected_projects", []):
            project_id = project_selection.get("id")
            project_data = id_to_project.get(project_id)
            name = project_data.get("name", "Unknown") if project_data else "Unknown"
            bullet_indices = project_selection.get("bullet_indices", [])
            before = len(bullet_indices)
            project_selection["bullet_indices"] = bullet_indices[:3]
            after = len(project_selection["bullet_indices"])
            print(f"    Project: {name} ({project_id}): {before} -> {after} bullets")
    
    else:
        # Default: ensure minimums but don't truncate excessively
        # Professional: up to 4 bullets, Additional: 3 bullets, Projects: 3 bullets
        print(f"  Target: 4 bullets per professional job (default), 3 per additional, 3 per project")
        for job_selection in selection.get("selected_jobs", []):
            job_id = job_selection.get("id")
            job_data = id_to_job.get(job_id)
            company = job_data.get("company", "Unknown") if job_data else "Unknown"
            bullet_indices = job_selection.get("bullet_indices", [])
            before = len(bullet_indices)
            # Limit to 4 bullets max
            job_selection["bullet_indices"] = bullet_indices[:4]
            after = len(job_selection["bullet_indices"])
            print(f"    Regular: {company} ({job_id}): {before} -> {after} bullets")
        
        if selection.get("additional_experience"):
            for job_selection in selection["additional_experience"]:
                job_id = job_selection.get("id")
                job_data = id_to_additional_job.get(job_id) or id_to_job.get(job_id)
                company = job_data.get("company", "Unknown") if job_data else "Unknown"
                bullet_indices = job_selection.get("bullet_indices", [])
                before = len(bullet_indices)
                job_selection["bullet_indices"] = bullet_indices[:3]
                after = len(job_selection["bullet_indices"])
                print(f"    Additional: {company} ({job_id}): {before} -> {after} bullets")
        
        for project_selection in selection.get("selected_projects", []):
            project_id = project_selection.get("id")
            project_data = id_to_project.get(project_id)
            name = project_data.get("name", "Unknown") if project_data else "Unknown"
            bullet_indices = project_selection.get("bullet_indices", [])
            before = len(bullet_indices)
            project_selection["bullet_indices"] = bullet_indices[:3]
            after = len(project_selection["bullet_indices"])
            print(f"    Project: {name} ({project_id}): {before} -> {after} bullets")
    
    print("=== Final bullet counts ===")
    for job_selection in selection.get("selected_jobs", []):
        job_id = job_selection.get("id")
        job_data = id_to_job.get(job_id)
        company = job_data.get("company", "Unknown") if job_data else "Unknown"
        print(f"  Regular: {company} ({job_id}): {len(job_selection.get('bullet_indices', []))} bullets")
    if selection.get("additional_experience"):
        for job_selection in selection["additional_experience"]:
            job_id = job_selection.get("id")
            job_data = id_to_additional_job.get(job_id) or id_to_job.get(job_id)
            company = job_data.get("company", "Unknown") if job_data else "Unknown"
            print(f"  Additional: {company} ({job_id}): {len(job_selection.get('bullet_indices', []))} bullets")
    for project_selection in selection.get("selected_projects", []):
        project_id = project_selection.get("id")
        project_data = id_to_project.get(project_id)
        name = project_data.get("name", "Unknown") if project_data else "Unknown"
        print(f"  Project: {name} ({project_id}): {len(project_selection.get('bullet_indices', []))} bullets")
    print("==========================\n")
    
    # Fallback to default selection if no jobs were selected
    default_selection = _build_default_selection(jobs, projects)

    # Fallback to default selection if no jobs were selected
    if not selection.get("selected_jobs") and not selection.get("additional_experience"):
        selection["selected_jobs"] = default_selection["selected_jobs"]
        # Apply truncation rules to fallback selection
        num_professional_fallback = len(selection["selected_jobs"])
        if num_professional_fallback == 5:
            for job_selection in selection["selected_jobs"]:
                bullet_indices = job_selection.get("bullet_indices", [])
                job_selection["bullet_indices"] = bullet_indices[:3]
        else:
            for job_selection in selection["selected_jobs"]:
                bullet_indices = job_selection.get("bullet_indices", [])
                job_selection["bullet_indices"] = bullet_indices[:4]
    
    if not selection.get("selected_projects"):
        selection["selected_projects"] = default_selection["selected_projects"]
        # Apply truncation rules to fallback projects
        num_professional_fallback = len(selection.get("selected_jobs", []))
        has_additional_fallback = bool(selection.get("additional_experience"))
        if num_professional_fallback == 5:
            project_max = 2
        elif num_professional_fallback == 4:
            project_max = 2 if has_additional_fallback else 3
        else:
            project_max = 3
        for project_selection in selection["selected_projects"]:
            bullet_indices = project_selection.get("bullet_indices", [])
            project_selection["bullet_indices"] = bullet_indices[:project_max]
    
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



