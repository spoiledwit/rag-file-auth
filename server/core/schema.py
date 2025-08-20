"""Custom drf-spectacular preprocessing to limit exposed endpoints in generated schema.

We keep only these endpoints:
  - /api/auth/login/
  - /api/v1/categories/
  - /api/v1/query-document/
  - /api/v1/task/{task_id}/
"""
from typing import List, Tuple

Endpoint = Tuple[str, str, str, str, object]

ALLOWED_PATH_PREFIXES = [
    '/api/auth/login/',
    '/api/v1/categories/',
    '/api/v1/query-document/',
    '/api/v1/task/',  # dynamic UUID follows
]

def only_selected_endpoints(endpoints: List[Endpoint]) -> List[Endpoint]:
    """Filter endpoints to only allowed paths defined above.

    drf-spectacular passes a list of tuples:
        (path, path_regex, path_prefix, method, callback)
    """
    filtered = []
    for ep in endpoints:
        path = ep[0]
        if any(path.startswith(prefix) for prefix in ALLOWED_PATH_PREFIXES):
            filtered.append(ep)
    return filtered
