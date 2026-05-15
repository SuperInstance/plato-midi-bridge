#!/usr/bin/env python3
"""Branch cleanup — merge stale branches, delete merged ones.

Usage: python3 branch_cleanup.py --repo <repo> [--dry-run]
"""

import subprocess, json, sys

def stale_branches(repo, dry_run=True):
    """Find stale branches across a repo."""
    result = subprocess.run(
        ["gh", "api", f"repos/SuperInstance/{repo}/branches", "--jq", ".[] | {name: .name, sha: .commit.sha}"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        return []
    
    branches = []
    for line in result.stdout.strip().split('\n'):
        if not line:
            continue
        try:
            data = json.loads(line)
            if data['name'] != 'main' and 'feat/' in data['name']:
                branches.append(data['name'])
        except:
            pass
    
    return branches

if __name__ == '__main__':
    repo = sys.argv[2] if len(sys.argv) > 1 else 'plato-midi-bridge'
    dry = '--dry-run' in sys.argv
    
    stale = stale_branches(repo, dry)
    print(f'{repo}: {len(stale)} stale branches')
    for b in stale:
        print(f'  {b}')
