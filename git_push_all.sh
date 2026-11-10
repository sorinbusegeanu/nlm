#!/usr/bin/env bash
set -euo pipefail

allowed_path() {
    case "$1" in
        .gitignore|git_push_all.sh|docs/*|src/*|tests/*) return 0 ;;
        *) return 1 ;;
    esac
}

check_outgoing_history() {
    local upstream commit path
    upstream=$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}')
    while IFS= read -r commit; do
        while IFS= read -r path; do
            if ! allowed_path "$path"; then
                printf 'Refusing to push: outgoing commit %s contains disallowed path: %s\n' "$commit" "$path" >&2
                return 1
            fi
        done < <(git diff-tree --root --no-commit-id --name-only -r "$commit")
    done < <(git rev-list --reverse "$upstream..HEAD")
}

# A push sends all commits beyond the upstream, including blobs later deleted
# from the tip, so validate existing outgoing history before making a commit.
check_outgoing_history

# Only stage the explicitly approved project content.
git add -- .gitignore docs/ src/ tests/

# Do not accidentally commit already-staged files outside the allowlist.
while IFS= read -r path; do
    if ! allowed_path "$path"; then
        printf 'Refusing to commit: staged path is outside docs/, src/, tests/, .gitignore, and git_push_all.sh: %s\n' "$path" >&2
        exit 1
    fi
done < <(git diff --cached --name-only)

if ! git diff --cached --quiet; then
    git commit -m "update"
fi

# Include the new commit in the outgoing-history check before pushing.
check_outgoing_history
git push
