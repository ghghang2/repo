#!/usr/bin/env bash
# --------------------------------------------------------------
#  Fetch all stories and comments for a HackerNews user
# --------------------------------------------------------------
set -euo pipefail

USERNAME="johnnyanmac"   # <-- change if you want a different account
BASE_ITEM="https://hacker-news.firebaseio.com/v0/item"
BASE_USER="https://hacker-news.firebaseio.com/v0/user"
ALGOLIA="https://hn.algolia.com/api/v1/search_by_date"

# 1. Grab user meta-data
echo "Fetching user metadata for $USERNAME …"
USER_JSON=$(curl -s "$BASE_USER/$USERNAME.json")
if [ "$USER_JSON" = "null" ]; then
    echo "❌  User \"$USERNAME\" not found on HackerNews."
    exit 1
fi
echo "$USER_JSON" > "${USERNAME}_user.json"

# 2. Pull story IDs (first 200 for speed – remove slice for all)
echo "Obtaining story IDs …"
POST_IDS=$(echo "$USER_JSON" | jq -r '.submitted[:200] | @csv')
echo "$POST_IDS" | tr ',' ' ' > "${USERNAME}_post_ids.txt"

# 3. Download each story
echo "Downloading stories …"
> "${USERNAME}_stories.json"
while read -r ID; do
  curl -s "$BASE_ITEM/$ID.json" | jq -c '.' >> "${USERNAME}_stories.json"
  sleep 0.1   # polite pause
done < "${USERNAME}_post_ids.txt"

# 4. Pull comments via Algolia (paginated)
echo "Fetching comments …"
> "$USERNAME_comments.json"
PAGE=0
while :; do
  RESP=$(curl -s "$ALGOLIA?tags=comment&hitsPerPage=1000&user=$USERNAME&page=$PAGE")
  # Extract only the comment text
  echo "$RESP" | jq -r '.hits[] | .text // empty' >> "$USERNAME_comments.txt"
  HITS=$(echo "$RESP" | jq '.hits | length')
  [[ "$HITS" -lt 1000 ]] && break
  PAGE=$((PAGE+1))
  sleep 0.1
done

echo "All data written to:"
echo "  - ${USERNAME}_user.json"
echo "  - ${USERNAME}_stories.json"
echo "  - ${USERNAME}_comments.json"
