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
# Limit the number of stories fetched for quick testing (default 5)
MAX_STORIES=${MAX_STORIES:-5}
POST_IDS=$(echo "$USER_JSON" | jq -r ".submitted[:$MAX_STORIES] | @csv")
# Convert the CSV list into one ID per line for easier processing
echo "$POST_IDS" | tr ',' '\n' > "${USERNAME}_post_ids.txt"

# 3. Download each story
echo "Downloading stories …"
> "${USERNAME}_stories.json"
while read -r ID; do
  curl -s "$BASE_ITEM/$ID.json" | jq -c '.' >> "${USERNAME}_stories.json"
  sleep 0.1   # polite pause
done < "${USERNAME}_post_ids.txt"

# 4. Pull comments via Algolia (paginated)
echo "Fetching comments …"
COMMENTS_FILE="${USERNAME}_comments.json"
> "$COMMENTS_FILE"
COUNT=0
PAGE=0
while :; do
  RESP=$(curl -s "$ALGOLIA?query=user:$USERNAME&tags=comment&hitsPerPage=1000&page=$PAGE")
  # Get list of comment IDs from Algolia response
  COMMENT_IDS=$(echo "$RESP" | jq -r '.hits[]? | .objectID')
  for CID in $COMMENT_IDS; do
    # Stop if we reached the max comments
    if [ "$COUNT" -ge "$MAX_COMMENTS" ]; then break; fi
    # Fetch the full comment via HN API
    CMT=$(curl -s "$BASE_ITEM/$CID.json")
    # Extract the text field
    TEXT=$(echo "$CMT" | jq -r '.text // empty')
    echo "$TEXT" >> "$COMMENTS_FILE"
    COUNT=$((COUNT+1))
    sleep 0.05
  done
  # Stop if reached max
  if [ "$COUNT" -ge "$MAX_COMMENTS" ]; then break; fi
  HITS=$(echo "$RESP" | jq '.hits | length')
  [[ "$HITS" -lt 1000 ]] && break
  PAGE=$((PAGE+1))
  sleep 0.1
done

echo "All data written to:"
echo "  - ${USERNAME}_user.json"
echo "  - ${USERNAME}_stories.json"
echo "  - ${USERNAME}_comments.json"
