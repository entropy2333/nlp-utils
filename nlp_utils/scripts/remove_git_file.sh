PATH_TO_REMOVE_FILE='xxx'
git filter-branch --force --index-filter 'git \
rm --cached --ignore-unmatch ${PATH_TO_REMOVE_FILE}' \
--prune-empty --tag-name-filter cat -- --all

# git push origin main --force --all