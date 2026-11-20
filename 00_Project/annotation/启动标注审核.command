#!/bin/zsh
cd -- "$(dirname -- "$0")/../.." || exit 1
python3 02_Code/utilities/review_boll_drafts.py --open
