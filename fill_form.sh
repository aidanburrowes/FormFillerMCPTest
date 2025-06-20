#!/usr/bin/env bash
set -euo pipefail                       # safer bash

API="http://localhost:8000"
PDF="form.pdf"                          # path to your blank PDF
NAME="Aidan"
EMAIL="aidan@example.com"
BIRTHDAY="1990-01-01"

LOG="fill_log_$(date +%Y%m%dT%H%M%S).txt"
exec > >(tee -a "$LOG") 2>&1            # tee stdout & stderr to log

echo "=== $(date)  Starting PDF fill session ==="

# 1) Create context
echo -e "\n[1] Creating context …"
CTX_JSON=$(curl -s -F "pdf=@${PDF}" "$API/contexts")
echo "Response: $CTX_JSON"
CTX_ID=$(echo "$CTX_JSON" | jq -r .id)
echo "Context ID: $CTX_ID"

# 2) Send answers
echo -e "\n[2] Sending answers …"
curl -s -X POST "$API/contexts/$CTX_ID/messages" \
     -H "Content-Type: application/json" \
     -d @- <<EOF | tee /dev/tty
{
  "answers": {
    "Name":     "$NAME",
    "Email":    "$EMAIL",
    "Birthday": "$BIRTHDAY"
  }
}
EOF

# 3) Ask agent to finish & download PDF
echo -e "\n[3] Requesting filled PDF …"
curl -s -X POST "$API/contexts/$CTX_ID/steps" --output filled_form.pdf
echo "filled_form.pdf written ✓"

echo -e "\n=== $(date)  Done. Full log in $LOG ==="
