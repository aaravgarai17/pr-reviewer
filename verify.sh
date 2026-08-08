#!/usr/bin/env bash
#
# One-command verification. Runs entirely offline — no API keys, no network,
# no cost — because GitHub and Claude are stubbed at their interfaces while
# every other stage runs the real production code.
#
# Usage:  ./verify.sh

set -uo pipefail

PY="${PYTHON:-python3}"
pass=0
fail=0

ok()  { echo "  ✓ $1"; pass=$(( pass + 1 )); }
bad() { echo "  ✗ $1"; fail=$(( fail + 1 )); }

echo "=================================================="
echo " 0/5  Preflight"
echo "=================================================="
command -v $PY >/dev/null || { echo "  ✗ python3 not found"; exit 1; }
if ! $PY -c "import pytest, fastapi, httpx, yaml" 2>/dev/null; then
  echo "  ✗ dependencies missing. Run:"
  echo "      python3 -m venv .venv && source .venv/bin/activate"
  echo "      pip install -r requirements.txt"
  exit 1
fi
ok "python and dependencies available"

echo ""
echo "=================================================="
echo " 1/5  Test suite"
echo "=================================================="
if $PY -m pytest -q -p no:cacheprovider 2>&1 | tail -3; then
  ok "all tests passed"
else
  bad "tests failed"
fi

echo ""
echo "=================================================="
echo " 2/5  Diff position mapping"
echo "=================================================="
# The core engineering: a line number in a file is NOT its position in a diff,
# and conflating them silently puts comments on the wrong lines.
if $PY - <<'EOF'
from app.diff import parse_diff

diff = """diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -10,3 +10,4 @@ def first():
     a = 1
+    b = 2
     c = 3
@@ -50,3 +51,4 @@ def second():
     x = 10
+    y = 20
     z = 30
"""
f = parse_diff(diff)[0]

second_hunk_first_line = f.hunks[1].lines[0]
assert second_hunk_first_line.new_line == 51, second_hunk_first_line.new_line
assert second_hunk_first_line.position == 5, second_hunk_first_line.position

added = [l for h in f.hunks for l in h.lines if l.kind.value == "added"]
assert [(l.new_line, l.position) for l in added] == [(11, 2), (52, 6)]
EOF
then
  ok "file line numbers and diff positions tracked separately"
else
  bad "position mapping is wrong"
fi

echo ""
echo "=================================================="
echo " 3/5  Webhook rejects forged signatures"
echo "=================================================="
if $PY - <<'EOF'
import hmac
from hashlib import sha256
from app.github import verify_signature

body = b'{"action":"opened"}'
good = "sha256=" + hmac.new(b"secret", body, sha256).hexdigest()

assert verify_signature("secret", body, good) is True
assert verify_signature("secret", body, "sha256=" + "0" * 64) is False
assert verify_signature("secret", b'{"action":"evil"}', good) is False
assert verify_signature("secret", body, None) is False
EOF
then
  ok "HMAC verification accepts valid and rejects forged requests"
else
  bad "signature verification is broken"
fi

echo ""
echo "=================================================="
echo " 4/5  Noise control filters model output"
echo "=================================================="
demo_output=$($PY demo.py 2>&1)

# Whitespace-tolerant: the demo aligns these columns, and hard-coding the
# padding makes the check brittle for no benefit.
check_stat() {   # check_stat <label regex> <expected count> <description>
  if echo "$demo_output" | grep -Eq "$1[[:space:]]+$2([[:space:]]|\$)"; then
    ok "$3"
  else
    bad "$3 (got: $(echo "$demo_output" | grep -E "$1" | tr -s ' '))"
  fi
}

check_stat "the model returned" 8 "8 raw findings from the model"
check_stat "dropped: not in the diff" 1 "dropped a finding citing a line outside the diff"
check_stat "dropped: duplicates" 1 "collapsed a duplicate comment"
check_stat "dropped: below severity floor" 1 "dropped a nit below the severity threshold"
check_stat "posted" 5 "5 comments survived filtering"

echo ""
echo "=================================================="
echo " 5/5  Comments land on real added lines"
echo "=================================================="
echo "$demo_output" | grep -q "api/users.py:17 (diff position 4)" \
  && ok "comment anchored to line 17 at diff position 4" \
  || bad "anchoring produced unexpected coordinates"

# The lockfile legitimately appears in the parsed-diff listing; what matters is
# that no comment was anchored to it (comment lines are prefixed with "──").
echo "$demo_output" | grep -q -- "── package-lock.json" \
  && bad "lockfile was reviewed (should be ignored)" \
  || ok "lockfile parsed but excluded from review"

echo "$demo_output" | grep -qi "no api keys were used" \
  && ok "entire pipeline ran offline" \
  || bad "demo did not complete"

echo ""
echo "=================================================="
echo " Results: $pass passed, $fail failed"
echo "=================================================="
[[ $fail -eq 0 ]] && echo "VERIFIED — every README claim checks out." \
                  || echo "FAILED — see above."
exit $(( fail > 0 ? 1 : 0 ))
