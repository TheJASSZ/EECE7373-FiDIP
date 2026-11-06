#!/usr/bin/env bash
# Build report/FiDIP_Report.pdf with tectonic and summarise what still needs attention.
#
#   bash report/build.sh            # build + warning summary
#   bash report/build.sh --png DIR  # also render every page to DIR/page-NN.png (needs pdftoppm)
#
# Needs tectonic (macOS: `brew install tectonic`; Linux: conda-forge `tectonic`). Tectonic fetches
# the LaTeX packages it needs on first use and caches them (~50 MB).
# Overleaf alternative: upload report/ and compile report.tex with pdfLaTeX.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PNG_DIR=""
if [[ "${1:-}" == "--png" ]]; then PNG_DIR="${2:?usage: build.sh --png DIR}"; fi

# provenance stamp shown under the title (not an experimental number)
sha=$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || echo unknown)
dirty=""
if ! git -C "$HERE" diff --quiet HEAD -- . 2>/dev/null || [[ -n "$(git -C "$HERE" ls-files --others --exclude-standard -- . 2>/dev/null)" ]]; then
  dirty="+uncommitted report changes"
fi
mkdir -p generated build
cat > generated/build_info.tex <<EOF
% written by report/build.sh -- do not edit
\\setres{build.date}{$(date '+%Y-%m-%d %H:%M %Z')}
\\setres{build.commit}{\\texttt{${sha}}${dirty:+ (${dirty})}}
EOF

tectonic -X compile report.tex --keep-logs --keep-intermediates --outdir build >build/tectonic.out 2>&1 || {
  echo "tectonic FAILED -- last lines:"; tail -40 build/tectonic.out; exit 1; }
cp build/report.pdf FiDIP_Report.pdf

LOG=build/report.log
pages=$(grep -a -o 'Output written on [^ ]* ([0-9]* page' "$LOG" | grep -o '[0-9]* page' | grep -o '[0-9]*' || true)
[[ -z "$pages" ]] && command -v pdfinfo >/dev/null && pages=$(pdfinfo FiDIP_Report.pdf | awk '/^Pages:/{print $2}')
echo "built report/FiDIP_Report.pdf (${pages:-?} pages)"
echo "--- attention list -------------------------------------------------------------"
printf '  undefined result keys : %s\n' "$(grep -a -o "Result key \`[^']*' is not defined" "$LOG" | sort -u | wc -l | tr -d ' ')"
grep -a -o "Result key \`[^']*' is not defined" "$LOG" | sort -u | sed 's/^/      /' || true
printf '  TBD values typeset    : %s\n' "$(grep -a -o '[0-9]* TBD placeholder' "$LOG" | grep -o '^[0-9]*' || echo 0)"
printf '  interpretation notes  : %s\n' "$(grep -a -o '[0-9]* interpretation placeholder' "$LOG" | grep -o '^[0-9]*' || echo 0)"
printf '  pending gen. files    : %s\n' "$(grep -a -o '[0-9]* pending generated' "$LOG" | grep -o '^[0-9]*' || echo 0)"
printf '  undefined refs/cites  : %s\n' "$(grep -a -c -E "(Reference|Citation) .* undefined" "$LOG" || true)"
grep -a -E "(Reference|Citation) .* undefined" "$LOG" | sort -u | sed 's/^/      /' || true
printf '  overfull boxes        : %s\n' "$(grep -a -c '^Overfull' "$LOG" || true)"
grep -a -A1 '^Overfull' "$LOG" | grep -a -v '^--$' | sed 's/^/      /' | head -40 || true

if [[ -n "$PNG_DIR" ]]; then
  mkdir -p "$PNG_DIR"
  pdftoppm -r 70 -png FiDIP_Report.pdf "$PNG_DIR/page"
  echo "pages rendered to $PNG_DIR"
fi
