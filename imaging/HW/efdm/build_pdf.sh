#!/bin/bash
# Build script for handout.pdf
# This script handles the DVI to PDF conversion automatically

# Set up Madagascar environment
export RSFROOT="$HOME/Programs/madagascar"
export RSFSRC="$RSFROOT/src"
export PATH="$HOME/bin:$RSFROOT/bin:$PATH"
export LD_LIBRARY_PATH="$RSFROOT/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$RSFROOT/lib/python3.13/site-packages:${PYTHONPATH:-}"

# If no arguments, default to handout.pdf
if [ $# -eq 0 ]; then
    TARGET="handout.pdf"
else
    TARGET="$@"
fi

# Run scons to build the target
/home/x/Programs/miniconda3/bin/scons "$TARGET"

# Check exit code
SCONS_EXIT=$?

# If DVI was built, convert to PDF
for dvi in handout*.dvi; do
    if [ -f "$dvi" ]; then
        pdf="${dvi%.dvi}.pdf"
        echo "Converting $dvi to $pdf..."
        dvipdf "$dvi" "$pdf"
    fi
done

exit $SCONS_EXIT
