export RSFROOT="$HOME/Programs/madagascar"
export RSFSRC="$RSFROOT/src"
export PATH="$RSFROOT/bin:$PATH"
export LD_LIBRARY_PATH="$RSFROOT/lib:${LD_LIBRARY_PATH:-}"
export MANPATH="$RSFROOT/share/man:${MANPATH:-}"
export DATAPATH="$HOME/rsfdata/"
export TEXINPUTS=".:$RSFSRC/texmf//:${TEXINPUTS:-}"
export PYTHONPATH="$RSFROOT/lib/python3.13/site-packages:$RSFROOT/lib:$HOME/Workspace/mines/imaging/HW/CODE:$RSFSRC/book/Recipes:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16

