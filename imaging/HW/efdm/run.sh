scons -c
scons         
dvips handout.dvi -o handout.ps
ps2pdf handout.ps handout.pdf