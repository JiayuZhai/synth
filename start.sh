python -m visdom.server >out.txt 2>&1 &
jupyter notebook >out.txt 2>&1 &
x-www-browser 'http://localhost:8097' >out.txt 2>&1 &
