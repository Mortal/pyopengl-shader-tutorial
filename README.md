Following the OpenGLContext Python Tutorials in Python 3

http://pyopengl.sourceforge.net/context/tutorials/

Get dependencies working using 2to3 and some small patches:

```
pyvenv venv
source venv/bin/activate
for i in OpenGLContext PyVRML97 PyVRML97-accelerate; do
pip install -d venv $i
cd venv
tar xf $i*.tar.*
cd $i*/
git init . && git add . && git commit -m 'Initial commit' && 2to3 -j 4 -w -n --no-diffs . && git commit -am 2to3
patch -p1 < ../../`basename \`pwd\``-post2to3.patch
git commit -am Fix
cd ../..
done
pip install -r requirements.txt
```
