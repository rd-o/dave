from importlib.machinery import SourceFileLoader
alrp = SourceFileLoader("openalpr", "/home/ale/projects/openalprbin/usr/lib/python2.7/dist-packages/openalpr/openalpr.py").load_module()
from openalpr import Alpr
alpr = Alpr("us", "/home/ale/projects/openalprbin/etc/openalpr/openalpr.conf", "/home/ale/projects/openalprbin/usr/share/openalpr/runtime_data")
if not alpr.is_loaded():
    print("Error loading OpenALPR")
    sys.exit(1)

alpr.set_top_n(20)
alpr.set_default_region("md")

results = alpr.recognize_file("autos/01.jpg")

print(results)
