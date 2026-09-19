import os,io,contextlib,traceback
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell
os.environ['MLP14_MULTI_OUT']=str(Path('outputs/multi_threshold_2016_14').resolve())
os.environ['MPLCONFIGDIR']='/private/tmp/mlp11-matplotlib'
p=Path('outputs/14_继续增加规则.ipynb')
n=nbformat.read(p,as_version=4); shell=InteractiveShell.instance(); scope={}; count=0
for cell in n.cells[int(Path("outputs/multi_threshold_2016_14/start_cell.txt").read_text()):]:
 if cell.cell_type!='code': continue
 count+=1; cell.execution_count=count; cell.outputs=[]
 def publish(data,metadata=None,**kwargs):
  cell.outputs.append(nbformat.v4.new_output('display_data',data=data,metadata=metadata or {}))
 shell.display_pub.publish=publish
 buf=io.StringIO()
 try:
  with contextlib.redirect_stdout(buf): exec(compile(cell.source,str(p),'exec'),scope)
 except Exception:
  print(buf.getvalue()); traceback.print_exc(); nbformat.write(n,p); raise
 if buf.getvalue():
  cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=buf.getvalue())); print(buf.getvalue())
 print('Completed cell',count,flush=True)
nbformat.validate(n); nbformat.write(n,p)
print('2016 all-variable charts complete.')
