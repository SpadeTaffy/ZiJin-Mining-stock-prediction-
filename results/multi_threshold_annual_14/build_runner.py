from pathlib import Path
out=Path(__file__).resolve().parent
s=Path('outputs/multi_threshold_2016_14/experiment.py').read_text().split("if __name__=='__main__':")[0]
s=s.replace("OUT=Path(os.environ.get('MLP14_MULTI_OUT',str(Path(__file__).resolve().parent)))", "RUN_OUT=Path(__file__).resolve().parent\nOUT=RUN_OUT")
s=s.replace('def experiment():','def experiment(year):')
s=s.replace('te=x.index[x.index.year==2016];assert len(te)==268','te=x.index[x.index.year==year];assert len(te)>0')
s=s.replace('effective.year==2016','effective.year==year')
s=s.replace("prior/'2016/updates.csv'", "prior/str(year)/'updates.csv'")
s=s.replace("pd.Timestamp('2016-01-01')", "pd.Timestamp(f'{year}-01-01')")
s=s.replace("prior/'2016'/", "prior/str(year)/")
s=s.replace("pd.Timestamp('2017-01-01')", "pd.Timestamp(f'{year+1}-01-01')")
s=s.replace("prior/'2016/predictions.csv'", "prior/str(year)/'predictions.csv'")
s=s.replace('config=dict(year=2016,','config=dict(year=year,')
# Reuse frozen deployed decisions even when prior 2020 selection JSONs were not materialized.
s=s.replace("        if cache.exists():", "        matched=old_updates[old_updates.Effective_date==date]\n        if len(matched):\n            r=matched.iloc[0];decision=dict(Selected_seed=int(r.Seed),Validation_MAPE=float(r.Validation_MAPE),Source='reused baseline update seed')\n            (selection_dir/cache.name).write_text(json.dumps(decision,indent=2))\n        elif cache.exists():")
s=s.replace("print(f'Original updates", "print(f'{year}: Original updates")
s=s.replace("print(f'Refit", "print(f'{year}: Refit")
s+='''

def year_run(year):
    global OUT
    OUT=RUN_OUT/str(year);OUT.mkdir(exist_ok=True)
    if (OUT/'config.json').exists() and (OUT/'summary.csv').exists():
        print(f'{year}: reuse completed results',flush=True);return year
    experiment(year)
    print(f'{year}: COMPLETED',flush=True)
    return year

if __name__=='__main__':
    import shutil
    old2016=RUN_OUT.parent/'multi_threshold_2016_14'
    if not old2016.exists():old2016=ROOT/'results/multi_threshold_2016_14'
    target=RUN_OUT/'2016';target.mkdir(exist_ok=True)
    for pattern in ['*.csv','config.json']:
        for file in old2016.glob(pattern):shutil.copy2(file,target/file.name)
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(year_run,year) for year in range(2015,2027)]
        for f in as_completed(futures):f.result()
    print('ALL YEARS COMPLETE',flush=True)
'''
(out/'run.py').write_text(s)
