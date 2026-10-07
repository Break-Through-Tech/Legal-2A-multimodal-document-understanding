import sys, json, hashlib, copy, math, collections, importlib
from pathlib import Path
import numpy as np, pandas as pd
p=Path('ah-clustering-experiment').resolve()
sys.path.insert(0,str(p))
from src.artifacts import read_artifact
from dashboard.data import scan_catalog, load_run, load_projection
from src.provenance import source_identity
baseline={"metrics_to_clusters":{"metrics-017648c85fd92d988bb0":"clusters-2c570644206a69347818","metrics-059e6287ca1b8d5d3f00":"clusters-39ac0cd9385da467f171","metrics-2abb4654f1be19776302":"clusters-a3b97f5bd553ae9cc906","metrics-3026701ed2d79b33bfc8":"clusters-cc13d69204a20acd6d26","metrics-36b2ec46394c4b78e326":"clusters-4700e2bad113d169c12c","metrics-47a7fef83bccfe8950fe":"clusters-baecfbbcb1204706c2c5","metrics-4abc90e447daf4c326b9":"clusters-b187c9c9938cc1b5bc38","metrics-54e1fcbaf1a0bc914c94":"clusters-f0085da7b8db3aa2a971","metrics-67d2b7b2919d072373d1":"clusters-5faff8c976b54b2a2fbd","metrics-71967027d3b1c04fcd18":"clusters-40b280874929384e5bc9","metrics-925a5dbbdc796edc8422":"clusters-79f57ed40739b91cf392","metrics-ca3f65b30ad9565e6fee":"clusters-dbe077efe92a963b03df","metrics-caa6e7d343a717723785":"clusters-7c58a2ef2e2dc1e1560a","metrics-da8f99af889c4ea9e822":"clusters-c876fa642aa6c4948b1b"},"hashes":{"metrics-2abb4654f1be19776302":"1f0f8ade47df695134e3f2a3e1d4fa8a28ce0f9bd9b6f83ed589230a62f00447","metrics-47a7fef83bccfe8950fe":"c51f325be71e431e558d18eba6157ab636f5237ed1677aea809b954225610a58","metrics-67d2b7b2919d072373d1":"5e538b25562eca9fe3914e5a2d24de435037d6958ffe564293f7fd77877a7015","clusters-79f57ed40739b91cf392":"cbd5cb0e7f12474edbd6c1e8ce8e8d824320abb53d552599847a0148637b3e85","metrics-54e1fcbaf1a0bc914c94":"7c89afc7c1629e229e199fbad66739c34a25b3aa92715d17e159e877eccdba6f","clusters-baecfbbcb1204706c2c5":"09fe4655a483cc0f7e4557cda021de108c996bd91a8ee28e4488fc0910af9b53","clusters-40b280874929384e5bc9":"5060ddff62c1147e2efc90c188ceab8e76a7fd41a4bc67db437bb9b3b93602e4","metrics-017648c85fd92d988bb0":"7fd540440b2c301bd6f993662fae6bb812fd49004ff5c72bf947e320b2e66587","metrics-3026701ed2d79b33bfc8":"45bd51b62aa19d925efd1c48efe57ce9f0ff0e05b079b773a66b59664c781c44","metrics-925a5dbbdc796edc8422":"7f7dce08ec56fc3900a11ab04416b26b1382312b7f8e2ac577c0f6acf27435a7","metrics-4abc90e447daf4c326b9":"20f81d35614a897371fa1ad643a980c37d4ff50034c2022d3c199363613c1983","metrics-ca3f65b30ad9565e6fee":"1ef41e54fe08d0668e562a25b6b7e7a729b8989c90510d99ad063a97dfba4664","clusters-5faff8c976b54b2a2fbd":"c99c7b849bf5c0d63086056b2790cb859e863e44bd11de6e2e5120ff3e074f5b","metrics-059e6287ca1b8d5d3f00":"39236452ccf660b531caddaaca1fdf8eaf1a44476eb63ada91e2ae01df34d0d0","metrics-71967027d3b1c04fcd18":"44dd02956292133ac02a6d1dfa705cdd32609ff0b7d4152aac896b11932ffd06","clusters-c876fa642aa6c4948b1b":"beb1eb3431813e14520ffaf19e5ecf2b79917759114c4aa1b70d352962c28a6d","clusters-dbe077efe92a963b03df":"a701d467a1ff1197b972b292557d3d8563c339a865beaa69dbdb4f5d5a5fefbf","clusters-b187c9c9938cc1b5bc38":"5c1f111dcd8cd947da25a36f47f3efc2f9b8a40b8a4185de9dd20b0e7c6b2e71","clusters-7c58a2ef2e2dc1e1560a":"be08115cbab44e607978fcbd7c3e98572c4721f3529db38343ff9791e5e12b05","metrics-36b2ec46394c4b78e326":"95f4ba4416df17ce0910c7c8eac499d5184e72c10e24d8cb6afe1072747c44af","clusters-a3b97f5bd553ae9cc906":"d1e70157d55cf20803d56a34d06f2119fa3418730c54bf09da05aac856aa17dc","clusters-cc13d69204a20acd6d26":"239edc7fd6fe820633cd3ea5e21e6307b727cfaee8d1cb9ea6bc5fdea76ce033","clusters-2c570644206a69347818":"ce11d8c93b63f162787e132ddbd8a67b2f4b9075dce92a4974cc6683cd9e840b","metrics-caa6e7d343a717723785":"bfbff9fda38a08bba4d79daa0618410d386146d447aa3faae18bbd4f878b0d6d","clusters-4700e2bad113d169c12c":"8c5e73fae8ebbf1d01822ac2e56324e460aa762fdefdbe29086ac0e44e1fce74","clusters-f0085da7b8db3aa2a971":"983e7189a9e337e17655732f9c4d313dd798e7a0ef4bd2479881a1bb77764be7","clusters-39ac0cd9385da467f171":"a1047146f3741fea8c325f83650c8d41db1bd3cf1d4c94cba156b373515c03b8","metrics-da8f99af889c4ea9e822":"3085775ca578d4667b4d493358fc265446689900ca24e0ba13c23b0aba9b7485"}}
assert hashlib.sha256(Path('data/split.csv').read_bytes()).hexdigest()=='0f4c5d9e29795744d7eff33e192460a56ae4651b536b1723900fd2fe30e82e0c'
truth=pd.read_csv('data/split.csv').query("split == 'train'").sort_values('id').set_index('id')
ids=truth.index.tolist()
assert len(ids)==700
assert hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()=='7e430f9bad462d4fd417f5d7641afc26fcd2e4bfd66434586e1795ebe8528968'
for i,digest in baseline['hashes'].items():
    assert hashlib.sha256((p/'outputs/metadata'/i/'artifact.json').read_bytes()).hexdigest()==digest, i
    read_artifact(p,i)
settings=json.loads((p/'configs/evaluation.json').read_text())
assert len(settings['runs'])==18
assert set(baseline['metrics_to_clusters'].values()) <= {r['artifact_id'] for r in settings['runs']}
assert 'layoutlmv3-ocr' not in settings['pending']
catalog=scan_catalog(p)
assert len(catalog['runs'])==18, len(catalog['runs'])
assert set(baseline['metrics_to_clusters']) <= {r['artifact_id'] for r in catalog['runs']}
ocr=[r for r in catalog['runs'] if r['model']=='layoutlmv3-ocr']
assert len(ocr)==4
assert {r['algorithm'] for r in ocr}=={'kmeans','agglomerative','hdbscan','spectral'}
def agreement_ref(y,z):
    n=len(y); c=collections.Counter(zip(y,z)); a=collections.Counter(y); b=collections.Counter(z)
    pair=lambda k:k*(k-1)/2
    index=sum(map(pair,c.values())); expected=sum(map(pair,a.values()))*sum(map(pair,b.values()))/pair(n)
    maximum=(sum(map(pair,a.values()))+sum(map(pair,b.values())))/2
    ari=1.0 if maximum==expected else (index-expected)/(maximum-expected)
    entropy=lambda counts:-sum(v/n*math.log(v/n) for v in counts.values())
    mi=sum(v/n*math.log(v*n/(a[i]*b[j])) for (i,j),v in c.items())
    denominator=(entropy(a)+entropy(b))/2
    return ari,1.0 if denominator==0 else mi/denominator
def compare_tables(assignments, documents):
    assert assignments.document_id.tolist()==ids, 'Assignment cohort differs'
    assert documents.document_id.tolist()==ids, 'Evaluation cohort differs'
    assert assignments.cluster_id.tolist()==documents.cluster_id.tolist(), 'Evaluation assignments differ'
    assert documents.label.tolist()==truth.label.tolist(), 'Ground truth labels differ'
projections=set();records=[];controls=[]
for row in ocr:
    run=load_run(p,row['artifact_id'])
    cluster=read_artifact(p,row['clustering_artifact'])
    assert cluster['config']['embedding_artifact']=='embeddings-14f0c9c4cc8cb38ae601'
    assert cluster['config']['cohort']['document_ids']==ids
    assert cluster['config']['representation']=='cls' and cluster['config']['normalization']=={'kind':'l2'}
    assignments=cluster['tables']['assignments']
    documents=run['documents']
    compare_tables(assignments,documents)
    assert {r['method'] for r in run['projections']}=={'pca','umap'}
    for ref in run['projections']:
        frame=load_projection(p,ref,ids)
        assert frame.document_id.tolist()==ids
        assert np.isfinite(frame[['x','y']].to_numpy()).all()
        projections.add(ref['artifact_id'])
    for scope in ('including_noise','excluding_noise'):
        selected=documents if scope=='including_noise' else documents.loc[documents.cluster_id!=-1]
        scores=run['scores'][scope]
        for metric in scores['metrics'].values():
            assert (metric['value'] is None and bool(metric['reason'])) or (isinstance(metric['value'],(int,float)) and np.isfinite(metric['value']) and metric['reason'] is None)
        if len(selected)>=2:
            ari,nmi=agreement_ref(selected.label.tolist(),selected.cluster_id.tolist())
            assert abs(ari-scores['metrics']['ari']['value'])<1e-12
            assert abs(nmi-scores['metrics']['nmi']['value'])<1e-12
    records.append({'algorithm':row['algorithm'],'metrics':row['artifact_id'],'clusters':row['clustering_artifact'],'cluster_count':run['summary']['cluster_count'],'noise_count':run['summary']['noise_count'],'ari':run['scores']['including_noise']['metrics']['ari']['value'],'nmi':run['scores']['including_noise']['metrics']['nmi']['value']})
    if not controls:
        for name,bad_assignments,bad_documents in [
            ('missing_assignment_id',assignments.iloc[:-1].copy(),documents),
            ('changed_assignment_label',assignments.copy(),documents),
            ('changed_truth_label',assignments,documents.copy())]:
            if name=='changed_assignment_label':
                bad_assignments.loc[bad_assignments.index[0],'cluster_id']=999
            if name=='changed_truth_label':
                bad_documents.loc[bad_documents.index[0],'label']='BROKEN_ORACLE_CONTROL'
            try:compare_tables(bad_assignments,bad_documents)
            except AssertionError as e:controls.append({'control':name,'detected':str(e)})
            else:raise AssertionError('Negative control was not detected: '+name)
assert len(projections)==2
guards=[]
for module,names in [('experiments.run_clustering',['transformers','colpali_engine','src.embeddings.models']),('experiments.run_evaluation',['transformers','colpali_engine','src.embeddings.models','tensorflow'])]:
    blocker=importlib.import_module(module).NoEncoderImports()
    for name in names:
        try:blocker.find_spec(name)
        except ImportError as error:guards.append({'entrypoint':module,'module':name,'blocked':str(error)})
        else:raise AssertionError('Encoder guard allowed '+name)
assert 'transformers' not in sys.modules and 'colpali_engine' not in sys.modules and 'src.embeddings.models' not in sys.modules
print(json.dumps({'source_digest':source_identity(p)['source_digest'],'original_manifests_unchanged':len(baseline['hashes']),'original_metrics_retained':len(baseline['metrics_to_clusters']),'training_count':len(ids),'ocr_runs':records,'projection_ids':sorted(projections),'negative_controls':controls,'encoder_guard_checks':guards,'catalog_keys':list(catalog),'catalog_issues':{k:v for k,v in catalog.items() if k!='runs'}},default=str))

