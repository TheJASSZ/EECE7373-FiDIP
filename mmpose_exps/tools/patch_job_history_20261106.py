# One-off provenance fix (2026-11-06 ~07:10 EST): reconstruct job_history/training_* fields in run.json for
# runs whose post-train steps were completed by a later finalize job (D19 casualties). Run on the cluster.
import json, os
R = '/scratch/purohitham.v/fidip/runs/'
A80, L40S, A40 = 'NVIDIA A100-SXM4-80GB', 'NVIDIA L40S', 'NVIDIA A100-SXM4-40GB'
fails = {'R103': ('11659881', 'd1026', A40, 'OOM (packed with R104 on 40 GB), dir archived in runs/_failed'),
         'R104': ('11659881', 'd1026', A40, 'OOM (packed with R103 on 40 GB), dir archived in runs/_failed'),
         'R105': ('11659885', 'd1026', A40, 'OOM (packed with R106/R107 on 40 GB), dir archived in runs/_failed'),
         'R107': ('11659885', 'd1026', A40, 'OOM (packed with R105/R106 on 40 GB), dir archived in runs/_failed')}
train = {'R103': ('11664862', 'd4042', L40S, 'im_pack_screenB2 with R107'),
         'R104': ('11664861', 'd1029', A80, 'im_pack_screenA2 with R105'),
         'R105': ('11664861', 'd1029', A80, 'im_pack_screenA2 with R104'),
         'R107': ('11664862', 'd4042', L40S, 'im_pack_screenB2 with R103'),
         'R108': ('11664863', 'd4042', L40S, 'im_pack_lrL256 with R109'),
         'R109': ('11664863', 'd4042', L40S, 'im_pack_lrL256 with R108')}
fin = {k: ('11670941', 'd4042', L40S) for k in ('R103', 'R104', 'R105', 'R107')}
fin.update({k: ('11665220', 'd1026', A40) for k in ('R108', 'R109')})
for d in sorted(os.listdir(R)):
    k = d[:4]
    if k not in train or not os.path.isfile(R + d + '/run.json'):
        continue
    p = R + d + '/run.json'; r = json.load(open(p))
    h = []
    if k in fails:
        j, n, g, why = fails[k]; h.append(f'job={j} node={n} gpu={g} stage=train FAILED: {why}')
    j, n, g, why = train[k]; h.append(f'job={j} node={n} gpu={g} stage=train (all 30 epochs; {why}); post-train steps cut off by an in-place repo sync (D19)')
    j, n, g = fin[k]; h.append(f'job={j} node={n} gpu={g} stage=resume-at-epoch-30 (no training) + Val400 pred dump + eval_coco + finalize')
    r['job_history'] = h
    r['training_slurm_job_id'], r['training_node'], r['training_gpu'] = train[k][0], train[k][1], train[k][2]
    r['note_job_fields'] = ('slurm_job_id/partition/node/gpu = the finalize job; training_* = the job that trained '
                            '(reconstructed from slurm_logs/im_pack_*.out and the mmengine train logs)')
    json.dump(r, open(p, 'w'), indent=1)
    print('patched', d, r['training_gpu'])
