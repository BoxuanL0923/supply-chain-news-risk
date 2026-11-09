"""Finish feature-only years, development fit and freeze; never invokes evaluation."""
import td2
import pipeline
import decoder
from runtime_v3 import OUT,sha

if __name__=='__main__':
    for year in range(2016,2025):
        p=OUT/f'features/year_{year}.parquet'; coverage=OUT/f'coverage/year_{year}.json'
        if p.exists():
            if not coverage.exists() or decoder.json_read(coverage)['feature_sha256']!=sha(p):
                raise PermissionError('Incomplete or changed feature partition')
        else: pipeline.build_year(year)
    if not (OUT/'features/year_2015.parquet').exists():
        raise PermissionError('2015 feature job still pending; run fit/cost/freeze after it completes')
    pipeline.fit()
    pipeline.cost()
    pipeline.freeze()
