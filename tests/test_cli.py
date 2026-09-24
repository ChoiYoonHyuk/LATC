import json
from pathlib import Path
import pytest
from latc.cli import execute


def test_cli_provenance_and_overwrite_guard(tmp_path):
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(rq='rq1',parameters=dict(dimensions=[3],budgets=[0,4],basis_dimensions=[]))))
    out=tmp_path/'results'
    result=execute(config,out)
    assert result['status']=='complete'
    assert result['result']['curve_rows']==2
    assert len(result['config_sha256'])==64
    assert (out/'curves.csv').is_file()
    with pytest.raises(FileExistsError):
        execute(config,out)


def test_configuration_typo_rejected(tmp_path):
    config=tmp_path/'bad.json'
    config.write_text(json.dumps(dict(rq='rq2',parameters=dict(sampels=16))))
    with pytest.raises(ValueError,match='Unknown'):
        execute(config,tmp_path/'out')
