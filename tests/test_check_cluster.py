import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash script: chạy trong docker")


def _stub(path: Path, body: str):
    path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    path.chmod(0o755)


def _setup(tmp_path, live_dn):
    bin_dir = tmp_path / "bin"
    kh = tmp_path / "kafka" / "bin"
    bin_dir.mkdir()
    kh.mkdir(parents=True)
    _stub(bin_dir / "hdfs", 'echo "Live datanodes (%d):"\n' % live_dn)
    _stub(bin_dir / "curl", 'echo \'{"workers":[{"state":"ALIVE"},{"state":"ALIVE"},{"state":"DEAD"}]}\'\n')
    box = tmp_path / "msg"
    _stub(kh / "kafka-topics.sh", "exit 0\n")
    _stub(kh / "kafka-console-producer.sh", 'cat > "%s"\n' % box)
    _stub(kh / "kafka-console-consumer.sh", 'cat "%s"\n' % box)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "cluster:\n  expected_datanodes: 2\n  expected_workers: 2\n"
        "  spark_master_ui: http://x:8080\n  kafka_home: %s\n  kafka_bootstrap: x:9092\n"
        % (tmp_path / "kafka"), encoding="utf-8")
    env = dict(os.environ, PATH="%s:%s" % (bin_dir, os.environ["PATH"]))
    return cfg, env


def _run(cfg, env):
    return subprocess.run(["bash", str(ROOT / "scripts" / "check_cluster.sh"), str(cfg)],
                          env=env, capture_output=True, text=True, timeout=60)


def test_all_pass(tmp_path):
    cfg, env = _setup(tmp_path, live_dn=2)
    r = _run(cfg, env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("[PASS]") == 3


def test_hdfs_fail(tmp_path):
    cfg, env = _setup(tmp_path, live_dn=1)
    r = _run(cfg, env)
    assert r.returncode == 1
    assert "[FAIL] HDFS" in r.stdout and r.stdout.count("[PASS]") == 2
