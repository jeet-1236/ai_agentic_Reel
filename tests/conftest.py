import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.context import RunContext  # noqa: E402


@dataclass
class TmpRunContext(RunContext):
    """RunContext writing outputs/logs/data under a tmp dir; configs & prompts still come from the repo."""
    tmp: Path = Path(".")

    @property
    def out_dir(self) -> Path:
        p = self.tmp / "outputs" / self.run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def logs_dir(self) -> Path:
        p = self.tmp / "logs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def sources_dir(self) -> Path:
        p = self.tmp / "data" / "sources"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cache_db(self) -> Path:
        return self.tmp / "data" / "cache.sqlite"


@pytest.fixture
def ctx(tmp_path) -> TmpRunContext:
    return TmpRunContext(run_id="test-run", seed=42, brief_path=ROOT / "briefs" / "P00.yaml",
                         mock_llm=True, mock_media=True, offline=True, tmp=tmp_path)


@pytest.fixture
def log(ctx):
    from core.logging import RunLogger

    return RunLogger(ctx)


@pytest.fixture
def llm(ctx, log):
    from core.llm import LLM

    return LLM(ctx, log, mock=True)
