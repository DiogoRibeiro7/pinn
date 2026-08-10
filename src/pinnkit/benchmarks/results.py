"""JSON-serializable benchmark result schema."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

ReferenceKind = Literal["analytic", "numerical", "observational"]


@dataclass(frozen=True)
class BenchmarkReference:
    """Describe the provenance of benchmark reference data."""

    kind: ReferenceKind
    name: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate reference provenance fields."""
        if self.kind not in {"analytic", "numerical", "observational"}:
            raise ValueError(f"unknown reference kind: {self.kind}")
        if not self.name:
            raise ValueError("reference name must not be empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {"kind": self.kind, "name": self.name, "details": self.details}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkReference":
        """Build a reference descriptor from decoded JSON."""
        return cls(
            kind=cast(ReferenceKind, data["kind"]),
            name=data["name"],
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BenchmarkMetric:
    """Named scalar metric captured in a benchmark report."""

    name: str
    value: float
    units: str | None = None
    higher_is_better: bool | None = None

    def __post_init__(self) -> None:
        """Validate metric fields."""
        if not self.name:
            raise ValueError("metric name must not be empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "name": self.name,
            "value": float(self.value),
            "units": self.units,
            "higher_is_better": self.higher_is_better,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkMetric":
        """Build a metric from decoded JSON."""
        return cls(
            name=data["name"],
            value=float(data["value"]),
            units=data.get("units"),
            higher_is_better=data.get("higher_is_better"),
        )


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark case and its metrics."""

    name: str
    reference: BenchmarkReference
    metrics: tuple[BenchmarkMetric, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate case fields."""
        if not self.name:
            raise ValueError("case name must not be empty")
        if not self.metrics:
            raise ValueError("case must contain at least one metric")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "name": self.name,
            "reference": self.reference.to_dict(),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkCase":
        """Build a case from decoded JSON."""
        return cls(
            name=data["name"],
            reference=BenchmarkReference.from_dict(data["reference"]),
            metrics=tuple(
                BenchmarkMetric.from_dict(metric) for metric in data["metrics"]
            ),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class BenchmarkReport:
    """Versioned report for reproducible benchmark output."""

    cases: tuple[BenchmarkCase, ...]
    generated_at: str = field(default_factory=lambda: _utc_now_iso())
    schema_version: str = "1.0"
    git_commit: str | None = field(default_factory=lambda: current_git_commit())
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate report fields."""
        if not self.cases:
            raise ValueError("report must contain at least one benchmark case")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "git_commit": self.git_commit,
            "metadata": self.metadata,
            "cases": [case.to_dict() for case in self.cases],
        }

    def to_json(self, path: str | Path) -> None:
        """Write the report as deterministic, indented JSON."""
        destination = Path(path)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkReport":
        """Build a report from decoded JSON."""
        return cls(
            schema_version=data.get("schema_version", "1.0"),
            generated_at=data["generated_at"],
            git_commit=data.get("git_commit"),
            metadata=dict(data.get("metadata", {})),
            cases=tuple(BenchmarkCase.from_dict(case) for case in data["cases"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "BenchmarkReport":
        """Read a benchmark report from JSON."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def current_git_commit() -> str | None:
    """Return the current git commit hash when available."""
    git_dir = _find_git_dir(Path.cwd())
    if git_dir is None:
        return None
    head = git_dir / "HEAD"
    try:
        value = head.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if value.startswith("ref: "):
        ref_path = git_dir / value.removeprefix("ref: ")
        try:
            return ref_path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return _read_packed_ref(git_dir, value.removeprefix("ref: "))
    return value or None


def _find_git_dir(start: Path) -> Path | None:
    for directory in (start, *start.parents):
        candidate = directory / ".git"
        if candidate.is_dir():
            return candidate
        if candidate.is_file():
            try:
                text = candidate.read_text(encoding="utf-8").strip()
            except OSError:
                return None
            if text.startswith("gitdir: "):
                return (directory / text.removeprefix("gitdir: ")).resolve()
    return None


def _read_packed_ref(git_dir: Path, ref_name: str) -> str | None:
    try:
        lines = (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    suffix = f" {ref_name}"
    for line in lines:
        if line.startswith("#") or not line.endswith(suffix):
            continue
        return line.split(" ", maxsplit=1)[0]
    return None
