"""磁盘上的每个子包都必须出现在 pyproject 的显式包列表里。

pyproject 用的是显式 `[tool.setuptools] packages = [...]`，不是自动发现。漏加一个
子包时 **wheel 照常构建成功**，只有装上去 import 才炸 —— 而那时通常已经在设备上了。

0.0.7a0 构建时就漏过一次：新增的 `voxedge/text/` 没进列表，而
`backends/jetson/matcha_trt.py` 已经 `from voxedge.text import zh_numbers`，
那个 wheel 装上去必然 ModuleNotFoundError。
"""
import pathlib
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _declared() -> set[str]:
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return set(cfg["tool"]["setuptools"]["packages"])


def _on_disk() -> set[str]:
    pkgs = set()
    for init in (ROOT / "voxedge").rglob("__init__.py"):
        rel = init.parent.relative_to(ROOT)
        if "__pycache__" in rel.parts or "tests" in rel.parts:
            continue
        pkgs.add(".".join(rel.parts))
    return pkgs


def test_every_package_on_disk_is_declared():
    missing = _on_disk() - _declared()
    assert not missing, (
        "这些子包在磁盘上存在但没写进 pyproject 的 packages 列表，"
        f"打出来的 wheel 会缺失它们：{sorted(missing)}"
    )


def test_no_declared_package_is_missing_from_disk():
    stale = _declared() - _on_disk()
    assert not stale, f"pyproject 里声明了但磁盘上不存在的包：{sorted(stale)}"
