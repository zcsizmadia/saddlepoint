"""Solve every MPS file in a directory with HiGHS and write reference.csv.

The tests compare saddlepoint's objective values against this file. Run it
wherever highspy is installed (for example a WSL venv):

    python scripts/highs_reference.py data/netlib

Output columns: name,status,objective (objective is empty unless optimal).
"""

import pathlib
import sys

import highspy


def solve(path: pathlib.Path) -> tuple[str, str]:
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("presolve", "on")
    h.readModel(str(path))
    h.run()
    status = h.modelStatusToString(h.getModelStatus()).lower()
    if status == "optimal":
        return "optimal", repr(h.getInfo().objective_function_value)
    if "infeasible" in status and "unbounded" not in status:
        return "infeasible", ""
    if "unbounded" in status:
        return "unbounded", ""
    return status.replace(",", ";"), ""


def main() -> None:
    folder = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/netlib")
    rows = ["name,status,objective"]
    for path in sorted(folder.glob("*.mps")):
        status, obj = solve(path)
        rows.append(f"{path.stem},{status},{obj}")
        print(rows[-1])
    (folder / "reference.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
