"""CATPart에 들어 있는 관리면 Geometrical Set을 콘솔에 표시한다."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad_import.management_surface import discover_management_groups


def com_type_name(value) -> str:
    try:
        return str(value._oleobj_.GetTypeInfo().GetDocumentation(-1)[0])
    except Exception:
        return type(value).__name__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("catpart", type=Path, nargs="+")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    catia = win32com.client.DispatchEx("CATIA.Application")
    catia.Visible = False
    try:
        for path in args.catpart:
            print(f"===== {path.name}", flush=True)
            document = catia.Documents.Open(str(path.resolve()))
            try:
                groups = discover_management_groups(document.Part)
                print(f"관리 그룹 {len(groups)}개", flush=True)
                for group in groups:
                    print(group.to_dict(), flush=True)
                    if args.verbose and group.group_object is not None:
                        for index in range(1, group.group_object.HybridShapes.Count + 1):
                            shape = group.group_object.HybridShapes.Item(index)
                            print(f"  shape: {shape.Name} / {com_type_name(shape)}", flush=True)
                        for boundary in group.boundary_objects:
                            print(f"  set: {boundary.Name}", flush=True)
                            for index in range(1, boundary.HybridShapes.Count + 1):
                                shape = boundary.HybridShapes.Item(index)
                                print(f"    shape: {shape.Name} / {com_type_name(shape)}", flush=True)
                                try:
                                    reference = document.Part.CreateReferenceFromObject(shape)
                                    measurable = document.GetWorkbench("SPAWorkbench").GetMeasurable(reference)
                                    print(f"      length: {float(measurable.Length):.3f}", flush=True)
                                    print(f"      points: {measurable.GetPointsOnCurve()}", flush=True)
                                except Exception as exc:
                                    print(f"      measurable error: {exc}", flush=True)
            finally:
                document.Close()
    finally:
        catia.Quit()
        pythoncom.CoUninitialize()


if __name__ == "__main__":
    main()
