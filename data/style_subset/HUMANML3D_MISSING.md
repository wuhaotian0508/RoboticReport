# HumanML3D Missing

Expected root: `D:\roboticsreport\datasets\HumanML3D`

Missing required paths:
- D:\roboticsreport\datasets\HumanML3D\texts
- D:\roboticsreport\datasets\HumanML3D\train.txt
- D:\roboticsreport\datasets\HumanML3D\val.txt
- D:\roboticsreport\datasets\HumanML3D\test.txt

Place the official HumanML3D `texts/`, `train.txt`, `val.txt`, and `test.txt` files here, then rerun:

```bash
cd /d/roboticsreport
./MoConVQ/.venv/Scripts/python.exe scripts/filter_style_subset.py
```
