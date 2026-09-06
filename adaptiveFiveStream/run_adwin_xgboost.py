"""Run ADWIN-XGBoost on the five scenario streams."""

from pathlib import Path
import sys

from adaptive_five_stream_models import main


if __name__ == "__main__":
    sys.argv.extend(["--models", "ADWIN-XGBoost"])
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", str(Path(__file__).resolve().parent / "outputs_adwin_xgboost")])
    main()
