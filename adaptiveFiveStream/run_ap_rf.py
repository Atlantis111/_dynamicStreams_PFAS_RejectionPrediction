"""Run the five-stream adaptive performance-triggered random forest."""
from adaptive_five_stream_models import main

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.argv.extend(["--models", "AP-RF"])
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", str(Path(__file__).resolve().parent / "outputs_ap_rf")])
    main()
