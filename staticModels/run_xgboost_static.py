"""Convenience entry point for grouped static XGBoost evaluation."""
from static_grouped_evaluation import main

if __name__ == "__main__":
    import sys
    sys.argv.extend(["--model", "xgboost"])
    main()
