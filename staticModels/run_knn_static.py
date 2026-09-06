"""Convenience entry point for grouped static KNN evaluation."""
from static_grouped_evaluation import main

if __name__ == "__main__":
    import sys
    sys.argv.extend(["--model", "knn"])
    main()
