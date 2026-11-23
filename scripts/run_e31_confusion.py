from src.e31_confusion import run


if __name__ == "__main__":
    result = run()
    print(f"E31-Confusion complete: {result['runtime_seconds']:.1f}s; report: docs/e31_confusion.md")
