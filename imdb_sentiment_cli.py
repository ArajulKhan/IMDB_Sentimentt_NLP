#!/usr/bin/env python3
import argparse, html, re, sys
from pathlib import Path
import joblib

def clean_text(text):
    text=html.unescape(str(text))
    text=re.sub(r"<br\s*/?>"," ",text,flags=re.IGNORECASE)
    text=re.sub(r"<[^>]+>"," ",text)
    return re.sub(r"\s+"," ",text).strip()

def main():
    parser=argparse.ArgumentParser(description="Predict positive/negative sentiment for a movie review.")
    parser.add_argument("review",nargs="*",help="Review text; omit to enter interactive mode.")
    parser.add_argument("--model",default="imdb_sentiment_model.pkl",help="Path to saved model.")
    args=parser.parse_args()
    path=Path(args.model)
    if not path.exists():
        print(f"Model not found: {path.resolve()}",file=sys.stderr)
        print("Run the notebook, download imdb_sentiment_model.pkl, and place it beside cli.py.",file=sys.stderr)
        return 2
    # Only load joblib/pickle files from trusted sources.
    model=joblib.load(path)
    if args.review:
        reviews=[" ".join(args.review)]
    else:
        print("IMDB Sentiment CLI — type quit or exit to stop.")
        reviews=None
    def predict(raw):
        text=clean_text(raw)
        if not text:
            print("Please enter a non-empty review."); return
        label=model.predict([text])[0]
        print("Predicted sentiment:",label)
        if hasattr(model,"decision_function"):
            print(f"Decision score (not a probability): {float(model.decision_function([text])[0]):.4f}")
        elif hasattr(model,"predict_proba"):
            scores=model.predict_proba([text])[0]
            print("Class scores:",{str(c):round(float(p),4) for c,p in zip(model.classes_,scores)})
    if reviews is not None:
        predict(reviews[0]); return 0
    while True:
        try: raw=input("\\nEnter a movie review: ").strip()
        except (EOFError,KeyboardInterrupt):
            print("\\nGoodbye!"); return 0
        if raw.lower() in {"quit","exit"}:
            print("Goodbye!"); return 0
        predict(raw)

if __name__=="__main__":
    raise SystemExit(main())
