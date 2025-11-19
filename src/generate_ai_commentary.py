import os
import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# ----------------------------
# Load environment & setup
# ----------------------------
project_root = Path(__file__).resolve().parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# ----------------------------
# AI Commentary Function
# ----------------------------
def generate_commentary(df):
    # Build the summary text to feed the AI
    summary_text = df.to_string()

    prompt = f"""
    Write a clear, concise professional performance attribution commentary
    based on this sector-level attribution summary:

    {summary_text}

    Structure the output with:
    - Overview
    - Top contributors
    - Largest detractors
    - Conclusion

    Keep it readable and suitable for an investment performance report.
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

# ----------------------------
# Main block (ONLY runs when executing THIS file directly)
# ----------------------------
if __name__ == "__main__":
    data_path = project_root / "outputs" / "sector_attribution_for_tableau.csv"
    df = pd.read_csv(data_path)

    commentary = generate_commentary(df)

    print("\n=== AI Commentary ===\n")
    print(commentary)

    # Save output
    out_path = project_root / "outputs" / "sector_commentary.txt"
    with open(out_path, "w") as f:
        f.write(commentary)

    print(f"\nCommentary saved to: {out_path}")