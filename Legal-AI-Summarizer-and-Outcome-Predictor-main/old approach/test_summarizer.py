import time
from summarizer import summarize_pdf
from utils import load_pdf_chunks

def main():
    print("📄 Running Legal Summarizer Test...\n")

    # Example PDF (replace with your test file)
    pdf_path = "data/Smt_Amrita_Choudhary_vs_The_State_Of_Madhya_Pradesh_on_26_October_2015.PDF"

    # Pre-check chunks
    chunks = load_pdf_chunks(pdf_path, chunk_size=1500, chunk_overlap=200)
    print(f"📑 Loaded {len(chunks)} chunks from {pdf_path}")

    if len(chunks) > 10:
        print("⚡ Pipeline chosen: MAP-REDUCE (large document)\n")
    else:
        print("⚡ Pipeline chosen: STUFF (small document)\n")

    # Run summarizer with timing
    try:
        start_time = time.time()
        summary = summarize_pdf(pdf_path)
        elapsed = time.time() - start_time

        print("\n✅ Generated FIRAC Summary:\n")
        print(summary)
        print("\n⏱️ Processing Time: {:.2f} seconds".format(elapsed))

    except Exception as e:
        print("\n❌ Error during summarization!")
        print("Error Message:", str(e))

if __name__ == "__main__":
    main()
