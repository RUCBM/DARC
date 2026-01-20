import argparse
import os

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=os.getenv("SENTENCE_TRANSFORMER_PATH"))
    parser.add_argument("--data_path", type=str, default=os.getenv("DATA_PATH"))
    parser.add_argument("--output_path", type=str, default=os.getenv("OUTPUT_PATH"))
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    if not args.model_path or not args.data_path or not args.output_path:
        raise ValueError("SENTENCE_TRANSFORMER_PATH, DATA_PATH, and OUTPUT_PATH are required.")

                               
               
                               
    data = pd.read_parquet(args.data_path).to_dict(orient="records")

    questions = [
        sample["prompt"][1]["content"]
        for sample in data
    ]

                               
                
                               
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model_path, device=device)

                               
                          
                               
    embeddings = model.encode(
        questions,
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,                                        
        show_progress_bar=True,
    )

                               
                                
                               
    threshold = args.threshold

    kept_indices = []
    kept_embeddings = []

    for idx, emb in tqdm(enumerate(embeddings), total=len(embeddings)):
        if len(kept_embeddings) == 0:
            kept_indices.append(idx)
            kept_embeddings.append(emb)
            continue

        sims = util.cos_sim(emb, torch.stack(kept_embeddings))[0]
        max_sim = sims.max().item()

        if max_sim < threshold:
            kept_indices.append(idx)
            kept_embeddings.append(emb)
                            

                               
                         
                               
    filtered_data = [data[i] for i in kept_indices]

    print(f"Original size: {len(data)}")
    print(f"Filtered size: {len(filtered_data)}")
    print(f"Removed: {len(data) - len(filtered_data)}")

                               
                        
                               
    df_filtered = pd.DataFrame(filtered_data)
    df_filtered.to_parquet(args.output_path, index=False)

    print(f"Saved filtered data to: {args.output_path}")


if __name__ == "__main__":
    main()
